"""Module C - AI Predictive Maintenance."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from aeroready import config, csv_io, db, model, risk, ui, workorders

engine = ui.get_engine()
ui.page_header("Predictive Maintenance",
               "Component-level maintenance-risk scores with explanations and decision-support suggestions. "
               "Experimental: trained and evaluated on synthetic data only; not an airworthiness determination.")
ui.show_flash()

_m = engine.metrics
if engine.method != "ml":
    st.info(f"Risk is computed with the transparent rule-based score: on the validation split the ML model "
            f"(ROC-AUC {_m['val_roc_auc_ml']:.3f}) did not beat it ({_m['val_roc_auc_rules']:.3f}), so there is no "
            "demonstrated benefit from the more complex model on this synthetic data. The score is a weighted "
            "index, not a calibrated failure probability.")
else:
    st.caption("Scores are prototype maintenance-risk indicators from a model trained on synthetic data; they are "
               "not calibrated failure probabilities.")

scored = ui.scored_components()
wos = db.table("work_orders")
active = wos[wos["status"].isin(config.ACTIVE_WO_STATUSES)]
scored = scored.merge(active[["component_id", "wo_id", "status"]].rename(columns={"status": "wo_status"}),
                      on="component_id", how="left")

tab_rank, tab_whatif, tab_card = st.tabs(["Fleet risk ranking", "What-if assessment", "Model card & evaluation"])

# --------------------------------------------------------------------------- ranking
with tab_rank:
    f1, f2, f3, f4 = st.columns([2, 2, 2, 1.4], vertical_alignment="bottom")
    levels = f1.multiselect("Risk level", config.RISK_LEVELS, default=["Medium", "High", "Critical"])
    types = f2.multiselect("Component type", list(config.COMPONENT_TYPES))
    tails = f3.multiselect("Aircraft", sorted(scored["tail_no"].unique()))
    only_open = f4.toggle("Only without work order", value=False)
    view = scored[scored["risk_level"].isin(levels or config.RISK_LEVELS)]
    if types:
        view = view[view["component_type"].isin(types)]
    if tails:
        view = view[view["tail_no"].isin(tails)]
    if only_open:
        view = view[view["wo_id"].isna()]
    view = view.sort_values("risk_score", ascending=False)

    counts = scored["risk_level"].value_counts()
    ui.kpi_row([ui.kpi(f"{lvl} risk", int(counts.get(lvl, 0)), "components",
                       {"Low": "good", "Medium": "warn", "High": "serious", "Critical": "bad"}[lvl])
                for lvl in config.RISK_LEVELS])
    st.write("")

    if view.empty:
        ui.empty_state("No components match these filters.")
    else:
        sel = st.dataframe(
            view[["component_id", "tail_no", "component_type", "risk_score", "risk_level", "top_factors",
                  "recommended_action", "telemetry_status", "wo_id", "wo_status"]],
            hide_index=True, on_select="rerun", selection_mode="single-row", key="rank_table", height=380,
            column_config={
                "component_id": "Component", "tail_no": "Aircraft", "component_type": "Type",
                "risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.0f"),
                "risk_level": "Category", "top_factors": "Largest differences from reference",
                "recommended_action": "Proposed action", "telemetry_status": "Telemetry",
                "wo_id": "Work order", "wo_status": "WO status",
            })
        st.caption("Select a row to see the explanation and raise a work order. Telemetry 'Stale' or 'No data' means "
                   "the condition is not currently known - it does not mean the component is healthy.")
        rows = sel.selection.rows if sel and sel.selection else []
        if rows:
            comp = view.iloc[rows[0]].to_dict()
            feats = risk.to_features(pd.DataFrame([comp])).iloc[0].to_dict()
            a = risk.assess(engine, feats, comp.get("telemetry_status", "Fresh"))
            st.divider()
            st.markdown(f"#### {comp['component_id']} · {comp['component_type']} {comp['position']} on {comp['tail_no']}")
            c1, c2 = st.columns([2, 3], gap="medium")
            with c1:
                ui.assessment_card(a, engine.method_label, engine.score_semantics, comp.get("telemetry_status", "Fresh"))
            with c2:
                st.plotly_chart(ui.explanation_chart(a["explanation"]), config={"displayModeBar": False})
            with st.expander("Create maintenance work order", expanded=a["level"] in ("High", "Critical")):
                ui.work_order_panel(comp, a, key=f"rank_{comp['component_id']}")

    pending = scored[scored["risk_level"].isin(["High", "Critical"]) & scored["wo_id"].isna()]
    st.divider()
    b1, b2 = st.columns([3, 1], vertical_alignment="center")
    b1.markdown(f"**{len(pending)}** High/Critical component(s) have no work order. Bulk creation uses each "
                "component's proposed action and default spare part. It does not ground any aircraft - set "
                "grounding explicitly on a work order if your procedures require it.")
    if b2.button("Create all recommended work orders", disabled=pending.empty, width="stretch"):
        made, blocked = [], 0
        for r in pending.to_dict("records"):
            try:
                res = workorders.create_work_order(
                    r["tail_no"], r["component_id"], r["recommended_action"] or "Inspect & Service", r["risk_level"],
                    grounding=False, risk_score=r["risk_score"],
                    recommended_action=r["recommendation"], source="Predictive")
                made.append(res["wo_id"])
                blocked += 0 if res["parts_reserved"] else 1
            except workorders.WorkOrderError as exc:
                ui.flash("error", str(exc))
        ui.flash("success", f"Created {len(made)} work order(s): {', '.join(made)}."
                 + (f" {blocked} are awaiting parts." if blocked else ""))
        st.rerun()

# --------------------------------------------------------------------------- what-if
with tab_whatif:
    st.caption("Adjust any input to recompute the risk instantly. What-if values are not saved - use the "
               "Digital Twin to commit a simulated reading to the record.")
    labels = {r["component_id"]: ui.component_label(r) for r in scored.to_dict("records")}
    ids = scored.sort_values("risk_score", ascending=False)["component_id"].tolist()
    default = st.session_state.get("sel_comp")
    cid = st.selectbox("Component", ids, index=ids.index(default) if default in ids else 0,
                       format_func=labels.get, key="whatif_comp")
    comp = scored.set_index("component_id").loc[cid].to_dict() | {"component_id": cid}
    base = risk.to_features(pd.DataFrame([comp])).iloc[0].to_dict()

    left, right = st.columns([2, 3], gap="medium")
    with left:
        feats = {}
        for f in config.FEATURES:
            lo, hi, step = config.FEATURE_RANGES[f]
            val = float(base[f]) if pd.notna(base[f]) else float(engine.reference[f])
            val = min(max(val, lo), hi)
            if f == "prior_faults":
                feats[f] = st.slider(config.FEATURE_LABELS[f], int(lo), int(hi), int(val), 1, key=f"wi_{cid}_{f}")
            else:
                feats[f] = st.slider(config.FEATURE_LABELS[f], lo, hi, round(val / step) * step, step,
                                     key=f"wi_{cid}_{f}")
    a = risk.assess(engine, feats)
    base_a = risk.assess(engine, base)
    with right:
        ui.assessment_card(a, engine.method_label, engine.score_semantics)
        delta = a["score"] - base_a["score"]
        if abs(delta) >= 0.1:
            st.caption(f"Recorded condition scores {base_a['score']:.0f} ({base_a['level']}); "
                       f"what-if changes the score by {delta:+.1f} points.")
        st.plotly_chart(ui.explanation_chart(a["explanation"]), config={"displayModeBar": False})
    with st.expander("Create a work order from this assessment"):
        ui.work_order_panel(comp, a, key=f"wi_{cid}")

# --------------------------------------------------------------------------- model card
with tab_card:
    m = engine.metrics
    st.markdown(
        f"**Active method:** {engine.method_label}  \n"
        f"**Target:** synthetic label 'unscheduled removal within the next 50 flying hours'.  \n"
        f"**Data:** {engine.n_train + engine.n_val + engine.n_test:,} reproducibly generated synthetic component "
        f"records (seed {config.RANDOM_SEED}); stratified split {engine.n_train:,} train / {engine.n_val:,} "
        f"validation / {engine.n_test:,} test.  \n"
        f"**Procedure:** fit on train; choose ML or rules on **validation** ROC-AUC; refit the ML model on "
        f"train+validation; evaluate the frozen choice once on the test split and report both methods. "
        f"Rule: {m['selection_rule']}.  \n"
        f"**Trained:** {engine.trained_at} · scikit-learn {engine.sklearn_version}")
    st.warning(
        "These metrics measure how well each method recovers the rules of the synthetic data generator. "
        "They say nothing about real-world failure prediction, which would require real labelled "
        "maintenance and health-monitoring data and independent validation. The score is not a calibrated "
        "probability.")
    st.markdown("**Method comparison**")
    cmp_df = pd.DataFrame([
        {"Method": "Monotonic gradient boosting (ML)", "Validation ROC-AUC": m["val_roc_auc_ml"],
         "Test ROC-AUC": m["ml_test"]["roc_auc"], "Test avg. precision": m["ml_test"]["average_precision"],
         "Test Brier": m["ml_test"]["brier"], "Active": engine.method == "ml"},
        {"Method": "Rule-based weighted index", "Validation ROC-AUC": m["val_roc_auc_rules"],
         "Test ROC-AUC": m["rules_test"]["roc_auc"], "Test avg. precision": m["rules_test"]["average_precision"],
         "Test Brier": m["rules_test"]["brier"], "Active": engine.method == "rules"},
    ])
    st.dataframe(cmp_df, hide_index=True, column_config={
        c: st.column_config.NumberColumn(c, format="%.3f")
        for c in ["Validation ROC-AUC", "Test ROC-AUC", "Test avg. precision", "Test Brier"]})
    st.caption("Brier score is only meaningful for calibrated probabilities; the rule-based index is not one. "
               "The method was selected using the validation column, never the test column.")
    diff = m["ml_test"]["roc_auc"] - m["rules_test"]["roc_auc"]
    st.info(f"On the held-out synthetic test set the ML model's ROC-AUC is {abs(diff):.3f} "
            f"{'above' if diff >= 0 else 'below'} the rule-based score - "
            + ("comparable performance, so the explainable rule score is a defensible choice."
               if abs(diff) < 0.02 else "a material difference."))
    ui.kpi_row([
        ui.kpi(f"Active method ROC-AUC (test)", f"{m['roc_auc']:.3f}", f"{'ML' if engine.method == 'ml' else 'rules'}"),
        ui.kpi("Average precision (test)", f"{m['average_precision']:.3f}", f"prevalence {m['prevalence_test']:.1%}"),
        ui.kpi("Precision / recall at High", f"{m['precision']:.2f} / {m['recall']:.2f}", f"threshold {m['threshold']}"),
        ui.kpi("Validation ROC-AUC (ML vs rules)", f"{m['val_roc_auc_ml']:.3f} / {m['val_roc_auc_rules']:.3f}",
               "used for selection"),
    ])
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=m["roc_fpr"], y=m["roc_tpr"], mode="lines", name="ML model", line=dict(width=2, color=ui.SERIES[0]))
        fig.add_scatter(x=m["rules_roc_fpr"], y=m["rules_roc_tpr"], mode="lines", name="Rule-based score",
                        line=dict(width=2, color=ui.SERIES[1]))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(width=1, dash="dot", color=ui.MUTED))
        fig.update_layout(title="ROC curves (held-out synthetic test set)", height=320,
                          xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(fig, config={"displayModeBar": False})
    with c2:
        imp = engine.importances.sort_values("importance")
        fig = go.Figure(go.Bar(x=imp["importance"], y=imp["label"], orientation="h", marker_color=ui.SERIES[0],
                               error_x=dict(type="data", array=imp["std"], color=ui.MUTED),
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_layout(title="ML model: permutation importance (drop in test ROC-AUC)", height=320)
        st.plotly_chart(fig, config={"displayModeBar": False})
    c3, c4 = st.columns(2, gap="medium")
    with c3:
        st.markdown(f"**Operating points by risk band** (held-out test set, active method: {'ML' if engine.method == 'ml' else 'rules'})")
        bt = pd.DataFrame(m["by_threshold"])
        st.dataframe(bt, hide_index=True, column_config={
            "threshold": st.column_config.NumberColumn("Score threshold", format="%.2f"),
            "flags_at_or_above": "Flags at or above",
            "flagged_pct": st.column_config.NumberColumn("Share flagged", format="percent"),
            "precision": st.column_config.NumberColumn("Precision", format="%.2f"),
            "recall": st.column_config.NumberColumn("Recall", format="%.2f"),
            "f1": st.column_config.NumberColumn("F1", format="%.2f")})
        st.caption("Lower score thresholds catch more synthetic failures (recall) at the cost of more false alarms (precision).")
    with c4:
        cm = m["confusion"]
        st.markdown(f"**Confusion matrix at the High threshold ({m['threshold']})**")
        st.dataframe(pd.DataFrame(cm, index=["Actual: no failure", "Actual: failure"],
                                  columns=["Predicted: no", "Predicted: yes"]))
        st.caption(f"Accuracy {m['accuracy']:.3f} vs. {m['majority_baseline_accuracy']:.3f} for always predicting "
                   "'no failure' - accuracy alone is misleading with imbalanced classes.")
    with st.expander("Explanation method and limitations"):
        st.markdown(
            "- **Global importance:** permutation importance on the held-out test split.\n"
            "- **Per-prediction explanation:** each feature is reset to the median of healthy training "
            "components (a reference profile); the resulting change in the score is shown for that feature. "
            "This describes the model's behaviour relative to a reference - it does not prove that a sensor "
            "caused the risk. Changes do not sum exactly to the score because features interact.\n"
            "- **Rule-based score:** weighted, clipped distance of each feature from a nominal value "
            f"(weights {', '.join(f'{config.FEATURE_LABELS[k]} {v[2]:.2f}' for k, v in model.RULE_SPEC.items())}).\n"
            "- Sensor values, limits and failure mechanisms are invented; the model is not physics-based.\n"
            "- Categories on the 0-1 score: Low < 0.20 ≤ Medium < 0.45 ≤ High < 0.70 ≤ Critical. The score is "
            "not a validated failure probability.")
        st.download_button("Download synthetic training data (CSV)",
                           csv_io.to_csv_bytes(model.generate_training_data()),
                           "aeroready_synthetic_training_data.csv", "text/csv")
