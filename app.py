"""Streamlit app for the steel plate fault models. Only loads saved models, no training."""
from __future__ import annotations

import json

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from plate_triage.dataset import FAULT_LABELS, HOLDOUT_CSV, REPO_ROOT, TARGET, measurement_columns
from plate_triage.grading import METRIC_NAMES, grade

MODEL_DIR = REPO_ROOT / "model"

st.set_page_config(page_title="Plate Defect Triage Desk", page_icon="🔩", layout="wide")


@st.cache_resource
def load_scorecard() -> dict:
    return json.loads((MODEL_DIR / "scorecard.json").read_text())


@st.cache_resource
def load_models() -> dict:
    entries = load_scorecard()["models"]
    return {title: joblib.load(MODEL_DIR / f"{e['slug']}.joblib") for title, e in entries.items()}


def find_batch_problems(batch: pd.DataFrame) -> list[str]:
    # collect all problems so the user sees them at once
    required = measurement_columns()
    missing = [column for column in [*required, TARGET] if column not in batch.columns]
    if missing:
        return [f"Missing column(s): {', '.join(missing)}"]

    problems = []
    non_numeric = [c for c in required if not pd.api.types.is_numeric_dtype(batch[c])]
    if non_numeric:
        problems.append(f"Non-numeric values in: {', '.join(non_numeric)}")
    if batch[[*required, TARGET]].isna().any(axis=None):
        problems.append("The file has blank cells. Every plate needs all values filled in.")
    unknown = sorted(set(batch[TARGET].dropna().astype(str)) - set(FAULT_LABELS))
    if unknown:
        problems.append(
            f"Unknown {TARGET} value(s): {', '.join(unknown)}. "
            f"Expected one of: {', '.join(FAULT_LABELS)}"
        )
    return problems


@st.cache_data(show_spinner="Scoring plates with the saved models...")
def score_batch(batch: pd.DataFrame) -> dict:
    # score with all models once, so switching models is instant
    X, y = batch[measurement_columns()], batch[TARGET]
    results = {}
    for title, model in load_models().items():
        predicted = model.predict(X)
        probabilities = model.predict_proba(X)
        results[title] = {
            "predicted": predicted,
            "probabilities": probabilities,
            "classes": model.classes_,
            "metrics": grade(y, predicted, probabilities, model.classes_),
        }
    return results


def confusion_heatmap(y_true, y_pred) -> alt.LayerChart:
    # colour by row share, otherwise the small classes are hard to see
    counts = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=FAULT_LABELS),
        index=FAULT_LABELS,
        columns=FAULT_LABELS,
    )
    cells = counts.rename_axis("True fault").reset_index().melt(
        "True fault", var_name="Predicted", value_name="Plates"
    )
    totals = cells.groupby("True fault")["Plates"].transform("sum").clip(lower=1)
    cells["Share"] = cells["Plates"] / totals

    base = alt.Chart(cells).encode(
        x=alt.X("Predicted:N", sort=FAULT_LABELS, axis=alt.Axis(labelAngle=-35, labelOverlap=False, ticks=False)),
        y=alt.Y("True fault:N", sort=FAULT_LABELS, axis=alt.Axis(ticks=False)),
    )
    squares = base.mark_rect(stroke="#fbfaf7", strokeWidth=2, cornerRadius=3).encode(
        color=alt.Color(
            "Share:Q",
            scale=alt.Scale(domain=[0, 1], range=["#edf2f6", "#123f63"]),
            legend=alt.Legend(title="Share of true fault", format=".0%"),
        ),
        tooltip=[
            "True fault:N",
            "Predicted:N",
            "Plates:Q",
            alt.Tooltip("Share:Q", format=".1%", title="Share of true fault"),
        ],
    )
    numbers = base.mark_text(fontSize=13).encode(
        text="Plates:Q",
        color=alt.condition("datum.Share > 0.5", alt.value("#ffffff"), alt.value("#23282d")),
    )
    return (squares + numbers).properties(height=380)


def show_shift_report(batch: pd.DataFrame, result: dict) -> None:
    columns = st.columns(len(METRIC_NAMES))
    for column, name in zip(columns, METRIC_NAMES):
        value = result["metrics"][name]
        column.metric(name, "n/a" if pd.isna(value) else f"{value:.3f}")
    st.caption("Precision, recall and F1 are macro averages. AUC is macro one-vs-rest.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Confusion matrix")
        st.altair_chart(
            confusion_heatmap(batch[TARGET], result["predicted"]), use_container_width=True
        )
    with right:
        st.subheader("Classification report")
        report = classification_report(
            batch[TARGET], result["predicted"], output_dict=True, zero_division=0
        )
        table = pd.DataFrame(report).T.drop(index="accuracy").rename(
            columns={"precision": "Prec.", "recall": "Rec.", "f1-score": "F1", "support": "Plates"}
        )
        table["Plates"] = table["Plates"].astype(int)
        st.dataframe(
            table.style.format({"Prec.": "{:.3f}", "Rec.": "{:.3f}", "F1": "{:.3f}"}),
            use_container_width=True,
            height=352,
        )


def show_lineup(results: dict, is_holdout: bool) -> None:
    board = pd.DataFrame({title: r["metrics"] for title, r in results.items()}).T[METRIC_NAMES]
    board.index.name = "ML Model Name"
    st.subheader("All five models on this batch")
    st.dataframe(
        board.style.format("{:.3f}", na_rep="n/a").highlight_max(
            axis=0, props="font-weight:700; background-color:#f3dfd2"
        ),
        use_container_width=True,
    )
    st.caption("Shading marks the best model for each metric.")

    metric = st.selectbox("Compare models on", METRIC_NAMES, index=METRIC_NAMES.index("MCC"))
    bars = board[[metric]].reset_index()
    chart = (
        alt.Chart(bars)
        .mark_bar(color="#123f63", cornerRadiusEnd=4, size=22)
        .encode(
            x=alt.X(
                f"{metric}:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(grid=False, tickCount=5),
            ),
            y=alt.Y(
                "ML Model Name:N",
                sort="-x",
                title=None,
                axis=alt.Axis(ticks=False, labelLimit=220),
            ),
            tooltip=["ML Model Name:N", alt.Tooltip(f"{metric}:Q", format=".3f")],
        )
        .properties(height=210)
    )
    st.altair_chart(chart, use_container_width=True)

    if is_holdout:
        st.subheader("How much of the gap is real?")
        st.write(
            "Scores saved at training time. Overlapping intervals mean the holdout "
            "cannot separate those models with confidence."
        )
        saved = load_scorecard()["models"]
        st.dataframe(
            pd.DataFrame(
                {
                    title: {
                        "Train MCC": entry["train_mcc"],
                        "5-fold CV MCC": entry["cv_mcc"],
                        "Holdout MCC": entry["holdout"]["MCC"],
                        "95% interval, low": entry["holdout_mcc_95ci"][0],
                        "95% interval, high": entry["holdout_mcc_95ci"][1],
                    }
                    for title, entry in saved.items()
                }
            ).T.style.format("{:.3f}"),
            use_container_width=True,
        )


def build_plate_log(batch: pd.DataFrame, results: dict, title: str) -> pd.DataFrame:
    chosen = results[title]
    predicted = chosen["predicted"]
    votes = np.column_stack([r["predicted"] for r in results.values()])
    rows = np.arange(len(batch))
    return pd.DataFrame(
        {
            "Plate": rows + 1,
            "True fault": batch[TARGET].to_numpy(),
            "Predicted": predicted,
            "Confidence": chosen["probabilities"][
                rows, np.searchsorted(chosen["classes"], predicted)
            ],
            "Models agreeing": (votes == predicted[:, None]).sum(axis=1),
            "Correct": predicted == batch[TARGET].to_numpy(),
        }
    )


def show_triage(batch: pd.DataFrame, results: dict, title: str) -> None:
    st.subheader("Consensus triage")
    st.write(
        f"**{title}** makes the call, and the other four act as a second opinion. Plates "
        "with enough backing are cleared automatically; the rest go to a human inspector."
    )
    log = build_plate_log(batch, results, title)
    needed = st.slider("Models that must agree (including the chosen one)", 1, 5, 4)
    cleared = log["Models agreeing"] >= needed

    def hit_rate(mask: pd.Series) -> str:
        return f"{log.loc[mask, 'Correct'].mean():.1%}" if mask.any() else "n/a"

    first, second, third = st.columns(3)
    first.metric("Auto-cleared plates", f"{cleared.mean():.1%}")
    second.metric("Accuracy when auto-cleared", hit_rate(cleared))
    third.metric("Accuracy when sent to inspector", hit_rate(~cleared))

    log["Route"] = np.where(cleared, "Auto-clear", "Inspector")
    view = st.radio("Show", ["All plates", "Inspector queue", "Wrong calls"], horizontal=True)
    shown = {
        "All plates": log,
        "Inspector queue": log[~cleared],
        "Wrong calls": log[~log["Correct"]],
    }[view]
    st.dataframe(
        shown,
        hide_index=True,
        use_container_width=True,
        column_config={"Confidence": st.column_config.NumberColumn(format="%.2f")},
    )
    st.download_button(
        "Download this view as CSV",
        shown.to_csv(index=False).encode(),
        file_name="plate_triage_log.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("Plate Defect Triage Desk")
    st.write(
        "Grades a batch of inspected steel plates against the saved fault classifiers. "
        "Models are loaded from disk and are never retrained on your file."
    )

    with st.sidebar:
        st.header("Batch")
        uploaded = st.file_uploader("Upload a test CSV", type="csv")
        st.download_button(
            "Download sample test_data.csv",
            HOLDOUT_CSV.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
        )
        st.caption(f"Needs all 27 measurement columns and a `{TARGET}` column.")
        st.header("Model")
        title = st.selectbox("Classifier", list(load_models()))

    try:
        batch = pd.read_csv(uploaded if uploaded is not None else HOLDOUT_CSV)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        st.error(f"That file could not be read as a CSV: {error}")
        st.stop()

    problems = find_batch_problems(batch)
    if problems:
        for problem in problems:
            st.error(problem)
        st.stop()

    source = uploaded.name if uploaded is not None else "bundled test_data.csv (holdout)"
    st.info(f"Batch: **{source}**, {len(batch)} plates")

    results = score_batch(batch)
    report_tab, lineup_tab, triage_tab = st.tabs(
        ["Shift report", "Model line-up", "Consensus triage"]
    )
    with report_tab:
        st.caption(f"Showing {title}")
        show_shift_report(batch, results[title])
    with lineup_tab:
        show_lineup(results, is_holdout=uploaded is None)
    with triage_tab:
        show_triage(batch, results, title)


main()
