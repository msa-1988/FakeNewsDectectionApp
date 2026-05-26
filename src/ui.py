from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from .config import CLASS_LABELS
from .inference import load_bundle, load_dashboard_payload, predict_article, predict_batch


@st.cache_resource
def get_bundle() -> dict:
    return load_bundle()


@st.cache_data
def get_dashboard_payload() -> dict:
    return load_dashboard_payload()


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f5f1e8;
            --panel: rgba(255, 251, 244, 0.92);
            --ink: #142023;
            --muted: #5d6a6f;
            --line: rgba(20, 32, 35, 0.12);
            --accent: #0f766e;
            --accent-soft: #d8f2ef;
            --alert: #b45309;
            --danger: #9f1239;
            --shadow: 0 18px 40px rgba(20, 32, 35, 0.08);
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(255, 214, 165, 0.38), transparent 28%),
                radial-gradient(circle at top right, rgba(151, 227, 214, 0.28), transparent 25%),
                linear-gradient(180deg, #fcf8f1 0%, var(--bg) 100%);
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        .block-container {
            max-width: 1160px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .hero, .panel, .verdict, .signal-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 24px;
            box-shadow: var(--shadow);
        }
        .hero {
            padding: 1.7rem 1.8rem 1.4rem 1.8rem;
            margin-bottom: 1.1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.3rem;
            line-height: 1.02;
            letter-spacing: -0.03em;
            color: var(--ink);
        }
        .hero p {
            margin: 0.8rem 0 0 0;
            color: var(--muted);
            max-width: 55rem;
        }
        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }
        .badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.34rem 0.68rem;
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(15, 118, 110, 0.15);
            color: var(--accent);
            font-size: 0.82rem;
            font-weight: 700;
        }
        .panel {
            padding: 1.15rem 1.15rem 1rem 1.15rem;
            margin-bottom: 1rem;
        }
        .subtle {
            color: var(--muted);
            font-size: 0.94rem;
        }
        .verdict {
            padding: 1rem 1.1rem 0.9rem 1.1rem;
            margin-bottom: 1rem;
        }
        .verdict-tag {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.28rem 0.6rem;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.03em;
        }
        .signal-card {
            padding: 0.8rem 0.9rem;
            min-height: 250px;
        }
        .signal-card h4 {
            margin-top: 0;
            margin-bottom: 0.6rem;
        }
        .token {
            display: inline-flex;
            border-radius: 999px;
            padding: 0.26rem 0.58rem;
            margin: 0 0.35rem 0.38rem 0;
            background: rgba(15, 118, 110, 0.10);
            color: var(--accent);
            font-size: 0.82rem;
            font-weight: 700;
        }
        .token-danger {
            background: rgba(159, 18, 57, 0.10);
            color: var(--danger);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(metrics: dict, metadata: dict) -> None:
    dataset_rows = metadata.get("dataset_rows", "n/a")
    best_model = metadata.get("model_name", "linear_svc")
    st.markdown(
        f"""
        <section class="hero">
            <h1>Fake News Detection Studio</h1>
            <p>
                A stronger fake-news classifier with separated title/body modeling,
                cross-validated model selection, explainable token-level signals, and
                a polished Streamlit interface for article screening.
            </p>
            <div class="badge-row">
                <span class="badge">Model: {best_model}</span>
                <span class="badge">Rows: {dataset_rows}</span>
                <span class="badge">Test F1: {metrics.get("test_f1_macro", 0):.3f}</span>
                <span class="badge">Test ROC-AUC: {metrics.get("test_roc_auc", 0):.3f}</span>
                <span class="badge">Explainable signals</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sample_picker(samples: list[dict]) -> tuple[str, str]:
    options = {"Custom input": None}
    for idx, sample in enumerate(samples, start=1):
        label = CLASS_LABELS[int(sample["label"])]
        title = str(sample["title"])[:80]
        options[f"Sample {idx} · {label}"] = sample

    selected = st.selectbox("Load a sample article", list(options.keys()))
    sample = options[selected]
    if sample is None:
        return "", ""
    return str(sample["title"]), str(sample["body"])


def render_verdict(result: dict) -> None:
    tag = "High-risk misinformation signal" if result["label"] == 1 else "More credible news pattern"
    st.markdown(
        f"""
        <div class="verdict">
            <span class="verdict-tag">{tag}</span>
            <h3 style="margin-bottom:0.2rem;">{result['label_name']}</h3>
            <p class="subtle" style="margin-top:0;">
                Margin-based confidence indicator: <b>{result['confidence']:.1%}</b>
                · Decision score: <b>{result['decision_score']:.3f}</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_panel(title: str, items: list[dict], danger: bool = False) -> None:
    token_class = "token token-danger" if danger else "token"
    badges = "".join(
        [
            f"<span class=\"{token_class}\">{item['feature'].split(': ', 1)[-1]} · {item['contribution']:.3f}</span>"
            for item in items
        ]
    )
    st.markdown(
        f"""
        <div class="signal-card">
            <h4>{title}</h4>
            <div>{badges or '<span class="subtle">No strong lexical signals surfaced.</span>'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_validation_tab(payload: dict) -> None:
    metrics = payload["metrics"]
    selection = payload["selection"]
    confusion = payload["confusion"]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Accuracy", f"{metrics.get('test_accuracy', 0):.3f}")
    metric_cols[1].metric("F1 Macro", f"{metrics.get('test_f1_macro', 0):.3f}")
    metric_cols[2].metric("Precision", f"{metrics.get('test_precision_macro', 0):.3f}")
    metric_cols[3].metric("ROC-AUC", f"{metrics.get('test_roc_auc', 0):.3f}")

    st.markdown("#### Model selection")
    if selection:
        st.dataframe(selection, use_container_width=True, hide_index=True)

    st.markdown("#### Confusion matrix")
    if confusion:
        labels = confusion.get("labels", [])
        matrix = confusion.get("matrix", [])
        if labels and matrix:
            st.dataframe(
                {
                    "Actual / Predicted": labels,
                    labels[0]: [row[0] for row in matrix],
                    labels[1]: [row[1] for row in matrix],
                },
                use_container_width=True,
                hide_index=True,
            )


def render_model_card_tab(payload: dict, metadata: dict) -> None:
    summary_cols = st.columns(2)
    class_balance = metadata.get("class_balance", {})
    with summary_cols[0]:
        st.markdown("#### Model summary")
        st.markdown(f"- Classifier: `{metadata.get('model_name', 'linear_svc')}`")
        st.markdown(f"- Training rows: `{metadata.get('dataset_rows', 'n/a')}`")
        st.markdown(f"- Trained at: `{metadata.get('trained_at_utc', 'n/a')}`")
        st.markdown("#### Design notes")
        for note in metadata.get("notes", []):
            st.markdown(f"- {note}")
    with summary_cols[1]:
        st.markdown("#### Class balance")
        if class_balance:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Label": list(class_balance.keys()),
                        "Rows": list(class_balance.values()),
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        st.markdown("#### Feature families")
        for feature_name in metadata.get("features", []):
            st.markdown(f"- {feature_name}")

    top_terms = payload["top_terms"]
    left, right = st.columns(2)
    with left:
        st.markdown("#### Top fake-leaning signals")
        for item in top_terms.get("fake_signals", []):
            st.markdown(f"- `{item['feature']}` ({item['weight']:.3f})")
    with right:
        st.markdown("#### Top real-leaning signals")
        for item in top_terms.get("real_signals", []):
            st.markdown(f"- `{item['feature']}` ({item['weight']:.3f})")

    st.info(
        "This model is a text classifier, not a fact-checking engine. It should support triage and review, not replace verification."
    )


def render_batch_tab(bundle: dict) -> None:
    st.subheader("Batch screening")
    st.caption(
        "Upload a CSV with `title` plus either `body` or `text` columns to score multiple articles at once."
    )
    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Recommended for newsroom triage, moderation queues, or validation demos.",
    )

    if uploaded_file is None:
        st.markdown(
            """
            <div class="panel">
                <h3 style="margin-top:0;">Expected CSV schema</h3>
                <p class="subtle">At minimum, include a headline and article body.</p>
                <pre style="white-space:pre-wrap;">title,body
Example headline,Example article body...</pre>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    try:
        batch_frame = pd.read_csv(uploaded_file)
        results = predict_batch(bundle, batch_frame)
    except Exception as exc:
        st.error(str(exc))
        return

    verdict_counts = results["predicted_label"].value_counts()
    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows scored", f"{len(results)}")
    metric_cols[1].metric("Likely fake", f"{int(verdict_counts.get(CLASS_LABELS[1], 0))}")
    metric_cols[2].metric("Likely real", f"{int(verdict_counts.get(CLASS_LABELS[0], 0))}")
    metric_cols[3].metric("Mean confidence", f"{results['confidence'].mean():.1%}")

    preview_columns = [
        column
        for column in ["title", "predicted_label", "confidence", "decision_score"]
        if column in results.columns
    ]
    st.dataframe(results[preview_columns], use_container_width=True, hide_index=True)
    st.download_button(
        "Download scored CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="fake-news-screening-results.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Fake News Detection Studio",
        page_icon="📰",
        layout="wide",
    )
    apply_theme()

    bundle = get_bundle()
    payload = get_dashboard_payload()
    metadata = bundle.get("metadata", {})
    metrics = payload.get("metrics", {})

    render_hero(metrics, metadata)

    left_col, right_col = st.columns([1.35, 0.95], gap="large")

    with left_col:
        tabs = st.tabs(["Analyze article", "Batch screening", "Validation", "Model card"])
        with tabs[0]:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("Article analysis")
            st.caption("Enter both title and body for the strongest signal. Samples come from the holdout test set.")

            sample_title, sample_body = render_sample_picker(payload.get("samples", []))
            title = st.text_input("Headline", value=sample_title, placeholder="Paste the article headline")
            body = st.text_area(
                "Article body",
                value=sample_body,
                height=280,
                placeholder="Paste the article body here...",
            )
            analyze = st.button("Analyze article", type="primary", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if analyze:
                if len(title.strip()) < 8 and len(body.strip()) < 120:
                    st.warning("Add a meaningful headline and a longer body excerpt before running the model.")
                else:
                    result = predict_article(bundle, title, body)
                    render_verdict(result)
                    signal_cols = st.columns(2)
                    with signal_cols[0]:
                        render_signal_panel("Signals pushing toward fake", result["explanation"]["supporting_fake"], danger=True)
                    with signal_cols[1]:
                        render_signal_panel("Signals pushing toward real", result["explanation"]["supporting_real"])
        with tabs[1]:
            render_batch_tab(bundle)
        with tabs[2]:
            render_validation_tab(payload)
        with tabs[3]:
            render_model_card_tab(payload, metadata)

    with right_col:
        st.markdown(
            """
            <div class="panel">
                <h3 style="margin-top:0;">How this version is stronger</h3>
                <p class="subtle">
                    This upgrade separates title and body features, benchmarks multiple classifiers,
                    records holdout metrics, supports batch screening, and exposes token-level
                    contributions instead of only a binary prediction.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="panel">
                <h3 style="margin-top:0;">Operational notes</h3>
                <ul class="subtle">
                    <li>The classifier is trained on historical labeled articles.</li>
                    <li>Author names are excluded from training to reduce source leakage.</li>
                    <li>The confidence score is margin-based, not a calibrated fact-check probability.</li>
                    <li>Use it for screening and analyst support, not final truth adjudication.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="panel">
                <h3 style="margin-top:0;">Quick checks</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"- Best model: `{metadata.get('model_name', 'linear_svc')}`")
        st.markdown(f"- Validation rows: `{metrics.get('test_size', 'n/a')}`")
        st.markdown(f"- Training rows: `{metadata.get('dataset_rows', 'n/a')}`")
        st.markdown(f"- Test accuracy: `{metrics.get('test_accuracy', 0):.3f}`")
        st.markdown(f"- Test F1 macro: `{metrics.get('test_f1_macro', 0):.3f}`")
