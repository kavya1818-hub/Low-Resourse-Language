"""
Low-Resource Hindi Sentiment Analysis - Interactive Results Dashboard
Case Study 02 | CSI4001 NLP & Computational Linguistics | VIT

Run from the project root (the folder that contains `results/`, `src/`, `dataset/`):

    streamlit run dashboard.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hindi Sentiment | Low-Resource NLP",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Paths (works if the file sits in the project root or inside src/)
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "results").exists() else HERE.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FINAL_FIGS = FIGURES / "final"
PROCESSED = ROOT / "dataset" / "processed"
MBERT_PATH = RESULTS / "mbert_final_model"
LORA_PATH = RESULTS / "mbert_lora_final_model"

# ----------------------------------------------------------------------------
# Results (taken from the case-study report; edit here if you re-run training)
# ----------------------------------------------------------------------------
MODELS = ["TF-IDF + Logistic Regression", "mBERT Fine-tuning", "mBERT + LoRA", "Qwen Zero-shot", "Qwen Few-shot"]

MAIN = pd.DataFrame(
    {
        "Model": MODELS,
        "Accuracy": [83.82, 82.66, 73.41, 61.85, 69.94],
        "Precision": [83.97, 82.73, 74.50, 76.27, 76.97],
        "Recall": [83.82, 82.66, 73.41, 61.85, 69.94],
        "F1-score": [83.79, 82.65, 73.14, 55.92, 67.92],
    }
)

LOW_RES = pd.DataFrame(
    {
        "Training samples": [100, 300, 500, 806],
        "Accuracy": [58.38, 80.35, 83.82, 86.71],
        "F1-score": [51.65, 80.29, 83.82, 86.71],
    }
)

CONFUSION = {  # rows = actual (Neg, Pos), cols = predicted (Neg, Pos)
    "TF-IDF + Logistic Regression": [[69, 17], [11, 76]],
    "mBERT Fine-tuning": [[73, 13], [17, 70]],
    "mBERT + LoRA": [[72, 14], [32, 55]],
}

DATASET = {"Positive": 580, "Negative": 572}
SPLITS = {"Training": 806, "Validation": 173, "Testing": 173}
LORA_TOTAL = 178_151_428
LORA_TRAINABLE = 296_450

COLORS = {
    "TF-IDF + Logistic Regression": "#4F46E5",
    "mBERT Fine-tuning": "#06B6D4",
    "mBERT + LoRA": "#10B981",
    "Qwen Zero-shot": "#F59E0B",
    "Qwen Few-shot": "#F43F5E",
}
METRICS = ["Accuracy", "Precision", "Recall", "F1-score"]
LABELS = ["Negative", "Positive"]

SAMPLES = {
    "Positive example": "यह फ़ोन बहुत शानदार है, कैमरा और बैटरी दोनों कमाल के हैं।",
    "Negative example": "सेवा बहुत खराब थी और डिलीवरी में काफ़ी देरी हुई।",
    "Mixed / tricky": "दाम थोड़ा ज़्यादा है लेकिन गुणवत्ता उम्मीद से बेहतर निकली।",
}

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px;}
    #MainMenu, footer {visibility: hidden;}
    .hero {
        background: linear-gradient(120deg, #312E81 0%, #4F46E5 55%, #06B6D4 100%);
        border-radius: 20px; padding: 30px 36px; color: #fff; margin-bottom: 1.4rem;
        box-shadow: 0 10px 30px rgba(79,70,229,.25);
    }
    .hero h1 {margin: 0 0 6px 0; font-size: 2rem; font-weight: 800; color: #fff;}
    .hero p {margin: 0; opacity: .92; font-size: 1.02rem;}
    .hero .tag {
        display: inline-block; background: rgba(255,255,255,.18); border-radius: 999px;
        padding: 3px 12px; font-size: .78rem; margin: 12px 8px 0 0; letter-spacing: .03em;
    }
    .kpi {
        border: 1px solid rgba(128,128,128,.25); border-radius: 16px; padding: 16px 20px;
        background: rgba(128,128,128,.07); height: 100%;
    }
    .kpi .l {opacity: .65; font-size: .74rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 600;}
    .kpi .v {font-size: 2rem; font-weight: 800; line-height: 1.25;}
    .kpi .s {opacity: .6; font-size: .82rem;}
    .callout {
        border-left: 4px solid #4F46E5; background: rgba(79,70,229,.08);
        border-radius: 8px; padding: 12px 16px; margin: 10px 0 4px 0; font-size: .95rem;
    }
    .callout.warn {border-left-color: #F59E0B; background: rgba(245,158,11,.10);}
    .section-title {font-size: 1.25rem; font-weight: 700; margin: 1.4rem 0 .3rem 0;}
    .pred-card {
        border-radius: 18px; padding: 22px 26px; text-align: center; color: #fff; margin-top: 6px;
    }
    .pred-pos {background: linear-gradient(120deg, #059669, #10B981);}
    .pred-neg {background: linear-gradient(120deg, #DC2626, #F43F5E);}
    .pred-card .big {font-size: 2rem; font-weight: 800;}
    div[data-testid="stSidebar"] .stRadio label {padding: 4px 0;}
</style>
""",
    unsafe_allow_html=True,
)


def kpi(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
        f'<div class="s">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def callout(text: str, warn: bool = False) -> None:
    st.markdown(f'<div class="callout{" warn" if warn else ""}">{text}</div>', unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def hero(title: str, subtitle: str, tags=()) -> None:
    chips = "".join(f'<span class="tag">{t}</span>' for t in tags)
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>{chips}</div>',
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title_font_size=16,
    )
    return fig


# ----------------------------------------------------------------------------
# Data / model helpers
# ----------------------------------------------------------------------------
def find_csv(keyword: str):
    if not PROCESSED.exists():
        return None
    for p in sorted(PROCESSED.glob("*.csv")):
        if keyword in p.name.lower():
            return p
    return None


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def detect_columns(df: pd.DataFrame):
    """Guess (hindi_col, english_col, label_col) from a processed CSV."""
    lower = {c.lower(): c for c in df.columns}
    hindi = next((lower[k] for k in ("hindi", "hindi_text", "text_hi", "hi", "text", "review_hi") if k in lower), None)
    eng = next((lower[k] for k in ("english", "english_text", "text_en", "en", "eng_review", "review_en") if k in lower), None)
    label = next((lower[k] for k in ("sentiment", "label", "labels", "target") if k in lower), None)
    return hindi, eng, label


def normalise_labels(series: pd.Series) -> pd.Series:
    def fix(v):
        s = str(v).strip().lower()
        if s in ("1", "positive", "pos"):
            return "Positive"
        if s in ("0", "negative", "neg"):
            return "Negative"
        return str(v)

    return series.map(fix)


@st.cache_resource(show_spinner="Loading Transformer model...")
def load_transformer(kind: str, path: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    if kind == "lora":
        from peft import PeftConfig, PeftModel

        cfg = PeftConfig.from_pretrained(path)
        base = AutoModelForSequenceClassification.from_pretrained(cfg.base_model_name_or_path, num_labels=2)
        model = PeftModel.from_pretrained(base, path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tok, model


def predict_transformer(tok, model, text: str):
    import torch

    enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0].tolist()
    return probs  # [P(negative), P(positive)]


@st.cache_resource(show_spinner="Fitting TF-IDF baseline on the training split...")
def fit_baseline():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    train_path = find_csv("train")
    if train_path is None:
        return None
    df = load_csv(str(train_path))
    hindi, _, label = detect_columns(df)
    if hindi is None or label is None:
        return None
    y = (normalise_labels(df[label]) == "Positive").astype(int)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=50000, sublinear_tf=True)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(vec.fit_transform(df[hindi].astype(str)), y)
    return vec, clf


def available_live_models():
    opts = {}
    if find_csv("train") is not None:
        opts["TF-IDF + Logistic Regression (re-fitted live)"] = ("baseline", None)
    if MBERT_PATH.exists():
        opts["mBERT Fine-tuning"] = ("mbert", str(MBERT_PATH))
    if LORA_PATH.exists():
        opts["mBERT + LoRA"] = ("lora", str(LORA_PATH))
    return opts


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 Hindi Sentiment")
    st.caption("Low-Resource NLP with Transformers & LLMs")
    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "📊 Model Comparison",
            "📉 Low-Resource Study",
            "⚡ LoRA Efficiency",
            "🤖 LLM Prompting",
            "🔮 Live Prediction",
            "🖼️ Figures Gallery",
            
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("**Task:** Binary sentiment (Hindi)")
    st.caption("**Dataset:** AI4Bharat IndicSentiment")
    st.caption("**Records:** 1,152 | Test: 173")

best_row = MAIN.loc[MAIN["Accuracy"].idxmax()]

# ============================================================================
# 1. OVERVIEW
# ============================================================================
if page == "🏠 Overview":
    hero(
        "Low-Resource Hindi Sentiment Analysis",
        "Comparing classical ML, multilingual Transformers, parameter-efficient LoRA and "
        "Large Language Model prompting on the same binary sentiment task.",
        ["mBERT", "LoRA / PEFT", "Qwen2.5-0.5B", "TF-IDF + LR", "IndicSentiment"],
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Best accuracy", f"{best_row['Accuracy']:.2f}%", best_row["Model"])
    with c2:
        kpi("Approaches compared", "5", "Classical, Transformer, PEFT, LLM x2")
    with c3:
        kpi("Dataset size", "1,152", "After cleaning & de-duplication")
    with c4:
        kpi("LoRA trainable params", f"{LORA_TRAINABLE / LORA_TOTAL * 100:.4f}%", f"{LORA_TRAINABLE:,} of {LORA_TOTAL:,}")

    section("Dataset")
    left, right = st.columns(2)
    with left:
        fig = px.pie(
            names=list(DATASET.keys()),
            values=list(DATASET.values()),
            hole=0.55,
            color=list(DATASET.keys()),
            color_discrete_map={"Positive": "#10B981", "Negative": "#F43F5E"},
            title="Sentiment distribution",
        )
        fig.update_traces(textinfo="label+value+percent")
        st.plotly_chart(style_fig(fig, 340), width="stretch")
    with right:
        fig = px.bar(
            x=list(SPLITS.keys()),
            y=list(SPLITS.values()),
            text=list(SPLITS.values()),
            color=list(SPLITS.keys()),
            color_discrete_sequence=["#4F46E5", "#06B6D4", "#F59E0B"],
            title="Stratified train / validation / test split",
            labels={"x": "Split", "y": "Records"},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, 340), width="stretch")

    section("Pipeline")
    p1, p2, p3, p4, p5 = st.columns(5)
    steps = [
        ("1 · Preprocess", "Remove missing & duplicate rows, stratified split"),
        ("2 · Baseline", "Char n-gram TF-IDF + Logistic Regression"),
        ("3 · Fine-tune", "bert-base-multilingual-cased on Hindi text"),
        ("4 · LoRA", "Train 0.17% of parameters with PEFT"),
        ("5 · LLM", "Qwen2.5-0.5B-Instruct, zero / few-shot"),
    ]
    for col, (t, d) in zip((p1, p2, p3, p4, p5), steps):
        with col:
            kpi(t, "", d)

    section("Key findings")
    callout(f"🏆 <b>The simplest model won.</b> TF-IDF + Logistic Regression reached "
            f"<b>{MAIN.loc[0, 'Accuracy']}%</b> accuracy, slightly ahead of full mBERT fine-tuning "
            f"(<b>{MAIN.loc[1, 'Accuracy']}%</b>) on this small binary dataset.")
    callout(f"⚡ <b>LoRA is extremely parameter-efficient</b> (only {LORA_TRAINABLE:,} trainable parameters) "
            f"but scored lower: <b>{MAIN.loc[2, 'Accuracy']}%</b> accuracy.")
    callout("🤖 <b>Few-shot prompting helps.</b> Giving Qwen four labelled examples lifted F1 from "
            "55.92% to 67.92%, though it still trails the trained classifiers.")
    callout("📈 <b>Data matters.</b> mBERT accuracy rises from 58.38% (100 samples) to 86.71% (806 samples).")

# ============================================================================
# 2. MODEL COMPARISON
# ============================================================================
elif page == "📊 Model Comparison":
    hero("Model Comparison", "Test-set performance of all five approaches (173 held-out Hindi reviews).")

    section("Metric explorer")
    chosen = st.multiselect("Metrics to display", METRICS, default=METRICS)
    if not chosen:
        st.info("Select at least one metric.")
    else:
        long = MAIN.melt(id_vars="Model", value_vars=chosen, var_name="Metric", value_name="Score")
        fig = px.bar(
            long, x="Model", y="Score", color="Metric", barmode="group", text="Score",
            color_discrete_sequence=["#4F46E5", "#06B6D4", "#10B981", "#F59E0B"],
            title="Scores by model",
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_yaxes(range=[0, 100], title="Score (%)")
        st.plotly_chart(style_fig(fig, 460), width="stretch")

    section("Results table")
    st.dataframe(
        MAIN,
        hide_index=True,
        width="stretch",
        column_config={
            m: st.column_config.ProgressColumn(m, min_value=0, max_value=100, format="%.2f%%") for m in METRICS
        },
    )

    left, right = st.columns(2)
    with left:
        section("Profile radar")
        fig = go.Figure()
        for _, r in MAIN.iterrows():
            vals = [r[m] for m in METRICS]
            fig.add_trace(
                go.Scatterpolar(
                    r=vals + [vals[0]], theta=METRICS + [METRICS[0]], name=r["Model"],
                    line=dict(color=COLORS[r["Model"]]), fill="toself", opacity=0.55,
                )
            )
        fig.update_layout(polar=dict(radialaxis=dict(range=[40, 100])))
        st.plotly_chart(style_fig(fig, 430), width="stretch")
    with right:
        section("Precision vs recall")
        fig = px.scatter(
            MAIN, x="Recall", y="Precision", color="Model", size=[22] * len(MAIN),
            color_discrete_map=COLORS, text="Model",
        )
        fig.update_traces(textposition="top center")
        fig.add_shape(type="line", x0=50, y0=50, x1=100, y1=100, line=dict(dash="dot", color="gray"))
        fig.update_xaxes(range=[55, 90])
        fig.update_yaxes(range=[55, 90])
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, 430), width="stretch")
        st.caption("Dotted line = precision equals recall. Qwen zero-shot sits far below it: "
                   "precise when it predicts positive, but it misses many positives.")

    section("Confusion matrices")
    cm_model = st.radio("Model", list(CONFUSION.keys()), horizontal=True)
    cm = CONFUSION[cm_model]
    c1, c2 = st.columns([1.1, 1])
    with c1:
        fig = px.imshow(
            cm, x=LABELS, y=LABELS, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"), title=cm_model,
        )
        st.plotly_chart(style_fig(fig, 400), width="stretch")
    with c2:
        tn, fp = cm[0]
        fn, tp = cm[1]
        total = tn + fp + fn + tp
        st.metric("Correct predictions", f"{tn + tp} / {total}")
        st.metric("False positives (Neg predicted Pos)", fp)
        st.metric("False negatives (Pos predicted Neg)", fn)
        if fn > fp * 1.8:
            callout("This model is biased toward predicting <b>Negative</b>: it misses many truly positive reviews.", warn=True)

# ============================================================================
# 3. LOW-RESOURCE
# ============================================================================
elif page == "📉 Low-Resource Study":
    hero("Low-Resource Learning Experiment",
         "How much labelled Hindi data does mBERT need? Evaluated on 100, 300, 500 and 806 training examples.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=LOW_RES["Training samples"], y=LOW_RES["Accuracy"], mode="lines+markers+text",
                             name="Accuracy", line=dict(color="#4F46E5", width=3), marker=dict(size=10),
                             text=LOW_RES["Accuracy"].map("{:.1f}".format), textposition="top center"))
    fig.add_trace(go.Scatter(x=LOW_RES["Training samples"], y=LOW_RES["F1-score"], mode="lines+markers+text",
                             name="F1-score", line=dict(color="#F43F5E", width=3, dash="dot"), marker=dict(size=10),
                             text=LOW_RES["F1-score"].map("{:.1f}".format), textposition="bottom center"))
    fig.update_xaxes(title="Number of training samples", tickvals=LOW_RES["Training samples"])
    fig.update_yaxes(title="Score (%)", range=[40, 95])
    fig.update_layout(title="mBERT performance vs training-set size")
    st.plotly_chart(style_fig(fig, 460), width="stretch")

    d = LOW_RES.copy()
    d["Share of full train set"] = (d["Training samples"] / 806 * 100).round(1)
    d["Accuracy gain (pts)"] = d["Accuracy"].diff().round(2)
    d["F1 gain (pts)"] = d["F1-score"].diff().round(2)

    left, right = st.columns([1.3, 1])
    with left:
        section("Detailed results")
        st.dataframe(d, hide_index=True, width="stretch")
    with right:
        section("What the curve says")
        jump = d.loc[1, "F1 gain (pts)"]
        callout(f"The biggest jump comes between 100 and 300 samples (<b>+{jump} F1 points</b>). "
                f"After that, gains shrink to a few points per step - diminishing returns.")
        callout(f"With only 100 samples, F1 is just <b>{d.loc[0, 'F1-score']}%</b>: the model is close to "
                f"guessing on one class.", warn=True)

    callout("ℹ️ This experiment is reported independently of the main benchmark, so its 806-sample score "
            "(86.71%) is not directly comparable with the 82.66% in the main comparison table.", warn=True)

# ============================================================================
# 4. LORA
# ============================================================================
elif page == "⚡ LoRA Efficiency":
    hero("Parameter-Efficient Fine-Tuning (LoRA)",
         "Low-Rank Adaptation trains a tiny set of extra weights while the multilingual BERT backbone stays frozen.")

    frozen = LORA_TOTAL - LORA_TRAINABLE
    full_ft = frozen  # backbone + classification head when fully fine-tuned
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Total parameters", f"{LORA_TOTAL:,}", "mBERT + LoRA adapters")
    with c2:
        kpi("Trainable (LoRA)", f"{LORA_TRAINABLE:,}", f"{LORA_TRAINABLE / LORA_TOTAL * 100:.4f}% of total")
    with c3:
        kpi("Reduction in trainable", f"{(1 - LORA_TRAINABLE / full_ft) * 100:.2f}%", "vs. full fine-tuning")
    with c4:
        kpi("Accuracy cost", f"{MAIN.loc[2, 'Accuracy'] - MAIN.loc[1, 'Accuracy']:.2f} pts", "LoRA vs. full mBERT")

    left, right = st.columns(2)
    with left:
        fig = px.pie(
            names=["Frozen", "Trainable (LoRA)"], values=[frozen, LORA_TRAINABLE], hole=0.6,
            color=["Frozen", "Trainable (LoRA)"],
            color_discrete_map={"Frozen": "#94A3B8", "Trainable (LoRA)": "#10B981"},
            title="Parameter split",
        )
        fig.update_traces(textinfo="percent", texttemplate="%{percent:.4%}" )
        st.plotly_chart(style_fig(fig, 400), width="stretch")
    with right:
        fig = go.Figure(
            go.Bar(
                x=["Full fine-tuning", "LoRA"], y=[full_ft, LORA_TRAINABLE],
                marker_color=["#06B6D4", "#10B981"],
                text=[f"{full_ft:,}", f"{LORA_TRAINABLE:,}"], textposition="outside",
            )
        )
        fig.update_yaxes(type="log", title="Trainable parameters (log scale)")
        fig.update_layout(title="Trainable parameters: full vs LoRA")
        st.plotly_chart(style_fig(fig, 400), width="stretch")

    section("Efficiency vs accuracy trade-off")
    trade = pd.DataFrame(
        {
            "Method": ["Full mBERT", "mBERT + LoRA"],
            "Trainable params": [full_ft, LORA_TRAINABLE],
            "F1-score": [MAIN.loc[1, "F1-score"], MAIN.loc[2, "F1-score"]],
        }
    )
    fig = px.scatter(trade, x="Trainable params", y="F1-score", color="Method", text="Method", log_x=True,
                     size=[30, 30], color_discrete_sequence=["#06B6D4", "#10B981"])
    fig.update_traces(textposition="top center")
    fig.update_yaxes(range=[60, 90])
    fig.update_layout(showlegend=False)
    st.plotly_chart(style_fig(fig, 380), width="stretch")
    callout("LoRA cuts trainable parameters by more than 99.8% but loses roughly 9.5 F1 points here. Its errors "
            "are mostly <b>false negatives</b> (32 positive reviews labelled negative), so a higher rank, more epochs "
            "or tuned learning rate are the natural next experiments.")

# ============================================================================
# 5. LLM
# ============================================================================
elif page == "🤖 LLM Prompting":
    hero("Large Language Model Evaluation",
         "Qwen2.5-0.5B-Instruct, evaluated without any task fine-tuning. Few-shot used four real training "
         "examples (two per class).")

    zs, fs = MAIN.loc[3], MAIN.loc[4]
    c1, c2, c3, c4 = st.columns(4)
    for col, m in zip((c1, c2, c3, c4), METRICS):
        with col:
            st.metric(m, f"{fs[m]:.2f}%", f"{fs[m] - zs[m]:+.2f} pts vs zero-shot")

    long = MAIN.iloc[3:].melt(id_vars="Model", value_vars=METRICS, var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group", text="Score",
                 color_discrete_map=COLORS, title="Zero-shot vs few-shot prompting")
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_yaxes(range=[0, 100], title="Score (%)")
    st.plotly_chart(style_fig(fig, 440), width="stretch")

    left, right = st.columns(2)
    with left:
        section("Takeaways")
        callout("Few-shot examples gave the biggest lift in <b>recall</b> and <b>F1</b>, meaning the model "
                "stops over-predicting a single class.")
        callout("Even so, a 0.5B instruction model stays <b>below</b> the trained classifiers (best LLM F1 67.92% "
                "vs 83.79% for the baseline).", warn=True)
    with right:
        section("Illustrative few-shot prompt layout")
        st.code(
            "System : You are a sentiment classifier. Answer Positive or Negative.\n"
            "Example 1 (Positive): <Hindi review>\n"
            "Example 2 (Negative): <Hindi review>\n"
            "Example 3 (Positive): <Hindi review>\n"
            "Example 4 (Negative): <Hindi review>\n"
            "Review  : <Hindi test review>\n"
            "Answer  :",
            language="text",
        )
        st.caption("Layout only - see src/train_llm.py for the exact prompt used in the experiments.")

# ============================================================================
# 6. LIVE PREDICTION
# ============================================================================
elif page == "🔮 Live Prediction":
    hero("Live Sentiment Prediction", "Type a Hindi review and see what your trained models predict.")

    live = available_live_models()
    if not live:
        st.warning("No trained models or processed training data were found. Expected folders: "
                   "`results/mbert_final_model`, `results/mbert_lora_final_model`, `dataset/processed/*train*.csv`.")
    else:
        choice = st.selectbox("Model", list(live.keys()))
        sample = st.selectbox("Load a sample", ["(write your own)"] + list(SAMPLES.keys()))
        default = SAMPLES.get(sample, "")
        text = st.text_area("Hindi review", value=default, height=120, placeholder="यहाँ हिंदी समीक्षा लिखें...")

        if st.button("Predict sentiment", type="primary"):
            if not text.strip():
                st.info("Please enter some text first.")
            else:
                kind, path = live[choice]
                try:
                    if kind == "baseline":
                        fitted = fit_baseline()
                        if fitted is None:
                            raise RuntimeError("Could not detect text/label columns in the training CSV.")
                        vec, clf = fitted
                        p_pos = float(clf.predict_proba(vec.transform([text]))[0][1])
                        probs = [1 - p_pos, p_pos]
                    else:
                        tok, model = load_transformer(kind, path)
                        probs = predict_transformer(tok, model, text)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Prediction failed: {exc}")
                else:
                    label = "Positive" if probs[1] >= probs[0] else "Negative"
                    conf = max(probs) * 100
                    cls = "pred-pos" if label == "Positive" else "pred-neg"
                    icon = "😊" if label == "Positive" else "😞"
                    st.markdown(
                        f'<div class="pred-card {cls}"><div>Predicted sentiment</div>'
                        f'<div class="big">{icon} {label}</div><div>{conf:.1f}% confidence</div></div>',
                        unsafe_allow_html=True,
                    )
                    fig = px.bar(x=[probs[0] * 100, probs[1] * 100], y=LABELS, orientation="h",
                                 color=LABELS, color_discrete_map={"Negative": "#F43F5E", "Positive": "#10B981"},
                                 labels={"x": "Probability (%)", "y": ""}, text=[f"{probs[0]*100:.1f}%", f"{probs[1]*100:.1f}%"])
                    fig.update_layout(showlegend=False)
                    fig.update_xaxes(range=[0, 100])
                    st.plotly_chart(style_fig(fig, 220), width="stretch")

        st.caption("Assumes label 0 = Negative and 1 = Positive. The LoRA model needs its base model "
                   "(bert-base-multilingual-cased) available locally or via internet on first load.")

    section("Browse the processed test set")
    test_csv = find_csv("test")
    if test_csv is None:
        st.info("No `*test*.csv` found in dataset/processed.")
    else:
        df = load_csv(str(test_csv))
        hindi, eng, lab = detect_columns(df)
        if lab is not None:
            df[lab] = normalise_labels(df[lab])
            pick = st.multiselect("Filter by sentiment", LABELS, default=LABELS)
            df = df[df[lab].isin(pick)]
        st.caption(f"{len(df)} rows from `{test_csv.name}`")
        st.dataframe(df.head(200), width="stretch", height=320)

# ============================================================================
# 7. FIGURES GALLERY
# ============================================================================
elif page == "🖼️ Figures Gallery":
    hero("Figures Gallery", "Every plot saved by your training and plotting scripts.")

    groups = {"Main figures": FIGURES, "Final figures": FINAL_FIGS}
    tabs = st.tabs(list(groups.keys()))
    for tab, (name, folder) in zip(tabs, groups.items()):
        with tab:
            imgs = sorted(folder.glob("*.png")) if folder.exists() else []
            if not imgs:
                st.info(f"No PNG files found in `{folder.relative_to(ROOT) if folder.exists() else folder}`.")
                continue
            cols = st.columns(2)
            for i, img in enumerate(imgs):
                with cols[i % 2]:
                    st.image(str(img), caption=img.stem.replace("_", " ").title(), width="stretch")
