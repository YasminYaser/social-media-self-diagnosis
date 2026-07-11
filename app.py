import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

st.set_page_config(
    page_title="Digital Self-Diagnosis | Social Media & Mental Health",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Poppins', sans-serif !important; }

    .stApp {
        background: linear-gradient(180deg, #F4F9F4 0%, #FBF8EF 100%);
        color: #223142;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1F4B47 0%, #2C4A7C 100%);
    }
    section[data-testid="stSidebar"] * { color: #F5F7FA !important; }
    section[data-testid="stSidebar"] .stRadio > label { font-weight: 500; }

    /* Titles */
    h1 { color: #1F3A5F; font-weight: 800; letter-spacing: -0.5px; }
    h2, h3 { color: #2C4A7C; font-weight: 700; }

    /* Metric cards */
    .metric-card {
        background: #FFFFFF; border-radius: 18px; padding: 22px 20px;
        box-shadow: 0 4px 16px rgba(31,58,95,0.08);
        border: 1px solid #E7EEF3; text-align: center;
        transition: transform .15s ease;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value { font-size: 30px; font-weight: 800; color: #1F6F5C; font-family: 'Poppins', sans-serif; }
    .metric-label { font-size: 13px; color: #55697A; margin-top: 6px; font-weight: 500; }

    /* Section divider look for markdown callouts */
    div[data-testid="stMarkdownContainer"] p { color: #33465A; }

    /* Buttons */
    .stButton>button, .stFormSubmitButton>button {
        background: #1F6F5C; color: #FFFFFF; border-radius: 10px;
        border: none; font-weight: 600; padding: 10px 18px;
    }
    .stButton>button:hover, .stFormSubmitButton>button:hover { background: #17594A; color: #fff; }

    /* Info / warning boxes */
    div[data-testid="stAlert"] { border-radius: 12px; }

    /* Hide the default streamlit menu/footer branding for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header [data-testid="stToolbar"] {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data & model loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_survey():
    df = pd.read_csv(DATA_DIR / "survey.csv")
    df["Primary Social Media Platforms"] = df["Primary Social Media Platforms"].fillna("")
    df["Commonly Self-Diagnosed Conditions Observed"] = df[
        "Commonly Self-Diagnosed Conditions Observed"
    ].fillna("")
    return df


@st.cache_data
def load_trends():
    df = pd.read_csv(DATA_DIR / "trends.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "model.joblib")
    schema = joblib.load(MODEL_DIR / "feature_schema.joblib")
    with open(MODEL_DIR / "metrics.json") as f:
        metrics = json.load(f)
    with open(MODEL_DIR / "feature_importance.json") as f:
        fi = json.load(f)
    return model, schema, metrics, fi


survey = load_survey()
trends = load_trends()
model, schema, metrics, feature_importance = load_model()

PLATFORM_COLORS = px.colors.qualitative.Set2

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Digital Self-Diagnosis")
st.sidebar.caption("The Impact of Social Media on Mental Health Self-Diagnosis")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Explore the Survey", "Google Trends", "Predict Consultation", "Model Performance"],
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Team:** Yasmin Yaser, Ahmed Hossam, Belal Mohamed, Ammar Yaser")

# ---------------------------------------------------------------------------
# PAGE: Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Digital Self-Diagnosis")
    st.subheader("The Impact of Social Media on Mental Health Self-Diagnosis")
    st.write(
        "A growing number of people are self-diagnosing mental health conditions through "
        "social media, often without professional supervision — with risks of misdiagnosis, "
        "self-treatment, and delayed professional help."
    )

    n = len(survey)
    pct_self_diag = (survey["Commonly Self-Diagnosed Conditions Observed"].str.strip() != "").mean() * 100
    avg_time_map = {"Less than 1 hour": 0.5, "1-3 hours": 2, "3-5 hours": 4, "More than 5 hours": 6}
    avg_time = survey["Daily Time Spent"].map(avg_time_map).mean()
    pct_consulted = (survey["Professional Consultation Post-Self-Diagnosis"] == "Yes").mean() * 100
    pct_purchased = (survey["Purchased Medication/Supplements Based on SM Advice"] == "Yes").mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1, c2, c3, c4],
        [f"{n}", f"{avg_time:.1f} hrs", f"{pct_consulted:.1f}%", f"{pct_purchased:.1f}%"],
        ["Survey Responses", "Avg. Daily Social Media Time", "Consulted a Doctor After", "Bought Meds Based on SM Advice"],
    ):
        col.markdown(
            f'<div class="metric-card"><div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Key Takeaways")
    st.markdown(
        """
        - **Depression, ADHD and Anxiety** are the most commonly self-diagnosed conditions.
        - **Young adults (18-24)** make up the large majority of respondents affected.
        - Only a **small fraction consult a doctor** after self-diagnosing — most take no action or self-treat.
        - Negative emotions (**confusion, doubt, anxiety**) are as common as relief after watching this content.
        """
    )

# ---------------------------------------------------------------------------
# PAGE: Explore the Survey
# ---------------------------------------------------------------------------
elif page == "Explore the Survey":
    st.title("Explore the Survey Data")

    age_filter = st.multiselect(
        "Filter by Age Group", options=sorted(survey["Age Group"].unique()), default=list(survey["Age Group"].unique())
    )
    df = survey[survey["Age Group"].isin(age_filter)]
    st.caption(f"Showing {len(df)} of {len(survey)} responses")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Age Group Distribution**")
        fig = px.histogram(df, x="Age Group", color="Age Group", color_discrete_sequence=PLATFORM_COLORS)
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Daily Time Spent on Social Media**")
        fig = px.histogram(
            df, x="Daily Time Spent",
            category_orders={"Daily Time Spent": ["Less than 1 hour", "1-3 hours", "3-5 hours", "More than 5 hours"]},
            color_discrete_sequence=["#e07a5f"],
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Most Commonly Self-Diagnosed Conditions**")
        conds = df["Commonly Self-Diagnosed Conditions Observed"].str.split(",").explode().str.strip()
        conds = conds[conds != ""]
        top_conds = conds.value_counts().head(10).sort_values()
        fig = go.Figure(go.Bar(x=top_conds.values, y=top_conds.index, orientation="h", marker_color="#c9184a"))
        fig.update_layout(height=350, xaxis_title="Responses")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Feeling After Self-Diagnosis Content**")
        feelings = df["General Feeling After Watching Self-Diagnosis Content"].value_counts()
        fig = px.pie(values=feelings.values, names=feelings.index, hole=0.45,
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Professional Consultation Rate by Platform**")
    platforms = ["TikTok", "Instagram", "YouTube", "Facebook", "Reddit", "X (Twitter)"]
    rows = []
    for p in platforms:
        sub = df[df["Primary Social Media Platforms"].str.contains(p, case=False, na=False)]
        if len(sub) == 0:
            continue
        rate = (sub["Professional Consultation Post-Self-Diagnosis"] == "Yes").mean() * 100
        rows.append({"Platform": p, "Consulted (%)": rate, "n": len(sub)})
    plat_df = pd.DataFrame(rows).sort_values("Consulted (%)")
    fig = px.bar(plat_df, x="Consulted (%)", y="Platform", orientation="h", color="Consulted (%)",
                 color_continuous_scale="Teal", text=plat_df["Consulted (%)"].round(1).astype(str) + "%")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View raw survey data"):
        st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE: Google Trends
# ---------------------------------------------------------------------------
elif page == "Google Trends":
    st.title("Google Trends: Mental Health Search Interest (2021–2026)")
    cols_map = {
        "Anxiety_Search_Index": "Anxiety",
        "Stress_Search_Index": "Stress",
        "Depression_Search_Index": "Depression",
        "Mental_Health_General_Search_Index": "General Mental Health",
    }
    selected = st.multiselect("Search terms", list(cols_map.values()), default=list(cols_map.values()))

    fig = go.Figure()
    for raw, label in cols_map.items():
        if label in selected:
            fig.add_trace(go.Scatter(x=trends["Date"], y=trends[raw], mode="lines+markers", name=label))
    fig.update_layout(height=480, yaxis_title="Search Interest Index (0-100)", legend_title="Term")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Recent Trend: Last 6 Months vs First 6 Months")
    first6 = trends.head(6)[list(cols_map.keys())].mean()
    last6 = trends.tail(6)[list(cols_map.keys())].mean()
    diff = (last6 - first6).rename(index=cols_map)
    fig2 = px.bar(x=diff.index, y=diff.values, color=diff.values, color_continuous_scale="Reds",
                  labels={"x": "Category", "y": "Increase in Search Index"})
    fig2.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)
    st.info("Mental health searches accelerated sharply in the most recent period across all categories.")

# ---------------------------------------------------------------------------
# PAGE: Predict
# ---------------------------------------------------------------------------
elif page == "Predict Consultation":
    st.title("Will This Person Seek Professional Help?")
    st.write(
        "This model predicts the **probability that someone will consult a real doctor** "
        "after self-diagnosing a mental health condition based on social media content. "
        "Fill in the profile below."
    )

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age_group = st.selectbox("Age Group", sorted(survey["Age Group"].unique()))
            daily_time = st.selectbox(
                "Daily Time Spent on Social Media",
                ["Less than 1 hour", "1-3 hours", "3-5 hours", "More than 5 hours"],
            )
            platforms_sel = st.multiselect("Primary Social Media Platforms", schema["top_platforms"], default=["TikTok"])
            search_symptoms = st.selectbox(
                "Searches Symptoms on Social Media Instead of Search Engines",
                ["No", "Sometimes", "Yes", "Always"],
            )
            freq_exposure = st.slider("Frequency of Medical Content Exposure (1-5)", 1, 5, 3)

        with c2:
            algo_reco = st.selectbox(
                "Algorithm Recommends Diagnostic Content?", ["No", "Sometimes", "Yes, constantly"]
            )
            action_taken = st.selectbox(
                "Action Taken When Seeing Matching Symptoms",
                ["Skip it", "Watch the full video", "Read comments", "Save the video", "Like the video", "Other"],
            )
            feeling = st.selectbox(
                "General Feeling After Watching Content",
                ["Relief from understanding myself", "Confusion", "Doubt and disbelief", "Anxiety and stress"],
            )
            verify = st.selectbox(
                "How They Verify Content Creator Credibility",
                [
                    "If they are a specialized doctor",
                    "If they mention scientific sources",
                    "Based on a convincing personal experience",
                    "Based on follower count",
                    "I don't verify the information",
                ],
            )
            conditions_sel = st.multiselect(
                "Commonly Self-Diagnosed Conditions Observed", schema["top_conditions"], default=["Anxiety"]
            )

        with c3:
            suspected = st.selectbox("Suspected a Self-Condition Based Solely on a Video?", ["Yes", "No", "Not sure"])
            next_step = st.selectbox(
                "Next Step After Suspecting Condition",
                ["Did nothing", "Tried self-treatment", "Consulted a specialist", "Started labeling myself without a doctor"],
            )
            danger = st.slider("Perceived Danger of Self-Diagnosis (1-5)", 1, 5, 3)
            simplifies = st.slider("Content Simplifies Complex Issues (1-5)", 1, 5, 3)
            purchased = st.selectbox("Purchased Medication/Supplements Based on SM Advice?", ["No", "Yes"])

        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        row = {
            "Frequency of Medical Content Exposure (1-5)": freq_exposure,
            "Perceived Danger of Self-Diagnosis (1-5)": danger,
            "Content Simplifies Complex Issues (1-5)": simplifies,
            "Num Platforms Used": len(platforms_sel),
            "Num Conditions Self-Diagnosed": len(conditions_sel),
            "Age Group": age_group,
            "Daily Time Spent": daily_time,
            "Search for Symptoms on Social Media Instead of Search Engines": search_symptoms,
            "Algorithmic Recommendations of Diagnostic Content": algo_reco,
            "Action Taken When Seeing Matching Symptoms": action_taken,
            "General Feeling After Watching Self-Diagnosis Content": feeling,
            "Verification of Content Creator Credibility": verify,
            "Suspected Self-Condition Based Solely on Video": suspected,
            "Next Step After Suspecting Condition": next_step,
            "Purchased Medication/Supplements Based on SM Advice": purchased,
        }
        for p, col in zip(schema["top_platforms"], schema["platform_flags"]):
            row[col] = int(p in platforms_sel)
        for cnd, col in zip(schema["top_conditions"], schema["condition_flags"]):
            row[col] = int(cnd in conditions_sel)

        X_input = pd.DataFrame([row])
        proba = model.predict_proba(X_input)[0, 1]
        pred = "Yes" if proba >= 0.5 else "No"

        st.markdown("---")
        colA, colB = st.columns([1, 2])
        with colA:
            st.metric("Predicted Probability of Consulting a Doctor", f"{proba:.0%}")
            st.markdown(f"### Prediction: **{pred}**")
        with colB:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#2f6f5e"},
                        "steps": [
                            {"range": [0, 33], "color": "#f4d0d0"},
                            {"range": [33, 66], "color": "#fbe6b3"},
                            {"range": [66, 100], "color": "#c9e4c5"},
                        ],
                    },
                    number={"suffix": "%"},
                )
            )
            fig.update_layout(height=250, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        if proba < 0.3:
            st.warning(
                "Low predicted likelihood of seeking professional help. This profile matches "
                "patterns associated with staying at the self-diagnosis stage without medical follow-up."
            )
        st.caption(
            "This is a statistical estimate from a small survey sample (355 respondents, ~9.6% consulted a "
            "doctor) — it is **not** a medical or diagnostic tool."
        )

# ---------------------------------------------------------------------------
# PAGE: Model Performance
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    st.title("Model Performance & Insights")
    st.write(
        "**Model:** Random Forest Classifier (class-balanced) · "
        "**Target:** Professional Consultation Post-Self-Diagnosis (Yes/No) · "
        "**Validation:** 5-fold stratified cross-validation"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, label in zip(
        [c1, c2, c3, c4, c5],
        [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"], metrics["roc_auc"]],
        ["Accuracy", "Precision (Yes)", "Recall (Yes)", "F1 (Yes)", "ROC-AUC"],
    ):
        col.markdown(
            f'<div class="metric-card"><div class="metric-value">{val:.2f}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Confusion Matrix")
    cm = metrics["confusion_matrix"]
    fig = px.imshow(
        cm, text_auto=True, x=["Predicted No", "Predicted Yes"], y=["Actual No", "Actual Yes"],
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Top Feature Importances")
    fi_df = pd.DataFrame(feature_importance)
    fig = px.bar(
        fi_df.sort_values("importance"), x="importance", y="feature", orientation="h",
        color="importance", color_continuous_scale="Viridis",
    )
    fig.update_layout(height=500, yaxis_title="", xaxis_title="Importance")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Limitations")
    st.markdown(
        f"""
        - The dataset has only **{metrics['n_samples']} respondents**, with just **{metrics['n_positive']} positive
          cases** (people who consulted a doctor). This limits how confidently the model can be trusted.
        - Precision on the minority ("Yes") class is moderate ({metrics['precision']:.2f}) — expect some false alarms.
        - This tool is for **educational / exploratory purposes** only, not a clinical prediction instrument.
        """
    )
