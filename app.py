import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from joblib import load

# ---------------------------
# Page config (competition UI)
# ---------------------------
st.set_page_config(
    page_title="Student Performance Classifier",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Simple CSS (mesmerizing look)
# ---------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .stMetric {background: rgba(255,255,255,0.55); border-radius: 16px; padding: 14px; border: 1px solid rgba(0,0,0,0.06);}
    .card {background: rgba(255,255,255,0.55); border-radius: 18px; padding: 18px; border: 1px solid rgba(0,0,0,0.06);}
    .small {opacity: 0.8; font-size: 0.92rem;}
    .badge {display:inline-block; padding: 6px 10px; border-radius: 999px; background: rgba(0,0,0,0.06); margin-right: 6px;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Paths
# ---------------------------
ART_DIR = Path("artifacts")
MODEL_PATH = ART_DIR / "best_model.pkl"
LABEL_PATH = ART_DIR / "label_mapping.json"
REPORT_PATH = ART_DIR / "model_report.csv"

DATA_DIR = Path("data")
CLEAN_PATH = DATA_DIR / "cleaned_students.csv"

# ---------------------------
# Helpers
# ---------------------------
def load_artifacts():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}. Run model.ipynb to create it.")
        st.stop()
    model = load(MODEL_PATH)

    classes = None
    if LABEL_PATH.exists():
        with open(LABEL_PATH, "r", encoding="utf-8") as f:
            classes = json.load(f).get("classes", None)

    return model, classes

def predict_with_proba(model, X: pd.DataFrame, classes=None):
    pred = model.predict(X)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
    # If classes not provided, infer
    if classes is None:
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
    return pred, proba, classes

def human_label(x: str) -> str:
    mapping = {"low": "Low", "medium": "Medium", "high": "High"}
    return mapping.get(str(x).lower(), str(x))

def color_for_label(x: str) -> str:
    # Streamlit doesn't need colors hardcoded; we keep semantic mapping for text
    x = str(x).lower()
    if x == "high":
        return "✅"
    if x == "medium":
        return "🟡"
    if x == "low":
        return "🔴"
    return "ℹ️"

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("🎯 Student Performance Classifier")
st.sidebar.caption("Competition-grade ML + clean UI")

page = st.sidebar.radio(
    "Navigate",
    ["🏁 Orientation", "🔮 Single Prediction", "📦 Batch Prediction", "📊 Model & Data Insights"],
    index=0
)

with st.sidebar.expander("📌 Project Objective", expanded=True):
    st.write(
        "Predict a student's **performance class** based on study habits, attendance, sleep, and learning environment.\n\n"
        "**Classes:** Low / Medium / High\n\n"
        "This app is designed for **competition demos**: interactive controls, batch scoring, explainable outputs."
    )

with st.sidebar.expander("⚙️ How to run", expanded=False):
    st.code(
        "streamlit run app.py\n\n"
        "Make sure you have:\n"
        "- artifacts/best_model.pkl\n"
        "- data/cleaned_students.csv (optional for insights)\n",
        language="bash"
    )

# Load model artifacts once
model, classes = load_artifacts()

# ---------------------------
# Orientation Page
# ---------------------------
if page == "🏁 Orientation":
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.title("🏁 Orientation")
        st.write(
            "Welcome! This app predicts the **student performance class** using a trained ML model.\n\n"
            "**What you can do here:**\n"
            "- Make a **single prediction** with interactive inputs\n"
            "- Upload a CSV and get **batch predictions**\n"
            "- View **model report** and basic data insights\n\n"
            "✅ Built with leakage-safe pipelines, cross-validation, and best-model selection."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("✨ Quick Tips for Competition Demo")
        st.write("- Start with **Single Prediction** (fast wow-factor).")
        st.write("- Then show **Batch Prediction** + download results.")
        st.write("- Use **Insights** page to explain model behavior.")
        st.markdown('<span class="badge">Interactive</span><span class="badge">Downloadable</span><span class="badge">Explainable</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # Optional: show class distribution if data is present
    if CLEAN_PATH.exists():
        df = pd.read_csv(CLEAN_PATH)
        if "exam_score_class" in df.columns:
            dist = df["exam_score_class"].value_counts().reset_index()
            dist.columns = ["class", "count"]
            dist["percent"] = (dist["count"]/len(df)*100).round(2)

            colA, colB = st.columns([1, 1])
            with colA:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("📌 Class Distribution (from cleaned dataset)")
                fig = px.bar(dist, x="class", y="count", text="percent", template="plotly_white")
                fig.update_traces(texttemplate='%{text}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with colB:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("🥧 Class Distribution (Pie)")
                fig2 = px.pie(dist, names="class", values="count", template="plotly_white")
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Optional: place `data/cleaned_students.csv` to show dataset insights on this page.")

# ---------------------------
# Single Prediction Page
# ---------------------------
elif page == "🔮 Single Prediction":
    st.title("🔮 Single Prediction")
    st.caption("Enter student details → get predicted performance class + probabilities.")

    left, right = st.columns([1.1, 1])

    # Input form (based on your dataset schema)
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧾 Inputs")

        age = st.slider("Age", 15, 30, 20)
        gender = st.selectbox("Gender", ["male", "female", "other"])
        course = st.selectbox("Course", ["diploma", "bca", "b.sc", "b.tech", "bba", "ba", "b.com"])
        study_hours = st.slider("Study Hours (per day)", 0.0, 16.0, 4.0, 0.1)
        class_attendance = st.slider("Class Attendance (%)", 0.0, 100.0, 70.0, 0.1)
        internet_access = st.selectbox("Internet Access", ["yes", "no"])
        sleep_hours = st.slider("Sleep Hours (per day)", 0.0, 24.0, 7.0, 0.1)
        sleep_quality = st.selectbox("Sleep Quality", ["poor", "average", "good"])
        study_method = st.selectbox("Study Method", ["coaching", "online videos", "mixed", "self-study", "group study"])
        facility_rating = st.selectbox("Facility Rating", ["low", "medium", "high"])
        exam_difficulty = st.selectbox("Exam Difficulty", ["easy", "moderate", "hard"])

        # Build single-row dataframe matching training columns (cleaned schema)
        row = {
            "age": age,
            "gender": gender,
            "course": course,
            "study_hours": study_hours,
            "class_attendance": class_attendance,
            "internet_access": internet_access,
            "sleep_hours": sleep_hours,
            "sleep_quality": sleep_quality,
            "study_method": study_method,
            "facility_rating": facility_rating,
            "exam_difficulty": exam_difficulty,
        }
        X_one = pd.DataFrame([row])

        st.markdown("</div>", unsafe_allow_html=True)

        predict_btn = st.button("🚀 Predict", use_container_width=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📌 Output")
        if predict_btn:
            pred, proba, cls = predict_with_proba(model, X_one, classes)

            pred_label = pred[0]
            emoji = color_for_label(pred_label)

            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction", f"{emoji} {human_label(pred_label)}")
            c2.metric("Model", "Best Model (PKL)")
            c3.metric("Input Mode", "Single")

            if proba is not None and cls is not None:
                prob_df = pd.DataFrame({"class": cls, "probability": proba[0]}).sort_values("probability", ascending=False)
                st.write("### 🔎 Probabilities")
                fig = px.bar(prob_df, x="class", y="probability", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

                st.write("### ✅ Quick Interpretation")
                top_class = prob_df.iloc[0]["class"]
                st.info(f"Model is most confident about: **{human_label(top_class)}**")
            else:
                st.warning("This model does not expose predict_proba(). Showing class label only.")

        else:
            st.write("Click **Predict** to see results.")
            st.caption("Tip: Try increasing Study Hours and Attendance to see what changes.")

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Batch Prediction Page
# ---------------------------
elif page == "📦 Batch Prediction":
    st.title("📦 Batch Prediction")
    st.caption("Upload a CSV of student records → download predictions.")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("### Upload CSV")
    up = st.file_uploader("Upload your CSV file", type=["csv"])

    st.write(
        "✅ Expected columns (like cleaned dataset, **without** `exam_score_class`):\n"
        "`age, gender, course, study_hours, class_attendance, internet_access, sleep_hours, sleep_quality, study_method, facility_rating, exam_difficulty`\n\n"
        "If your file contains extra columns (e.g., student_id), it’s okay — the app will ignore unknown columns only if the pipeline supports it."
    )

    if up is not None:
        batch = pd.read_csv(up)
        st.write("Preview:")
        st.dataframe(batch.head(20), use_container_width=True)

        # Remove target if present
        if "exam_score_class" in batch.columns:
            batch = batch.drop(columns=["exam_score_class"], errors="ignore")

        # Attempt prediction
        try:
            pred, proba, cls = predict_with_proba(model, batch, classes)

            out = batch.copy()
            out["prediction"] = pred

            if proba is not None and cls is not None:
                for i, c in enumerate(cls):
                    out[f"proba_{c}"] = proba[:, i]

            st.success("✅ Predictions generated.")
            st.dataframe(out.head(50), use_container_width=True)

            # Download
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

            # Quick aggregate
            st.write("### 📌 Prediction Distribution")
            dist = out["prediction"].value_counts().reset_index()
            dist.columns = ["class", "count"]
            fig = px.bar(dist, x="class", y="count", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error("❌ Batch prediction failed.")
            st.exception(e)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Insights Page
# ---------------------------
else:
    st.title("📊 Model & Data Insights")

    col1, col2 = st.columns([1, 1])

    # Model report table if exists
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🏆 Model Leaderboard (from training)")
        if REPORT_PATH.exists():
            rep = pd.read_csv(REPORT_PATH)
            st.dataframe(rep, use_container_width=True)
        else:
            st.info("Run model.ipynb to generate artifacts/model_report.csv")
        st.markdown("</div>", unsafe_allow_html=True)

    # Dataset insights if exists
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📌 Dataset Snapshot (optional)")
        if CLEAN_PATH.exists():
            df = pd.read_csv(CLEAN_PATH)
            st.write("Rows:", len(df), " | Columns:", df.shape[1])
            st.dataframe(df.sample(min(20, len(df))), use_container_width=True)
        else:
            st.info("Place data/cleaned_students.csv to enable dataset insights.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # If we have cleaned data, show interactive charts
    if CLEAN_PATH.exists():
        df = pd.read_csv(CLEAN_PATH)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔍 Interactive Exploration")

        # Choose feature to explore vs target
        target_col = "exam_score_class"
        possible_features = [c for c in df.columns if c not in ["student_id", "exam_score", target_col]]
        feature = st.selectbox("Choose a feature", possible_features)

        if pd.api.types.is_numeric_dtype(df[feature]):
            fig = px.box(df, x=target_col, y=feature, points="all", template="plotly_white",
                         title=f"{feature} vs {target_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            tmp = pd.crosstab(df[feature], df[target_col], normalize="index").reset_index()
            tmp_melt = tmp.melt(id_vars=[feature], var_name=target_col, value_name="ratio")
            fig = px.bar(tmp_melt, x=feature, y="ratio", color=target_col, barmode="stack",
                         template="plotly_white",
                         title=f"{feature} → Target composition")
            fig.update_layout(xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)

        st.caption("Use this section to explain your model patterns during competition demos.")
        st.markdown("</div>", unsafe_allow_html=True)
