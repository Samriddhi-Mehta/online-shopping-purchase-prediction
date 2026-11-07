
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ShopSense - Purchase Prediction",
    page_icon="🛍️",
    layout="wide"
)

# --------------------------------------------------
# 2. CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #FFFFFF;
}

.block-container {
    padding-top: 2rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #273444;
}

[data-testid="stSidebar"] {
    background-color: #F7F5F2;
    border-right: 1px solid #E8E3DE;
}

div[data-testid="stMetric"] {
    background: #FAF7F4;
    border: 1px solid #EAE1DA;
    padding: 15px;
    border-radius: 12px;
}

.stButton button,
.stDownloadButton button {
    background-color: #C97861;
    color: white;
    border-radius: 8px;
    border: none;
}

.stButton button:hover,
.stDownloadButton button:hover {
    background-color: #A95F4B;
    color: white;
}

hr {
    border-color: #EAE1DA;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. FILE PATHS AND MODEL NAMES
# --------------------------------------------------

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
DATA_PATH = BASE_DIR / "test_data.csv"

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "k_nearest_neighbors.pkl",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl"
}

descriptions = {
    "Logistic Regression": "A linear classification model.",
    "Decision Tree": "A tree-based classification model.",
    "K-Nearest Neighbors": "A distance-based classification model.",
    "Gaussian Naive Bayes": "A probabilistic classification model.",
    "Support Vector Machine": "A margin-based classification model."
}


# --------------------------------------------------
# 4. LOAD TRAINED MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    loaded_models = {}

    for name, filename in model_files.items():

        file_path = MODEL_DIR / filename

        if file_path.exists():
            loaded_models[name] = joblib.load(file_path)

    return loaded_models


models = load_models()

if len(models) != 5:
    st.error("Some model files are missing. Please check the model folder.")
    st.stop()


# --------------------------------------------------
# 5. HELPER FUNCTIONS
# --------------------------------------------------

def convert_target(value):

    if isinstance(value, str):
        return value.strip().lower() in ["true", "1", "yes"]

    return bool(value)


def get_scores(model, features):

    classes = [convert_target(c) for c in model.classes_]
    positive_index = classes.index(True)

    if hasattr(model, "predict_proba"):

        return model.predict_proba(features)[:, positive_index]

    scores = model.decision_function(features)

    if positive_index == 0:
        scores = -scores

    return scores


# --------------------------------------------------
# 6. SIDEBAR NAVIGATION AND DATA UPLOAD
# --------------------------------------------------

st.sidebar.title("🛍️ ShopSense")
st.sidebar.caption("Purchase Intelligence Platform")

st.sidebar.divider()

st.sidebar.subheader("Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["Evaluation", "Model Comparison", "Dataset"],
    label_visibility="collapsed"
)

st.sidebar.divider()

st.sidebar.subheader("Test Dataset")

use_sample = st.sidebar.checkbox(
    "Use bundled test_data.csv",
    value=True
)

uploaded_file = st.sidebar.file_uploader(
    "Upload another CSV",
    type=["csv"]
)

if uploaded_file is not None and not use_sample:

    data = pd.read_csv(uploaded_file)

elif use_sample and DATA_PATH.exists():

    data = pd.read_csv(DATA_PATH)

else:

    st.warning(
        "Select the bundled test dataset or upload a CSV file."
    )
    st.stop()


# --------------------------------------------------
# 7. DATA VALIDATION
# --------------------------------------------------

if "Revenue" not in data.columns:

    st.error("The dataset must contain a Revenue column.")
    st.stop()

y_test = data["Revenue"].apply(convert_target)
X_test = data.drop(columns=["Revenue"])

# Validate the feature columns
first_model = next(iter(models.values()))

if hasattr(first_model, "feature_names_in_"):

    expected_features = list(first_model.feature_names_in_)

    missing = set(expected_features) - set(X_test.columns)
    extra = set(X_test.columns) - set(expected_features)

    if missing or extra:

        st.error(
            f"Feature mismatch. Missing: {list(missing)}. "
            f"Unexpected: {list(extra)}"
        )
        st.stop()

    X_test = X_test[expected_features]


# Sidebar model selection
st.sidebar.divider()

st.sidebar.subheader("Model Selection")

selected_model_name = st.sidebar.selectbox(
    "Choose a classifier",
    list(models.keys())
)

st.sidebar.info(
    descriptions[selected_model_name]
)

st.sidebar.divider()

st.sidebar.success("All 5 models loaded")

st.sidebar.caption(f"Dataset records: {len(data):,}")
st.sidebar.caption(f"Input features: {X_test.shape[1]}")


# --------------------------------------------------
# 8. EVALUATE ALL FIVE MODELS
# --------------------------------------------------

all_predictions = {}
comparison = {}

for name, model in models.items():

    predicted = pd.Series(
        model.predict(X_test)
    ).apply(convert_target)

    all_predictions[name] = predicted

    scores = get_scores(model, X_test)

    comparison[name] = {

        "Accuracy": accuracy_score(
            y_test, predicted
        ),

        "AUC": roc_auc_score(
            y_test, scores
        ),

        "Precision": precision_score(
            y_test, predicted, zero_division=0
        ),

        "Recall": recall_score(
            y_test, predicted, zero_division=0
        ),

        "F1 Score": f1_score(
            y_test, predicted, zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y_test, predicted
        )
    }

results_df = pd.DataFrame(comparison).T


# --------------------------------------------------
# 9. MAIN PAGE HEADER
# --------------------------------------------------

st.title("Online Shopping Purchase Prediction")

st.write(
    "Analyse visitor browsing behaviour and predict "
    "whether an online shopping session will generate revenue."
)

st.caption(
    "Machine Learning Classification Project | Five Trained Models"
)

# Dataset summary
col1, col2, col3, col4 = st.columns(4)

col1.metric("Test Records", f"{len(data):,}")
col2.metric("Input Features", X_test.shape[1])
col3.metric("ML Models", len(models))
col4.metric("Purchase Rate", f"{y_test.mean() * 100:.2f}%")

st.divider()


# ==================================================
# PAGE 1: EVALUATION
# ==================================================

if page == "Evaluation":

    model = models[selected_model_name]
    predicted = all_predictions[selected_model_name]
    metrics = results_df.loc[selected_model_name]

    st.subheader("Individual Model Evaluation")

    st.write(f"Selected model: **{selected_model_name}**")

    # Six evaluation metrics
    metric_names = [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC"
    ]

    metric_cols = st.columns(6)

    for col, metric in zip(metric_cols, metric_names):

        col.metric(
            metric,
            f"{metrics[metric]:.4f}"
        )

    st.divider()

    left, right = st.columns(2)

    # Confusion Matrix
    with left:

        st.markdown("#### Confusion Matrix")

        matrix = confusion_matrix(
            y_test,
            predicted,
            labels=[False, True]
        )

        fig, ax = plt.subplots(figsize=(5, 4))

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Oranges",
            xticklabels=["No Purchase", "Purchase"],
            yticklabels=["No Purchase", "Purchase"],
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)
        plt.close(fig)

    # Classification Report
    with right:

        st.markdown("#### Classification Report")

        report = classification_report(
            y_test,
            predicted,
            labels=[False, True],
            target_names=["No Purchase", "Purchase"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).T

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )

    # ROC Curve
    st.divider()

    st.markdown("#### ROC Curve")

    scores = get_scores(model, X_test)

    fpr, tpr, thresholds = roc_curve(
        y_test, scores
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        fpr,
        tpr,
        color="#C97861",
        linewidth=2,
        label=f"AUC = {metrics['AUC']:.4f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)


# ==================================================
# PAGE 2: MODEL COMPARISON
# ==================================================

elif page == "Model Comparison":

    st.subheader("Comparison of All Five Models")

    st.write(
        "Compare the performance of all five classification "
        "algorithms using six evaluation metrics."
    )

    # Comparison Table
    st.markdown("#### Performance Comparison Table")

    st.dataframe(
        results_df.style.format("{:.4f}").highlight_max(
            axis=0,
            color="#E8F3E8"
        ),
        use_container_width=True
    )

    st.divider()

    # Interactive graph
    st.markdown("#### Model Performance Graph")

    selected_metric = st.selectbox(
        "Select evaluation metric",
        [
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1 Score",
            "MCC"
        ]
    )

    fig, ax = plt.subplots(figsize=(9, 4))

    results_df[selected_metric].sort_values(
        ascending=False
    ).plot(
        kind="bar",
        color="#D98268",
        edgecolor="#A65F4C",
        ax=ax
    )

    ax.set_title(
        f"Model Comparison - {selected_metric}"
    )

    ax.set_xlabel("Machine Learning Model")
    ax.set_ylabel(selected_metric)

    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Best model for each metric
    st.markdown("#### Best Model for Each Metric")

    best_models = pd.DataFrame({

        "Metric": results_df.columns,

        "Best Model": [
            results_df[metric].idxmax()
            for metric in results_df.columns
        ],

        "Score": [
            results_df[metric].max()
            for metric in results_df.columns
        ]
    })

    st.dataframe(
        best_models.style.format({
            "Score": "{:.4f}"
        }),
        use_container_width=True
    )

    # Download comparison
    st.download_button(
        "Download Model Comparison CSV",
        data=results_df.to_csv().encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv"
    )


# ==================================================
# PAGE 3: DATASET
# ==================================================

elif page == "Dataset":

    st.subheader("Dataset Overview")

    left, right = st.columns(2)

    # Revenue distribution
    with left:

        st.markdown("#### Revenue Distribution")

        counts = y_test.value_counts().reindex(
            [False, True],
            fill_value=0
        )

        fig, ax = plt.subplots(figsize=(5, 4))

        ax.pie(
            counts,
            labels=["No Purchase", "Purchase"],
            autopct="%1.1f%%",
            colors=["#E8A87C", "#C97861"],
            startangle=90
        )

        st.pyplot(fig)
        plt.close(fig)

    # Dataset information
    with right:

        st.markdown("#### Dataset Information")

        st.write("Total records:", len(data))
        st.write("Total features:", X_test.shape[1])
        st.write("Purchase records:", int(y_test.sum()))
        st.write(
            "Non-purchase records:",
            int((~y_test).sum())
        )

        st.write(
            "Duplicate records:",
            int(data.duplicated().sum())
        )

    st.divider()

    # Dataset preview
    st.markdown("#### Dataset Preview")

    st.dataframe(
        data.head(20),
        use_container_width=True
    )

    st.divider()

    # Missing values
    st.markdown("#### Missing Value Analysis")

    missing = data.isnull().sum()

    missing_df = pd.DataFrame({

        "Feature": missing.index,

        "Missing Values": missing.values

    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )


# --------------------------------------------------
# 10. DOWNLOAD PREDICTIONS
# --------------------------------------------------

st.divider()

st.subheader("Download Prediction Results")

selected_model = models[selected_model_name]

final_output = data.copy()

final_output["Predicted_Revenue"] = (
    all_predictions[selected_model_name].values
)

if hasattr(selected_model, "predict_proba"):

    final_output["Purchase_Score"] = get_scores(
        selected_model,
        X_test
    )

st.download_button(
    "Download Predictions CSV",
    data=final_output.to_csv(index=False).encode("utf-8"),
    file_name="purchase_predictions.csv",
    mime="text/csv"
)

st.divider()

st.caption(
    "ShopSense | Online Shopping Purchasing Intention Classification Project"
)
