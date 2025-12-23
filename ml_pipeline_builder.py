import io
from typing import Tuple
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz


def load_dataset(uploaded_file: io.BytesIO) -> pd.DataFrame:
   
    if uploaded_file is None:
        raise ValueError("No file provided")

    filename = uploaded_file.name.lower()
    try:
        if filename.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif filename.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
        else:
            raise ValueError(f"Unsupported file type: {filename}")
    except Exception as exc:
        raise ValueError(f"Error loading file: {exc}")


def preprocess_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    method: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if method == "Standardization":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return pd.DataFrame(X_train_scaled, columns=X_train.columns), pd.DataFrame(
            X_test_scaled, columns=X_test.columns
        )
    elif method == "Normalization":
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return pd.DataFrame(X_train_scaled, columns=X_train.columns), pd.DataFrame(
            X_test_scaled, columns=X_test.columns
        )
    return X_train, X_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
) -> object:
    if model_name == "Logistic Regression":
       
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=False)
    conf_mat = confusion_matrix(y_test, predictions)
    return {"accuracy": acc, "report": report, "confusion_matrix": conf_mat}


def draw_pipeline_diagram() -> graphviz.Digraph:
    diagram = graphviz.Digraph(format="png")
    diagram.attr(rankdir="LR", size="8,3")
    diagram.node("data", "Data Upload")
    diagram.node("pre", "Preprocessing\n(Standardization/Normalization)")
    diagram.node("split", "Train–Test Split")
    diagram.node("model", "Model Selection\n(Logistic/Decision Tree)")
    diagram.node("out", "Results")
    diagram.edge("data", "pre")
    diagram.edge("pre", "split")
    diagram.edge("split", "model")
    diagram.edge("model", "out")
    return diagram


def main() -> None:
    st.set_page_config(page_title=" ML Pipeline Builder", layout="wide")
    st.title("Machine Learning Pipeline Builder")
    st.write(
    )

    
    with st.sidebar:
        st.header("Pipeline Flow")
        diagram = draw_pipeline_diagram()
        st.graphviz_chart(diagram)

    
    st.header("1. Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file", type=["csv", "xlsx", "xls"], accept_multiple_files=False
    )
    dataset: pd.DataFrame | None = None
    if uploaded_file:
        try:
            dataset = load_dataset(uploaded_file)
            st.success("File loaded successfully!")
            st.write(f"Shape: {dataset.shape[0]} rows × {dataset.shape[1]} columns")
            st.write("Column names:")
            st.write(list(dataset.columns))
            st.write("Preview:")
            st.dataframe(dataset.head())
        except ValueError as e:
            st.error(str(e))

    
    if dataset is not None:
        
        st.header("2. Configure Features and Target")
        with st.form(key="config_form"):
            target_column = st.selectbox(
                "Select the target (label) column", options=dataset.columns
            )
            
            feature_candidates = [col for col in dataset.columns if col != target_column]
            selected_features = st.multiselect(
                "Select feature columns", options=feature_candidates, default=feature_candidates
            )
            preprocessing_method = st.selectbox(
                "Select preprocessing method",
                options=["None", "Standardization", "Normalization"],
                index=0,
            )
            test_size = st.slider(
                "Select test set proportion", min_value=0.1, max_value=0.5, value=0.2, step=0.05
            )
            model_name = st.selectbox(
                "Select model",
                options=["Logistic Regression", "Decision Tree"],
            )
            submit = st.form_submit_button("Run Pipeline")

        if submit:
            if not selected_features:
                st.error("Please select at least one feature column.")
            else:
                X = dataset[selected_features]
                y = dataset[target_column]
                
                X = pd.get_dummies(X, drop_first=True)

                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y if y.nunique() <= 10 else None
                )
                
                X_train_processed, X_test_processed = preprocess_features(
                    X_train, X_test, preprocessing_method
                )
                
                model = train_model(X_train_processed, y_train, model_name)
                
                results = evaluate_model(model, X_test_processed, y_test)
                
                st.header("5. Results")
                st.write(f"**Accuracy:** {results['accuracy']:.2%}")
                st.write("**Classification report:**")
                st.text(results["report"])
                
                st.write("**Confusion Matrix:**")
                fig, ax = plt.subplots()
                sns.heatmap(
                    results["confusion_matrix"],
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=False,
                    ax=ax,
                )
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)


if __name__ == "__main__":
    main()