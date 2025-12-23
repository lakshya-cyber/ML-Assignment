# Machine Learning Pipeline Builder


##  Objective

To allow users to:
- Upload datasets
- Apply preprocessing techniques
- Perform train–test split
- Select and train ML models
- View model performance



---

##  Features

### 1 Dataset Upload
- Supports CSV, XLSX, XLS files
- Displays dataset shape and column names
- Handles invalid file formats gracefully

### 2 Data Preprocessing
- Standardization (StandardScaler)
- Normalization (MinMaxScaler)

### 3 Train–Test Split
- User-controlled split ratios (70–30, 80–20)

### 4 Model Selection
- Logistic Regression
- Decision Tree Classifier

### 5 Model Output
- Execution status
- Accuracy score
- Classification report
- Confusion matrix visualization

---

##  Pipeline Flow

Data Upload → Preprocessing → Train/Test Split → Model Selection → Results

---

##  Tech Stack

- Python 3.11
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

##  How to Run the Project

### Install dependencies
```bash
python -m pip install streamlit pandas scikit-learn matplotlib seaborn
```

### Run the app
```bash
python -m streamlit run ml_pipeline_builder.py
```

### Open in browser
```
http://localhost:8501
```

---






