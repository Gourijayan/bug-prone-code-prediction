# 🐞 Bug-Prone Code Prediction Using Machine Learning

## 📌 Project Overview

This project predicts whether software modules are likely to contain defects using machine learning techniques. It analyzes software code metrics to help developers identify potentially defect-prone modules and prioritize software testing.

The project uses the NASA KC1 Software Defect Dataset and compares two classification algorithms: Logistic Regression and Random Forest.

A Streamlit web application allows users to upload CSV files and obtain defect predictions with estimated risk probabilities.

## 🎯 Objectives

- Analyze software metrics associated with defects.
- Perform data cleaning and exploratory data analysis.
- Train and compare machine learning classification models.
- Evaluate models using standard classification metrics.
- Develop an interactive defect prediction dashboard.

## 📊 Dataset

**Dataset:** NASA KC1 Software Defect Dataset

| Description | Value |
|---|---|
| Original records | 2,109 |
| Original columns | 22 |
| Input features | 21 |
| Cleaned records | 1,212 |
| Training records | 969 |
| Testing records | 243 |
| Target column | `defects` |

The target labels are:

- `0` — Not Defect-Prone
- `1` — Defect-Prone

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Visual Studio Code

## 🤖 Machine Learning Models

**1. Logistic Regression**

Used as a baseline binary classification model.

**2. Random Forest**

Used as the main prediction model because it achieved better overall evaluation performance.

## 📈 Model Performance

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 60.91% | 70.37% |
| Precision | 32.61% | 43.28% |
| Recall | 47.62% | 46.03% |
| F1-score | 38.71% | 44.62% |
| ROC-AUC | 60.79% | 63.19% |

**Selected Model:** Random Forest

## 🖥️ Streamlit Dashboard Features

- Dataset overview
- Defect distribution visualization
- Model performance comparison
- CSV file upload
- Defect-prone module prediction
- Estimated defect-risk percentage
- Downloadable prediction results

## 🚀 How to Run

**1. Clone the repository**

```bash
git clone https://github.com/gourijayan/bug-prone-code-prediction.git
cd bug-prone-code-prediction
```

**2. Install dependencies**

```bash
python -m pip install -r requirements.txt
```

**3. Launch the dashboard**

```bash
python -m streamlit run app.py
```

**4. Upload a CSV file**

Upload a CSV containing the same 21 software metrics used during training.

The application will predict whether each software module is defect-prone and display the estimated defect risk.

## ⚠️ Limitations

The system predicts defect-proneness but does not identify the exact bug type or faulty line of code.

The Random Forest model achieved 46.03% recall for defect-prone modules, so it may miss some actual defects.

Predictions should support software testing decisions rather than replace manual code review and testing.

## 🔮 Future Enhancements

- Hyperparameter tuning
- Feature selection
- Improved handling of class imbalance
- Evaluation using additional software datasets
- Enhanced dashboard visualizations

## 👩‍💻 Author

**Gouri Jayan**

GitHub: https://github.com/gourijayan