# 🚀 Customer Churn Prediction System

## 📌 Overview

This project is a Machine Learning-based system that predicts whether a customer is likely to churn (leave a service) using historical customer data. It helps businesses identify high-risk customers and take proactive actions to improve retention.

---

## 🎯 Objective

To build an intelligent system that analyzes customer data and predicts churn probability, enabling data-driven decision-making for customer retention strategies.

---

## 🧠 Features

* 🔍 Predicts customer churn (Yes/No)
* 📊 Displays churn probability score
* 🚦 Classifies risk levels (Low / Medium / High)
* 💡 Provides business recommendations
* 📈 Shows feature importance for insights
* 🖥️ Interactive UI using Streamlit

---

## 🏗️ Tech Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
* **Model:** Logistic Regression, Random Forest
* **Frontend/UI:** Streamlit

---

## 📂 Project Structure

```
customer-churn-project/
│
├── data/
│   └── churn.csv
│
├── preprocessing.py
├── model.py
├── train.py
├── app.py
├── model.pkl
└── README.md
```

---

## 🔄 Workflow

1. Data Collection (Kaggle dataset)
2. Data Preprocessing (cleaning & encoding)
3. Exploratory Data Analysis (EDA)
4. Model Training & Evaluation
5. Model Saving (pickle)
6. Deployment using Streamlit

---

## ▶️ How to Run

### 1. Clone Repository

```
git clone <https://github.com/Sharmila200608/churn_MLprojec>
cd customer-churn-project
```

### 2. Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Train Model

```
python train.py
```

### 5. Run Application

```
streamlit run app.py
```

---

## 📊 Model Evaluation

The model performance is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score

Special focus is given to **Recall**, as identifying potential churn customers is critical.

---

## 💡 Business Impact

* Helps identify customers at risk of leaving
* Enables proactive retention strategies
* Improves customer satisfaction
* Reduces revenue loss

---

## 🚀 Future Enhancements

* Add advanced models (XGBoost, LightGBM)
* Deploy on cloud (AWS / Streamlit Cloud)
* Add real-time API integration
* Enhance UI/UX

---

## 🙌 Acknowledgements

* Dataset sourced from Kaggle
* Built as part of a Machine Learning project

---

