# 🩺 GlucoPulse — Diabetes Readmission Prediction System

GlucoPulse is an end-to-end Machine Learning web application that predicts whether a diabetic patient is likely to be **readmitted to the hospital** based on clinical and hospital stay information.

The project demonstrates the **complete ML lifecycle** — from feature engineering and model training to real-time inference and cloud deployment using Streamlit.

---

## 🚀 Live Features

- Interactive Streamlit web interface
- Real-time readmission prediction
- Ensemble of multiple ML models
- Probability-based confidence scores
- Clean and responsive dark-mode UI
- Ready for cloud deployment

---

## 🧠 Models Used

| Model | Purpose |
|------|--------|
| Random Forest | Captures non-linear feature interactions |
| Decision Tree | Interpretable baseline model |
| XGBoost | Boosted ensemble for higher performance |
| TensorFlow Neural Model | Deep learning based probability estimator |

Final prediction is made using **average probability ensembling**.

---

## 🔬 Feature Engineering

- Encoded age groups
- Hospital stay duration analysis
- Emergency visit ratio
- Medication intensity features
- Diagnosis count aggregation
- Feature scaling using trained StandardScaler

All preprocessing steps are **consistent between training and inference**.

---

## 📊 Prediction Pipeline

1. User enters patient details via UI  
2. Features are engineered and aligned  
3. Input is scaled using trained scaler  
4. Each model predicts readmission probability  
5. Probabilities are averaged  
6. Final classification is displayed with confidence scores  

---

## 🖥️ Tech Stack

- **Programming Language:** Python  
- **ML Libraries:** scikit-learn, XGBoost, TensorFlow  
- **Frontend:** Streamlit  
- **Model Persistence:** joblib, Keras  
- **Deployment:** Streamlit Community Cloud  
- **Version Control:** Git & GitHub  

---

## 📁 Project Structure

GlucoPulse/
│
├── app.py
├── rf_model.pkl
├── dt_model.pkl
├── xgb_model.pkl
├── scaler.pkl
├── tf_model.keras
├── feature_columns.pkl
│
├── requirements.txt
├── README.md
└── .gitignore

🌐 Deployment

The application is designed to be deployed on Streamlit Community Cloud directly from this GitHub repository.

All trained models and preprocessing artifacts are included to ensure reproducible inference.
