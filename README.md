# Employee-Salary-Prediction

...
## 📌 Overview

This project focuses on predicting whether an employee earns more than \$50K per year using various demographic and work-related features. The solution involves building and deploying a classification model using machine learning techniques, with an interactive interface powered by **Streamlit**. The goal is to assist HR professionals or analysts in understanding salary patterns and making data-driven decisions.

---

## 🚀 Features

* 🎯 Predict salary class (`<=50K` or `>50K`) based on user inputs
* 📂 Upload a CSV file for **batch prediction**
* 📊 Visualize and explore input data before prediction
* 💾 Trained model integrated and saved using `joblib`
* 🌐 Web-based interface using **Streamlit**

---

## 📂 Project Structure

```
├── app.py                    # Streamlit web app for prediction
├── best_model.pkl           # Trained machine learning model
├── employee salary prediction.ipynb  # Jupyter notebook for training and analysis
├── requirements.txt         # Libraries used (optional, recommended to add)
└── README.md                # Project documentation
```

---

## ⚙️ System Requirements

* OS: Windows / macOS / Linux
* Python: 3.8 or above
* RAM: Minimum 4 GB (recommended: 8 GB)

---

## 📚 Libraries Used

```python
pandas
numpy
scikit-learn
streamlit
joblib
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🧠 Machine Learning Pipeline

1. **Data Cleaning & Preprocessing**

   * Handle missing values
   * Encode categorical variables
   * Scale numerical features

2. **Model Training**

   * Tried models: Logistic Regression, Random Forest, Gradient Boosting
   * Best model saved using `joblib`

3. **Model Deployment**

   * Deployed via `Streamlit` for easy use

---

## 🖥️ How to Run the App

1. Clone the repository:

```bash
git clone https://github.com/your-username/employee-salary-classification.git
cd employee-salary-classification
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the app:

```bash
streamlit run app.py
```

4. Open the local URL (usually `http://localhost:8501`) in your browser.

---

## 📌 Dataset

* Dataset Source: [UCI Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
* Features include: age, workclass, education, marital-status, occupation, race, gender, hours-per-week, etc.

---

## 🔮 Future Enhancements

* Add explainability tools like SHAP
* Include model performance metrics in the app
* Deploy on Streamlit Cloud or AWS
* Integrate with HRMS or databases for real-time use

---

## 📝 License

This project is open-source and free to use for educational purposes.

---
