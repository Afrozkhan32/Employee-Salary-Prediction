import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs (these must match your training feature columns)
st.sidebar.header("Input Employee Details")

# ✨ Replace these fields with your dataset's actual input columns
age = st.sidebar.slider("Age", 18, 65, 30)
education= st.sidebar.selectbox("Education Level", [
    "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
])
workclass=st.sidebar.selectbox("work class",['Private','Self-emp-not-inc','Local-gov','Others','State-gov','Self-emp-inc','Federal-gov','Without-pay','Never-worked'])
fnlwgt=st.sidebar.slider('fnlwgt',12285,1490400)
maritalstatus=st.sidebar.selectbox('Marital-status',['Married-civ-spouse','Never-married','Divorced','Separated','Widowed','Married-spouse-absent','Married-AF-spouse'])

occupation = st.sidebar.selectbox("Job Role", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"
])

relationship=st.sidebar.selectbox('relationship',['Husband','Not-in-family','Own-child','Unmarried','Wife','Other-relative'])
race=st.sidebar.selectbox('Race',['White','Black','Asian-Pac-Islander','Amer-Indian-Eskimo','Other'])
Gender = st.sidebar.selectbox('Gender',['Male','Female'])
capitalgain=st.sidebar.slider('capitalgain',0,99999)
capitalloss=st.sidebar.slider('capitalloss',0,4356)
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
#experience = st.sidebar.slider("Years of Experience", 0, 40, 5)
nativecountry=st.sidebar.selectbox('Native-country',[
    'United-States','Mexico','?','Philippines','Germany','Puerto-Rico','Canada','El-Salvador','India','Cuba','England','China','South','Jamaica','Italy','Dominican-Republic','Guatemala',
    'Poland','Vietnam','Columbia','Haiti','Portugal','Taiwan','Iran','Nicaragua','Greece','Peru','Ecuador','France','Ireland','Thailand','Hong','Cambodia','Trinadad&Tobago','Laos','Outlying-US(Guam-USVI-etc)','Yugoslavia','Scotland',
    'Honduras','Hungary','Holand-Netherlands'

])

# Build input DataFrame (⚠️ must match preprocessing of your training data)
input_df = pd.DataFrame({
    'age': [age],
    
    'education': [education],
    'workclass':[workclass],
    'fnlwgt': [fnlwgt],
    'marital-status':[maritalstatus],
    'occupation': [occupation],
    'relationship' : [relationship],
    'race' : [race],
    'gender':[Gender],
    'capital-gain':[capitalgain],
    'capital-loss':[capitalloss],

    'hours-per-week': [hours_per_week],
    'native-country':[nativecountry]
    #'experience': [experience]
})

st.write("### 🔎 Input Data")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())
    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')

