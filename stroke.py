import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import pandas as pd

# Load scaler dan label encoders
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="stroke_diagnosis.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul aplikasi
st.title("Cek Risiko Stroke")
st.write("Masukkan data berikut untuk mengetahui risiko seseorang mengalami stroke.")

# Input form pengguna
gender = st.selectbox("Jenis Kelamin", options=["Male", "Female"])
age = st.number_input("Usia", min_value=0, max_value=120, value=30)
hypertension = st.selectbox("Hipertensi", options=[0, 1])
heart_disease = st.selectbox("Penyakit Jantung", options=[0, 1])
ever_married = st.selectbox("Pernah Menikah", options=["Yes", "No"])
work_type = st.selectbox("Jenis Pekerjaan", options=["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Tinggal di", options=["Urban", "Rural"])
avg_glucose_level = st.number_input("Rata-rata Glukosa", min_value=0.0, max_value=300.0, value=90.0)
bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0)
smoking_status = st.selectbox("Status Merokok", options=["formerly smoked", "never smoked", "smokes", "Unknown"])

# Tombol prediksi
if st.button("Cek Hasil"):

    # Bentuk data input sebagai DataFrame satu baris
    input_dict = {
        "gender": [gender],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "ever_married": [ever_married],
        "work_type": [work_type],
        "Residence_type": [residence_type],
        "avg_glucose_level": [avg_glucose_level],
        "bmi": [bmi],
        "smoking_status": [smoking_status]
    }
    input_df = pd.DataFrame(input_dict)

    # Transform kolom kategorikal
    for col in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]:
        encoder = label_encoders[col]
        input_df[col] = encoder.transform(input_df[col])

    # Scaling
    input_scaled = scaler.transform(input_df).astype(np.float32)

    # Prediksi dengan model TFLite
    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]  # Output berupa probabilitas

    # Tampilkan hasil
    if prediction > 0.5:
        st.error("Hasil: Berisiko Mengalami Stroke")
    else:
        st.success("Hasil: Tidak Berisiko Mengalami Stroke")
