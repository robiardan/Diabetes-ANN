import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Memuat model Keras dari file .h5
diabetes_model = load_model('final_model.h5')

# Judul aplikasi
st.title('Prediksi Diabetes')

# Membagi kolom
col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.text_input('Input nilai Pregnancies', '0')
    Glucose = st.text_input('Input nilai Glucose', '0')
    BloodPressure = st.text_input('Input nilai Blood Pressure', '0')
    SkinThickness = st.text_input('Input nilai Skin Thickness', '0')

with col2:
    Insulin = st.text_input('Input nilai Insulin', '0')
    BMI = st.text_input('Input nilai BMI', '0.0')
    DiabetesPedigreeFunction = st.text_input('Input nilai Diabetes Pedigree Function', '0.0')
    Age = st.text_input('Input nilai Age', '0')

# Code untuk prediksi
diab_diagnosis = ''

# Membuat tombol untuk prediksi
if st.button('Test Prediksi Diabetes'):
    # Mengumpulkan input dan mengubah ke format yang diperlukan
    input_data = np.array([[float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
                             float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]])
    
    # Normalisasi input
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)  # Normalisasi input data

    # Melakukan prediksi
    diab_prediction = diabetes_model.predict(input_data)
    diab_prediction = (diab_prediction > 0.5).astype("int32")  # Mengubah prediksi menjadi biner

    # Menampilkan hasil prediksi
    if diab_prediction[0][0] == 1:
        diab_diagnosis = 'Pasien diprediksi terkena Diabetes'
    else:
        diab_diagnosis = 'Pasien diprediksi tidak terkena Diabetes'

# Menampilkan hasil di Streamlit
st.success(diab_diagnosis)
