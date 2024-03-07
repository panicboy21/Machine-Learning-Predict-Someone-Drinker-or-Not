import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import sklearn


# Membuat Header
st.header('Model Predict Drinker Classification')
st.write(sklearn.__version__)
# Load Model
with open('rf_gridcv_best.pkl', 'rb') as file_1:
  rf_gridcv_best = pickle.load(file_1)

with open('preprocessing_pipeline.pkl', 'rb') as file_2:
  preprocessing_pipeline = pickle.load(file_2)

with st.form('Input data'):
  sex = st.radio('Sex', ('Male', 'Female'), help='Jenis Kelamin')
  age = st.number_input('Age', min_value=5, step=5)
  height = st.number_input('Height', min_value=5, step=5)
  weight = st.number_input('Weight', min_value=1)
  waistline = st.number_input('Waistline', min_value=0)
  sight_left = st.number_input('Sight Left', min_value=0)
  sight_right = st.number_input('Sight Right', min_value=0)
  hear_left = st.slider('Hear Left', min_value=1, max_value=2)
  hear_right = st.slider('Hear Right', min_value=1, max_value=2)
  sbp = st.number_input('Systolic Blood Pressure[mmHg]', min_value=0)
  dbp = st.number_input('Diastolic blood pressure[mmHg]', min_value=0)
  blds = st.number_input('BLDS or FSG(fasting blood glucose)[mg/dL]', min_value=0)
  tot_chole = st.number_input('Total cholesterol[mg/dL]', min_value=0)
  HDL_chole = st.number_input('HDL cholesterol[mg/dL]', min_value=0)
  LDL_chole = st.number_input('LDL cholesterol[mg/dL]', min_value=0)
  triglyceride = st.number_input('triglyceride[mg/dL]', min_value=0)
  hemoglobin = st.number_input('hemoglobin[g/dL]', min_value=0)
  urine_protein = st.radio('Protein in urine', (1, 2, 3, 4, 5, 6), help='1(-), 2(+/-), 3(+1), 4(+2), 5(+3), 6(+4)')
  serum_creatinine = st.number_input('Serum(blood) creatinine[mg/dL]	', min_value=0)
  SGOT_AST = st.number_input('SGOT(Glutamate-oxaloacetate transaminase) AST(Aspartate transaminase)[IU/L]', min_value=0)
  SGOT_ALT = st.number_input('ALT(Alanine transaminase)[IU/L]', min_value=0)
  gamma_GTP = st.number_input('y-glutamyl transpeptidase[IU/L]', min_value=0)
  smoking_state = st.radio('Smoking State', (1, 2, 3), help='Status Perokok(1=never, 2=used smoke but quit, 3=still smoke)')

  sub = st.form_submit_button('Predict')

data_inference = {'sex' :sex,
                  'age' :age,
                  'height' : height,
                  'weight' : weight,
                  'waistline' : waistline,
                  'sight_left' : sight_left,
                  'sight_right' : sight_right,
                  'hear_left' : hear_left,
                  'hear_right' : hear_right,
                  'SBP' : sbp,
                  'DBP' : dbp,
                  'BLDS' : blds,
                  'tot_chole' : tot_chole,
                  'HDL_chole' : HDL_chole,
                  'LDL_chole' : LDL_chole,
                  'triglyceride' : triglyceride,
                  'hemoglobin' : hemoglobin,
                  'urine_protein' : urine_protein,
                  'serum_creatinine' : serum_creatinine,
                  'SGOT_AST' :  SGOT_AST,
                  'SGOT_ALT' : SGOT_ALT,
                  'gamma_GTP' : gamma_GTP,
                  'Smoking State' : smoking_state
                 }

df_inference = pd.DataFrame([data_inference])
st.dataframe(df_inference)

if sub:
 
  inference_final = preprocessing_pipeline.transform(df_inference)

  # Predict Model against Inference
  y_pred_inference = rf_gridcv_best.predict(inference_final)
  y_pred_inference

  if y_pred_inference[0] == 0:
    hasil_predik = 'Not Drinker'
  elif y_pred_inference[0] == 1:
    hasil_predik = 'Drinker'
  st.write('Hasil Prediksi adalah', hasil_predik)