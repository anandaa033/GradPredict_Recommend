import joblib
import pandas as pd
import streamlit as st
import numpy as np
import math

# Load the trained models
model = joblib.load('./Model/Education_recommen_logis.pkl')
model2 = joblib.load('./Model/Education_recommen_RandomForest2.pkl')

# Load the dataset used in train3.py
data = pd.read_csv('./dataSet/Resampled_Data.csv')

scaler = joblib.load('Model/scaler.pkl')  # Load the scaler used during model training

# Initialize session state to track which page the user is on
if 'page' not in st.session_state:
    st.session_state.page = 1

# Function to go to the next page
def next_page():
    st.session_state.page += 1

# Function to go back to the previous page
def previous_page():
    st.session_state.page = max(1, st.session_state.page - 1)


# Function for predicting graduation status
def predict(features):
    features_scaled = scaler.transform([features])  # Scale input data
    prediction = model.predict(features_scaled)  # Predict graduation status
    prediction_proba = model.predict_proba(features_scaled)  # Predict probabilities
    return prediction[0], max(prediction_proba[0])

# Streamlit app title
st.title('GradPredict Recommendation System')

# Mapping for user inputs
course_mapping = {
    'หลักสูตร วท.ม.สาขาวิชาคณิตศาสตร์ประยุกต์และวิทยาการคำนวณ': 1,
    'หลักสูตร วท.ม.สาขาวิชาเคมีประยุกต์': 2,
    'หลักสูตร ปร.ด.สาขาวิชาเทคโนโลยียาง': 3,
    'หลักสูตร วท.ม.สาขาวิชาวิทยาศาสตร์และเทคโนโลยีการเกษตร': 4,
    'หลักสูตร ปร.ด.สาขาวิชาการเพาะเลี้ยงสัตว์น้ำและทรัพยากรประมง': 5,
    'หลักสูตร วท.ม.สาขาวิชาเทคโนโลยียาง': 6,
    'หลักสูตรวิทยาศาสตรมหาบัณฑิต สาขาวิชาวิทยาการคำนวณและปัญญาประดิษฐ์': 7,
    'หลักสูตรวิศวกรรมศาสตรมหาบัณฑิต สาขาวิชาการจัดการอุตสาหกรรม': 8,
}

sex_mapping = {'ชาย': 0, 'หญิง': 1, 'เพศทางเลือก': 2}
status_mapping = {'โสด': 0, 'สมรส': 1}
time_mapping = {'ต้องการ': 2, 'ไม่แน่ใจ': 1, 'ไม่ต้องการ': 0}
work_mapping = {'ปฏิบัติ': 1, 'ไม่ปฏิบัติ': 0}
features_mapping = {'น้อย': 3, 'ปานกลาง': 4, 'มาก': 5}

# ---------------------------
# PAGE 1
# ---------------------------
if st.session_state.page == 1:
    st.header('กรุณากรอกข้อมูลส่วนตัว')
    course = st.selectbox('นักศึกษาสังกัดหลักสูตร', list(course_mapping.keys()))
    sex = st.selectbox('เพศ', list(sex_mapping.keys()))
    age = st.number_input('อายุ (ปี)', min_value=0, max_value=80, step=1)
    status = st.selectbox('สถานภาพสมรส (ปัจจุบัน)', list(status_mapping.keys()))
    time = st.selectbox('ท่านมีความต้องการสำเร็จการศึกษาตามระยะเวลาที่หลักสูตรกำหนดหรือไม่', list(time_mapping.keys()))
    work = st.selectbox('ต้องปฏิบัติงานประจำควบคู่ไปด้วยหรือไม่', list(work_mapping.keys()))

    if st.button('ถัดไป'):
        st.session_state.course_num = course_mapping[course]
        st.session_state.sex_num = sex_mapping[sex]
        st.session_state.age = age
        st.session_state.status_num = status_mapping[status]
        st.session_state.time_num = time_mapping[time]
        st.session_state.work_num = work_mapping[work]
        next_page()

# ---------------------------
# PAGE 2
# ---------------------------
elif st.session_state.page == 2:
    st.header('กรุณากรอกระดับความพร้อมในการเรียน')

    learning_factors = [
        ('ความรู้ความเข้าใจแผนการเรียนที่กำหนดไว้ในหลักสูตร', 'knowledge_course'),
        ('ความรู้และความเข้าใจในการเรียนในแต่ละรายวิชา', 'knowledge_subject'),
        ('หลักสูตรมีการจัดกิจกรรมการเรียนการสอนที่เน้นผู้เรียนเป็นสำคัญ', 'student_centered'),
        ('หลักสูตรมีความพร้อมของสถานที่ เครื่องมือ และอุปกรณ์การเรียน', 'facility_support'),
        ('การให้ความสนับสนุนข้อมูลต่างๆ ของเจ้าหน้าที่บัณฑิตศึกษา', 'grad_office_support'),
        ('การจัดเวลาให้นักศึกษาเข้าพบ', 'meeting_time'),
        ('การวางแผนการเรียนระหว่างอาจารย์กับนักศึกษา', 'study_plan'),
        ('การติดตามการทำวิทยานิพนธ์ของนักศึกษาอย่างสม่ำเสมอ', 'thesis_followup'),
        ('การมีความรู้ความเข้าใจในกฎระเบียบ และข้อกำหนดเกี่ยวกับวิทยานิพนธ์', 'thesis_regulations'),
        ('ความรอบรู้ และความชำนาญของอาจารย์ที่ปรึกษาในหัวข้อวิทยานิพนธ์', 'advisor_expertise'),
        ('ความพร้อมในการเป็นอาจารย์ที่ปรึกษา', 'advisor_availability'),
        ('การสนับสนุนให้นักศึกษาขอทุนสนับสนุนการวิจัย', 'research_funding'),
        ('การสนับสนุนให้นักศึกษานำเสนอผลงานในที่ประชุมหรือตีพิมพ์ในวารสารวิชาการ', 'presentation_support'),
        ('ความชื่นชอบอาจารย์ผู้สอน และอาจารย์ที่ปรึกษา', 'teacher_satisfaction'),
        ('การเขียนเค้าโครงวิทยานิพนธ์', 'thesis_outline'),
        ('การวางแผนและการดำเนินการวิทยานิพนธ์', 'thesis_planning'),
        ('สิ่งเร้าที่ทำให้นักศึกษามีพฤติกรรมในการอยากเรียนและศึกษาค้นคว้า', 'learning_motivation'),
        ('การเก็บรวบรวมข้อมูล', 'data_collection'),
        ('การวิเคราะห์ข้อมูล', 'data_analysis'),
        ('การเขียนวิทยานิพนธ์', 'thesis_writing'),
        ('การสอบโครงร่างวิทยานิพนธ์', 'thesis_proposal'),
        ('การสอบป้องกันร่างวิทยานิพนธ์', 'thesis_defense'),
        ('การส่งรูปเล่มวิทยานิพนิพนธ์', 'thesis_submission'),
        ('การเผยแพร่ผลงานวิทยานิพนธ์', 'thesis_publication'),
        ('มีวินัยในตนเอง', 'self_discipline'),
        ('มีความใฝ่รู้ใฝ่เรียน', 'curiosity'),
        ('ท่านมีการเข้าพบอาจารย์ที่ปรึกษา หรือติดต่อประสานงานกับอาจารย์ที่ปรึกษา', 'advisor_meeting'),
        ('ท่านมีการวางแผนการเรียนระหว่างนักศึกษากับอาจารย์', 'study_planning'),
        ('ความรู้และความสามารถในการวิจัย เช่น การวางแผน/เก็บตัวอย่าง/ทำแลป', 'research_skills'),
        ('มีความสามารถในการสืบค้นข้อมูลในการทำวิทยานิพนธ์ และแหล่งเรียนรู้ต่างๆ', 'information_retrieval'),
        ('ท่านมีทักษะการเขียน การวิเคราะห์ สรุปผล', 'writing_skills'),
        ('เมื่อท่านสำเร็จการศึกษาระดับปริญญาตรี ท่านมีความประสงค์จะศึกษาต่อระดับบัณฑิตศึกษา', 'postgraduate_interest'),
        ('มีความต้องการหาประสบการณ์และหาความรู้เพิ่มเติม', 'knowledge_seeking'),
        ('มีความเป็นไปได้มากน้อยเพียงใดที่คุณจะแนะนำเราให้กับเพื่อนหรือผู้ร่วมงาน', 'recommendation_likelihood'),
        ('แรงผลักดันจากครอบครัว', 'family_support'),
        ('สภาพคล่องด้านการเงิน', 'financial_situation')
    ]

    st.session_state.learning_factors = learning_factors
    options = list(features_mapping.keys())

    # สร้าง DataFrame
    df_init = pd.DataFrame({
        "คำถาม": [q for q, _ in learning_factors],
        "คำตอบ": [None]*len(learning_factors)
    })

    edited_df = st.data_editor(
        df_init,
        column_config={
            "คำตอบ": st.column_config.SelectboxColumn("คำตอบ", options=options)
        },
        hide_index=True,
    )

    # เก็บค่าลง session_state
    for (_, key), value in zip(learning_factors, edited_df["คำตอบ"]):
        if value:
            st.session_state[key] = features_mapping[value]

    all_filled = all(st.session_state.get(key) for _, key in learning_factors)

    if all_filled and st.button('ถัดไป'):
        next_page()
    elif not all_filled:
        st.warning("กรุณาตอบทุกคำถามก่อนดำเนินการต่อ")

# ---------------------------
# PAGE 3
# ---------------------------
elif st.session_state.page == 3:
    st.subheader('ทำนายผล')

    learning_factors = st.session_state.learning_factors
    features = [
        st.session_state.course_num,
        st.session_state.sex_num,
        st.session_state.age,
        st.session_state.status_num,
        st.session_state.time_num,
        *[st.session_state[key] for _, key in learning_factors],
        st.session_state.work_num
    ]

    prediction, confidence = predict(features)
    prediction_text = 'จบช้ากว่าระยะเวลาที่กำหนด' if prediction == 0 else 'จบภายในระยะเวลาที่กำหนด'
    
    st.write(f'ผลการทำนาย: **{prediction_text}**')
    st.write(f'ความน่าจะเป็น: **{confidence * 100:.2f}%**')

    # Predict using RandomForest
    prediction2 = model2.predict([features])
    months = prediction2[0]
    years = math.floor(months / 12)
    remaining_months = months % 12
    months_only = math.floor(remaining_months)
    days = round((remaining_months - months_only) * 30)

    st.write(f'การคาดการณ์จำนวนปีที่จบ: {years} ปี {months_only} เดือน {days} วัน')

    if st.button('ย้อนกลับ'):
        previous_page()
