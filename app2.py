import joblib
import pandas as pd
import streamlit as st
import numpy as np

# โหลดโมเดล
model = joblib.load('./Model/Education_recommen_logis.pkl')
model2 = joblib.load('./Model/Education_recommen_RandomForest2.pkl')

# โหลด dataset
data = pd.read_csv('./dataSet/Resampled_Data.csv')

# -------------------- Mapping --------------------
features_mapping = {
    "น้อย": 0,
    "กลาง": 1,
    "มาก": 2
}

# -------------------- Session State --------------------
if "page" not in st.session_state:
    st.session_state.page = 1

def next_page():
    st.session_state.page += 1

def prev_page():
    st.session_state.page -= 1

# -------------------- PAGE 1 --------------------
if st.session_state.page == 1:
    st.title("ระบบแนะนำแผนการเรียน")

    st.markdown("### ข้อมูลผู้ใช้")
    name = st.text_input("ชื่อ - นามสกุล", key="name")
    student_id = st.text_input("รหัสนักศึกษา", key="student_id")
    st.session_state.major = st.selectbox(
        "เลือกสาขา",
        ["IT", "CS", "DSI"],
        key="major"
    )

    if st.button("ถัดไป"):
        if name and student_id:
            next_page()
        else:
            st.warning("กรุณากรอกชื่อและรหัสนักศึกษา")

# -------------------- PAGE 2 --------------------
elif st.session_state.page == 2:
    st.header('กรุณากรอกระดับความพร้อมในการเรียน')

    learning_factors = [
        ('แรงผลักดันจากครอบครัว', 'family_support'),
        ('สภาพคล่องด้านการเงิน', 'financial_situation'),
        ('ความรู้ความเข้าใจแผนการเรียนที่กำหนดไว้ในหลักสูตร', 'knowledge_course'),
        ('ความรู้และความเข้าใจในการเรียนในแต่ละรายวิชา', 'knowledge_subject'),
        ('ทักษะในการวิเคราะห์และแก้ปัญหา', 'problem_solving'),
        ('ทักษะในการทำงานร่วมกันเป็นทีม', 'teamwork'),
        ('ทักษะด้านการบริหารเวลา', 'time_management'),
        ('สุขภาพกายและสุขภาพจิต', 'health'),
        ('แรงบันดาลใจและเป้าหมายในการเรียน', 'motivation'),
        ('สภาพแวดล้อมและสิ่งแวดล้อมที่เอื้อต่อการเรียน', 'environment')
    ]

    levels = ["น้อย", "กลาง", "มาก"]

    # สร้างหัวตาราง
    cols = st.columns([3, 1, 1, 1])
    cols[0].write("**คำถาม**")
    for i, level in enumerate(levels):
        cols[i+1].write(f"**{level}**")

    # สร้างแถวคำถาม
    for question, key in learning_factors:
        cols = st.columns([3, 1, 1, 1])
        cols[0].write(question)

        for i, level in enumerate(levels):
            checked = st.session_state.get(key) == level
            if cols[i+1].checkbox("", key=f"{key}_{level}", value=checked):
                st.session_state[key] = level
                # ยกเลิกช่องอื่นในแถวเดียวกัน
                for other in levels:
                    if other != level:
                        st.session_state[f"{key}_{other}"] = False

    # ตรวจว่าตอบครบหรือยัง
    all_filled = True
    for _, key in learning_factors:
        if st.session_state.get(key) is None:
            all_filled = False
        else:
            st.session_state[key] = features_mapping[st.session_state[key]]

    if all_filled and st.button("ถัดไป"):
        next_page()
    elif not all_filled:
        st.warning("กรุณาตอบทุกคำถามก่อนดำเนินการต่อ")

    if st.button("ย้อนกลับ"):
        prev_page()

# -------------------- PAGE 3 --------------------
elif st.session_state.page == 3:
    st.header("เลือกวิชาที่สนใจ")

    subjects = ["CS101", "CS102", "CS103", "DS201", "IT301"]
    chosen_subjects = st.multiselect("เลือกวิชา", subjects, key="subjects")

    if st.button("ถัดไป"):
        if chosen_subjects:
            next_page()
        else:
            st.warning("กรุณาเลือกอย่างน้อย 1 วิชา")

    if st.button("ย้อนกลับ"):
        prev_page()

# -------------------- PAGE 4 --------------------
elif st.session_state.page == 4:
    st.header("ผลการแนะนำ")

    # สมมุติใช้โมเดลทำนาย (ตัวอย่าง)
    features = [
        st.session_state.family_support,
        st.session_state.financial_situation,
        st.session_state.knowledge_course,
        st.session_state.knowledge_subject,
        st.session_state.problem_solving,
        st.session_state.teamwork,
        st.session_state.time_management,
        st.session_state.health,
        st.session_state.motivation,
        st.session_state.environment
    ]

    input_data = np.array(features).reshape(1, -1)

    pred1 = model.predict(input_data)[0]
    pred2 = model2.predict(input_data)[0]

    st.write(f"โมเดล Logistic Regression แนะนำ: {pred1}")
    st.write(f"โมเดล Random Forest แนะนำ: {pred2}")

    if st.button("ย้อนกลับ"):
        prev_page()
