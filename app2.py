import joblib
import pandas as pd
import streamlit as st
import numpy as np

# โหลดโมเดล
model = joblib.load('./Model/Education_recommen_logis.pkl')
model2 = joblib.load('./Model/Education_recommen_RandomForest2.pkl')
data = pd.read_csv('./dataSet/Resampled_Data.csv')
scaler = joblib.load('Model/scaler.pkl')

# จัดการหน้า
if 'page' not in st.session_state:
    st.session_state.page = 1

def next_page():
    st.session_state.page += 1

def previous_page():
    if st.session_state.page > 1:
        st.session_state.page -= 1

# ฟังก์ชันทำนายผล
def predict(features):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    prediction_proba = model.predict_proba(features_scaled)
    return prediction[0], max(prediction_proba[0])

# ---------------------------------------------------
# หน้าที่ 1: กรอกข้อมูลส่วนตัว
# ---------------------------------------------------
if st.session_state.page == 1:
    st.title('GradPredict Recommendation System')
    st.header('ข้อมูลส่วนตัว')

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

    with st.container():
        st.selectbox('นักศึกษาสังกัดหลักสูตร', list(course_mapping.keys()), key='course')
        st.selectbox('เพศ', list(sex_mapping.keys()), key='sex')
        st.number_input('อายุ (ปี)', 0, 80, key='age')
        st.selectbox('สถานภาพสมรส', list(status_mapping.keys()), key='status')
        st.selectbox('ท่านมีความต้องการสำเร็จการศึกษาตามระยะเวลาที่หลักสูตรกำหนดหรือไม่', list(time_mapping.keys()), key='time')
        st.selectbox('ต้องปฏิบัติงานประจำควบคู่ไปด้วยหรือไม่', list(work_mapping.keys()), key='work')

    if st.button('ถัดไป'):
        st.session_state.course_num = course_mapping[st.session_state.course]
        st.session_state.sex_num = sex_mapping[st.session_state.sex]
        st.session_state.status_num = status_mapping[st.session_state.status]
        st.session_state.time_num = time_mapping[st.session_state.time]
        st.session_state.work_num = work_mapping[st.session_state.work]
        next_page()

# ---------------------------------------------------
# หน้าที่ 2: แบบสอบถามปัจจัย (แบบตาราง)
# ---------------------------------------------------
elif st.session_state.page == 2:
    st.header('ปัจจัยส่งผลต่อการสำเร็จการศึกษาของนักศึกษา')
    st.markdown("โปรดเลือกระดับความสำคัญที่ตรงกับความคิดเห็นของท่านมากที่สุดเพียงระดับเดียว")

    st.markdown("""
        <style>
        table, th, td {
            border: 1px solid #888;
            border-collapse: collapse;
            padding: 6px;
            text-align: center;
        }
        th {
            background-color: #f0f0f0;
        }
        .section-header {
            background-color: #d9e1f2;
            font-weight: bold;
            text-align: left;
        }
        </style>
    """, unsafe_allow_html=True)

    features_mapping = {'น้อย': 3, 'ปานกลาง': 4, 'มาก': 5}

    # ====== จัดหมวด ======
    sections = {
        "1. ด้านลักษณะการจัดการเรียนการสอนและหลักสูตร": [
            ('ความรู้ความเข้าใจแผนการเรียนที่กำหนดไว้ในหลักสูตร', 'knowledge_course'),
            ('ความรู้และความเข้าใจในการเรียนในแต่ละรายวิชา', 'knowledge_subject'),
            ('หลักสูตรมีการจัดกิจกรรมการเรียนการสอนที่เน้นผู้เรียนเป็นสำคัญ', 'student_centered'),
            ('หลักสูตรมีความพร้อมของสถานที่ เครื่องมือ และอุปกรณ์การเรียน', 'facility_support'),
            ('การให้ความสนับสนุนข้อมูลต่างๆ ของเจ้าหน้าที่บัณฑิตศึกษา', 'grad_office_support'),
        ],
        "2. อาจารย์ที่ปรึกษาวิทยานิพนธ์": [
            ('การจัดเวลาให้นักศึกษาเข้าพบ', 'meeting_time'),
            ('การวางแผนการเรียนระหว่างอาจารย์กับนักศึกษา', 'study_plan'),
            ('การติดตามการทำวิทยานิพนธ์ของนักศึกษาอย่างสม่ำเสมอ', 'thesis_followup'),
            ('ความรอบรู้ และความชำนาญของอาจารย์ที่ปรึกษาในหัวข้อวิทยานิพนธ์', 'advisor_expertise'),
            ('การสนับสนุนให้นักศึกษาขอทุนสนับสนุนการวิจัย', 'research_funding'),
        ],
        "3. การทำวิทยานิพนธ์และการเผยแพร่ผลงาน": [
            ('การหาหัวข้อวิทยานิพนธ์', 'graduation_factors'),
            ('การวิเคราะห์ข้อมูล', 'data_analysis'),
            ('การเขียนวิทยานิพนธ์', 'thesis_writing'),
            ('การเผยแพร่ผลงานวิทยานิพนธ์', 'thesis_publication'),
        ],
        "4. ความพร้อมของนักศึกษา": [
            ('มีวินัยในตนเอง', 'self_discipline'),
            ('มีความใฝ่รู้ใฝ่เรียน', 'curiosity'),
            ('ท่านมีการเข้าพบอาจารย์ที่ปรึกษา', 'advisor_meeting'),
            ('ความรู้และความสามารถในการวิจัย', 'research_skills'),
            ('มีความสามารถในการสืบค้นข้อมูลในการทำวิทยานิพนธ์', 'information_retrieval'),
        ],
        "5. ปัจจัยแวดล้อมที่มีผลต่อการสำเร็จการศึกษา": [
            ('ท่านมีทักษะการเขียน การวิเคราะห์ สรุปผล', 'writing_skills'),
            ('แรงผลักดันจากครอบครัว', 'family_support'),
            ('สภาพคล่องด้านการเงิน', 'financial_situation'),
        ]
    }

    # ====== แสดงตาราง ======
    for section_title, items in sections.items():
        st.markdown(f"<div class='section-header'>{section_title}</div>", unsafe_allow_html=True)
        html_table = "<table><tr><th>หัวข้อ</th><th>มาก</th><th>ปานกลาง</th><th>น้อย</th></tr>"
        for text, key in items:
            choice = st.radio(f"{text}", ['มาก', 'ปานกลาง', 'น้อย'], horizontal=True, key=key)
            st.session_state[key] = features_mapping[choice]
            html_table += f"<tr><td style='text-align:left'>{text}</td><td>{'✓' if choice=='มาก' else ''}</td><td>{'✓' if choice=='ปานกลาง' else ''}</td><td>{'✓' if choice=='น้อย' else ''}</td></tr>"
        html_table += "</table>"
        st.markdown(html_table, unsafe_allow_html=True)
        st.markdown("---")

    if st.button("ถัดไป"):
        next_page()

    if st.button("ย้อนกลับ"):
        previous_page()

# ---------------------------------------------------
# หน้าที่ 3: ทำนายผล
# ---------------------------------------------------
elif st.session_state.page == 3:
    st.header('ผลการทำนายการสำเร็จการศึกษา')

    learning_factors = [
        key for key in st.session_state.keys() if key not in 
        ['page', 'course', 'sex', 'age', 'status', 'time', 'work',
         'course_num', 'sex_num', 'status_num', 'time_num', 'work_num']
    ]

    features = [
        st.session_state.course_num,
        st.session_state.sex_num,
        st.session_state.age,
        st.session_state.status_num,
        st.session_state.time_num,
        *[st.session_state[key] for key in learning_factors],
        st.session_state.work_num
    ]

    prediction, confidence = predict(features)
    prediction_text = '✅ จบภายในระยะเวลาที่กำหนด' if prediction == 1 else '⚠️ จบช้ากว่าระยะเวลาที่กำหนด'

    st.subheader(f"ผลการทำนาย: {prediction_text}")
    st.write(f"ความมั่นใจของแบบจำลอง: {confidence * 100:.2f}%")

    # RandomForest model
    prediction2 = model2.predict([features])
    import math
    months = prediction2[0]
    years = math.floor(months / 12)
    months_only = months % 12
    days = round((months_only - math.floor(months_only)) * 30)

    st.info(f"คาดว่าจะสำเร็จการศึกษาในประมาณ: {years} ปี {int(months_only)} เดือน {days} วัน")

    if st.button("ย้อนกลับ"):
        previous_page()
