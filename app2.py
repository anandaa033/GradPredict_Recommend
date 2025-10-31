import joblib
import pandas as pd
import streamlit as st
import numpy as np
import math

# Load trained models
model = joblib.load('./Model/Education_recommen_logis.pkl')
model2 = joblib.load('./Model/Education_recommen_RandomForest2.pkl')
data = pd.read_csv('./dataSet/Resampled_Data.csv')
scaler = joblib.load('Model/scaler.pkl')

# Initialize session
if 'page' not in st.session_state:
    st.session_state.page = 1

def next_page():
    st.session_state.page += 1

def previous_page():
    st.session_state.page = 1

def predict(features):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    prediction_proba = model.predict_proba(features_scaled)
    return prediction[0], max(prediction_proba[0])

# Title
st.title('GradPredict Recommendation System')

# Mapping
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

# ----------------------------- PAGE 1 -----------------------------
if st.session_state.page == 1:
    st.header('กรุณากรอกข้อมูลส่วนตัว')
    st.markdown("""
        <style>
        .form-container {
            background-color: #f0f0f0;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        </style>
        <div class="form-container">
    """, unsafe_allow_html=True)

    course = st.selectbox('นักศึกษาสังกัดหลักสูตร', list(course_mapping.keys()))
    sex = st.selectbox('เพศ', list(sex_mapping.keys()))
    age = st.number_input('อายุ (ปี)', min_value=0, max_value=80, step=1)
    status = st.selectbox('สถานภาพสมรส (ปัจจุบัน)', list(status_mapping.keys()))
    time = st.selectbox('ความต้องการสำเร็จตามเวลาหลักสูตร', list(time_mapping.keys()))
    work = st.selectbox('ต้องปฏิบัติงานประจำควบคู่หรือไม่', list(work_mapping.keys()))

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button('ถัดไป'):
        st.session_state.course_num = course_mapping[course]
        st.session_state.sex_num = sex_mapping[sex]
        st.session_state.age = age
        st.session_state.status_num = status_mapping[status]
        st.session_state.time_num = time_mapping[time]
        st.session_state.work_num = work_mapping[work]
        next_page()

# ----------------------------- PAGE 2 -----------------------------
elif st.session_state.page == 2:
    st.header("ปัจจัยส่งผลต่อการสำเร็จการศึกษาของนักศึกษา")
    st.write("โปรดเลือกระดับความสำคัญที่ตรงกับความคิดเห็นของท่านมากที่สุดเพียงระดับเดียว")

    with st.form("survey_form"):
        # CSS
        st.markdown("""
        <style>
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
            font-size: 15px;
        }
        th, td {
            border: 1px solid #666;
            text-align: center;
            padding: 6px;
        }
        th { background-color: #e6e6e6; }
        .section-title {
            background-color: #d9ead3;
            text-align: left;
            font-weight: bold;
            padding: 8px;
            border: 1px solid #666;
        }
        </style>
        """, unsafe_allow_html=True)

        # Sections
        sections = {
            "1. ด้านลักษณะการจัดการเรียนการสอนและหลักสูตร": [
                ('ความรู้ความเข้าใจแผนการเรียนที่กำหนดไว้ในหลักสูตร', 'knowledge_course'),
                ('ความรู้และความเข้าใจในการเรียนในแต่ละรายวิชา', 'knowledge_subject'),
                ('หลักสูตรมีการจัดกิจกรรมการเรียนการสอนที่เน้นผู้เรียนเป็นสำคัญ', 'student_centered'),
                ('หลักสูตรมีความพร้อมของสถานที่ เครื่องมือ และอุปกรณ์การเรียน', 'facility_support'),
                ('การให้ความสนับสนุนข้อมูลต่างๆ ของเจ้าหน้าที่บัณฑิตศึกษา', 'grad_office_support')
            ],
            "2. อาจารย์ที่ปรึกษาวิทยานิพนธ์": [
                ('การจัดเวลาให้นักศึกษาเข้าพบ', 'meeting_time'),
                ('การวางแผนการเรียนระหว่างอาจารย์กับนักศึกษา', 'study_plan'),
                ('การติดตามการทำวิทยานิพนธ์อย่างสม่ำเสมอ', 'thesis_followup'),
                ('ความเข้าใจในกฎระเบียบและข้อกำหนดวิทยานิพนธ์', 'thesis_regulations'),
                ('ความชำนาญของอาจารย์ที่ปรึกษาในหัวข้อวิทยานิพนธ์', 'advisor_expertise'),
                ('ความพร้อมในการเป็นอาจารย์ที่ปรึกษา', 'advisor_availability'),
                ('การสนับสนุนให้นักศึกษาขอทุนวิจัย', 'research_funding'),
                ('การสนับสนุนให้นำเสนอผลงานวิชาการ', 'presentation_support'),
                ('ความชื่นชอบอาจารย์ผู้สอนและที่ปรึกษา', 'teacher_satisfaction')
            ],
            "3. การทำวิทยานิพนธ์และการเผยแพร่ผลงานของนักศึกษา": [ 
                ('การหาหัวข้อวิทยานิพนธ์', 'graduation_factors'), 
                ('การเขียนเค้าโครงวิทยานิพนธ์', 'thesis_outline'), 
                ('การวางแผนและการดำเนินการวิทยานิพนธ์', 'thesis_planning'), 
                ('สิ่งเร้าที่ทำให้นักศึกษามีพฤติกรรมในการอยากเรียนและศึกษาค้นคว้า', 'learning_motivation'), 
                ('การเก็บรวบรวมข้อมูล', 'data_collection'), 
                ('การวิเคราะห์ข้อมูล', 'data_analysis'), 
                ('การเขียนวิทยานิพนธ์', 'thesis_writing'), 
                ('การสอบโครงร่างวิทยานิพนธ์', 'thesis_proposal'), 
                ('การสอบป้องกันร่างวิทยานิพนธ์', 'thesis_defense'), 
                ('การส่งรูปเล่มวิทยานิพนธ์', 'thesis_submission'), 
                ('การเผยแพร่ผลงานวิทยานิพนธ์', 'thesis_publication') 
            ], 
            "4. ความพร้อมของนักศึกษา": [ 
                ('ท่านมีความรู้ความเข้าใจในกฎระเบียบและข้อกำหนดเกี่ยวกับวิทยานิพนธ์', 'rules_comprehension'), 
                ('มีวินัยในตนเอง', 'self_discipline'), 
                ('มีความใฝ่รู้ใฝ่เรียน', 'curiosity'), 
                ('ท่านมีการเข้าพบอาจารย์ที่ปรึกษา หรือติดต่อประสานงานกับอาจารย์ที่ปรึกษา', 'advisor_meeting'), 
                ('ท่านมีการวางแผนการเรียนระหว่างนักศึกษากับอาจารย์', 'study_planning'), 
                ('ความรู้และความสามารถในการวิจัย เช่น การวางแผน/เก็บตัวอย่าง/ทำแลป', 'research_skills'), 
                ('มีความสามารถในการสืบค้นข้อมูลในการทำวิทยานิพนธ์ และแหล่งเรียนรู้ต่างๆ', 'information_retrieval') 
            ], 
            "5. ปัจจัยแวดล้อมที่มีผลต่อการสำเร็จการศึกษาของนักศึกษาระดับบัณฑิตศึกษา": [ 
                ('ท่านมีทักษะการเขียน การวิเคราะห์ สรุปผล', 'writing_skills'), 
                ('เมื่อท่านสำเร็จการศึกษาระดับปริญญาตรี ท่านมีความประสงค์จะศึกษาต่อระดับบัณฑิตศึกษา', 'postgraduate_interest'), 
                ('มีความต้องการหาประสบการณ์และหาความรู้เพิ่มเติม', 'knowledge_seeking'), 
                ('มีความเป็นไปได้มากน้อยเพียงใดที่คุณจะแนะนำเราให้กับเพื่อนหรือผู้ร่วมงาน', 'recommendation_likelihood'), 
                ('แรงผลักดันจากครอบครัว', 'family_support'), ('สภาพคล่องด้านการเงิน', 'financial_situation') 
            ] 
        }

        options = ["น้อย", "ปานกลาง", "มาก"]

        # แสดงแบบสอบถาม
        for section_title, questions in sections.items():
            st.markdown(f"<div class='section-title'>{section_title}</div>", unsafe_allow_html=True)
            for q_text, key in questions:
                radio_key = f"radio_{key}"
                if radio_key not in st.session_state:
                    st.session_state[radio_key] = "ปานกลาง"
                st.radio(q_text, options,
                         index=options.index(st.session_state[radio_key]),
                         key=radio_key,
                         horizontal=True)

        submitted = st.form_submit_button("ถัดไป")

        if submitted:
            # ตรวจสอบว่ากรอกครบไหม
            all_filled = all(st.session_state.get(f"radio_{key}", "") for _, key in sum(sections.values(), []))
            if not all_filled:
                st.warning("กรุณาตอบทุกคำถามก่อนดำเนินการต่อ")
            else:
                next_page()

# ----------------------------- PAGE 3 -----------------------------
elif st.session_state.page == 3:
    st.subheader('ผลการทำนาย')

    sections = {
        "1. ด้านลักษณะการจัดการเรียนการสอนและหลักสูตร": [
            ('ความรู้ความเข้าใจแผนการเรียนที่กำหนดไว้ในหลักสูตร', 'knowledge_course'),
            ('ความรู้และความเข้าใจในการเรียนในแต่ละรายวิชา', 'knowledge_subject'),
            ('หลักสูตรมีการจัดกิจกรรมการเรียนการสอนที่เน้นผู้เรียนเป็นสำคัญ', 'student_centered'),
            ('หลักสูตรมีความพร้อมของสถานที่ เครื่องมือ และอุปกรณ์การเรียน', 'facility_support'),
            ('การให้ความสนับสนุนข้อมูลต่างๆ ของเจ้าหน้าที่บัณฑิตศึกษา', 'grad_office_support')
        ],
        "2. อาจารย์ที่ปรึกษาวิทยานิพนธ์": [
            ('การจัดเวลาให้นักศึกษาเข้าพบ', 'meeting_time'),
            ('การวางแผนการเรียนระหว่างอาจารย์กับนักศึกษา', 'study_plan'),
            ('การติดตามการทำวิทยานิพนธ์อย่างสม่ำเสมอ', 'thesis_followup'),
            ('ความเข้าใจในกฎระเบียบและข้อกำหนดวิทยานิพนธ์', 'thesis_regulations'),
            ('ความชำนาญของอาจารย์ที่ปรึกษาในหัวข้อวิทยานิพนธ์', 'advisor_expertise'),
            ('ความพร้อมในการเป็นอาจารย์ที่ปรึกษา', 'advisor_availability'),
            ('การสนับสนุนให้นักศึกษาขอทุนวิจัย', 'research_funding'),
            ('การสนับสนุนให้นำเสนอผลงานวิชาการ', 'presentation_support'),
            ('ความชื่นชอบอาจารย์ผู้สอนและที่ปรึกษา', 'teacher_satisfaction')
        ],
        "3. การทำวิทยานิพนธ์และการเผยแพร่ผลงานของนักศึกษา": [ 
            ('การหาหัวข้อวิทยานิพนธ์', 'graduation_factors'), 
            ('การเขียนเค้าโครงวิทยานิพนธ์', 'thesis_outline'), 
            ('การวางแผนและการดำเนินการวิทยานิพนธ์', 'thesis_planning'), 
            ('สิ่งเร้าที่ทำให้นักศึกษามีพฤติกรรมในการอยากเรียนและศึกษาค้นคว้า', 'learning_motivation'), 
            ('การเก็บรวบรวมข้อมูล', 'data_collection'), 
            ('การวิเคราะห์ข้อมูล', 'data_analysis'), 
            ('การเขียนวิทยานิพนธ์', 'thesis_writing'), 
            ('การสอบโครงร่างวิทยานิพนธ์', 'thesis_proposal'), 
            ('การสอบป้องกันร่างวิทยานิพนธ์', 'thesis_defense'), 
            ('การส่งรูปเล่มวิทยานิพนธ์', 'thesis_submission'), 
            ('การเผยแพร่ผลงานวิทยานิพนธ์', 'thesis_publication') 
        ], 
        "4. ความพร้อมของนักศึกษา": [ 
            ('ท่านมีความรู้ความเข้าใจในกฎระเบียบและข้อกำหนดเกี่ยวกับวิทยานิพนธ์', 'rules_comprehension'), 
            ('มีวินัยในตนเอง', 'self_discipline'), 
            ('มีความใฝ่รู้ใฝ่เรียน', 'curiosity'), 
            ('ท่านมีการเข้าพบอาจารย์ที่ปรึกษา หรือติดต่อประสานงานกับอาจารย์ที่ปรึกษา', 'advisor_meeting'), 
            ('ท่านมีการวางแผนการเรียนระหว่างนักศึกษากับอาจารย์', 'study_planning'), 
            ('ความรู้และความสามารถในการวิจัย เช่น การวางแผน/เก็บตัวอย่าง/ทำแลป', 'research_skills'), 
            ('มีความสามารถในการสืบค้นข้อมูลในการทำวิทยานิพนธ์ และแหล่งเรียนรู้ต่างๆ', 'information_retrieval') 
        ], 
        "5. ปัจจัยแวดล้อมที่มีผลต่อการสำเร็จการศึกษาของนักศึกษาระดับบัณฑิตศึกษา": [ 
            ('ท่านมีทักษะการเขียน การวิเคราะห์ สรุปผล', 'writing_skills'), 
            ('เมื่อท่านสำเร็จการศึกษาระดับปริญญาตรี ท่านมีความประสงค์จะศึกษาต่อระดับบัณฑิตศึกษา', 'postgraduate_interest'), 
            ('มีความต้องการหาประสบการณ์และหาความรู้เพิ่มเติม', 'knowledge_seeking'), 
            ('มีความเป็นไปได้มากน้อยเพียงใดที่คุณจะแนะนำเราให้กับเพื่อนหรือผู้ร่วมงาน', 'recommendation_likelihood'), 
            ('แรงผลักดันจากครอบครัว', 'family_support'), ('สภาพคล่องด้านการเงิน', 'financial_situation') 
        ] 
    }
    

    learning_factors = [q for section in sections.values() for q in section]
    def map_value(val):
        return {'น้อย': 3, 'ปานกลาง': 4, 'มาก': 5}.get(val, 4)

    for _, key in learning_factors:
        radio_key = f"radio_{key}"
        if radio_key not in st.session_state:
            st.session_state[radio_key] = "ปานกลาง"

    features = [
        st.session_state.course_num,
        st.session_state.sex_num,
        st.session_state.age,
        st.session_state.status_num,
        st.session_state.time_num,
        *[map_value(st.session_state[f"radio_{key}"]) for _, key in learning_factors],
        st.session_state.work_num
    ]

    prediction, confidence = predict(features)
    prediction_text = 'จบช้ากว่าระยะเวลาที่กำหนด' if prediction == 0 else 'จบภายในระยะเวลาที่กำหนด'

    st.write(f'ผลการทำนาย: **{prediction_text}**')
    st.write(f'ความน่าจะเป็น: **{confidence * 100:.2f}%**')

    prediction2 = model2.predict([features])
    months = prediction2[0]
    years = math.floor(months / 12)
    remaining_months = months % 12
    months_only = math.floor(remaining_months)
    days = round((remaining_months - months_only) * 30)
    st.write(f'คาดว่าจะจบการศึกษาใน {years} ปี {months_only} เดือน {days} วัน')

    if st.button('ย้อนกลับ'):
        previous_page()
