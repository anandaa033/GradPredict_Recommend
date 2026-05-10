from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import numpy as np
import math

from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score
)

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC

from xgboost import XGBClassifier

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="GradPredict Recommendation System",
    layout="wide"
)

# =========================================================
# LOAD MODEL + DATA
# =========================================================

_BASE = Path(__file__).resolve().parent

model = joblib.load(_BASE / 'Model' / 'Education_recommen_logis.pkl')

model2 = joblib.load(_BASE / 'Model' / 'Education_recommen_RandomForest2.pkl')

scaler = joblib.load(_BASE / 'Model' / 'scaler.pkl')

TARGET_COL = 'ระยะเวลาในเดือน'

data = pd.read_csv(
    _BASE / 'dataSet' / 'Resampled_Data.csv',
    encoding='utf-8'
)

# =========================================================
# PREPARE DATA
# =========================================================

_feature_names = list(scaler.feature_names_in_)
_missing = set(_feature_names) - set(data.columns)
if _missing:
    raise ValueError(
        'ไฟล์ CSV ขาดคอลัมน์ให้ครบเมื่อเทียบกับ scaler โหลดจาก Model/scaler.pkl: '
        + ', '.join(sorted(_missing)[:10])
        + (' ...' if len(_missing) > 10 else '')
    )
if TARGET_COL not in data.columns:
    raise KeyError(
        f"ไฟล์ CSV ต้องมีคอลัมน์เป้าหมายชื่อ '{TARGET_COL}'"
    )
X = data.reindex(columns=_feature_names, copy=False)

X_scaled = scaler.transform(X)

# ชุดข้อมูลที่เทรนโมเดลใช้คอลัมน์นี้เป็นทั้งป้ายหมวด (0/1) และค่าเป้าหมายถดถอย
y_class = data[TARGET_COL].astype(int).to_numpy()

y_reg = data[TARGET_COL].astype(float).to_numpy()

# =========================================================
# K-FOLD EVALUATION
# =========================================================

def _stratified_kfold():
    return StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )


def _classification_cv_summary(estimator, X, y, skf):
    """เฉลี่ย ± ส่วนเบี่ยงเบนมาตรฐานจาก Stratified K-fold."""
    out = {}
    for name, scoring in [
        ('accuracy', 'accuracy'),
        ('precision', 'precision'),
        ('recall', 'recall'),
        ('f1', 'f1'),
    ]:
        scores = cross_val_score(
            clone(estimator),
            X,
            y,
            cv=skf,
            scoring=scoring
        )
        out[f'{name}_mean'] = float(np.nanmean(scores))
        out[f'{name}_std'] = float(np.nanstd(scores))
    return out


@st.cache_data
def evaluate_models():
    skf = _stratified_kfold()

    classifiers_spec = [
        (
            'Logistic Regression',
            LogisticRegression(max_iter=1000, random_state=42)
        ),
        (
            'XGBoost',
            XGBClassifier(
                n_estimators=120,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
            )
        ),
        (
            'SVM (RBF)',
            SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        ),
    ]

    classifiers = {
        label: _classification_cv_summary(est, X_scaled, y_class, skf)
        for label, est in classifiers_spec
    }

    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    rf_r2 = cross_val_score(
        rf_model,
        X_scaled,
        y_reg,
        cv=kf,
        scoring='r2'
    )

    rf_mse = cross_val_score(
        rf_model,
        X_scaled,
        y_reg,
        cv=kf,
        scoring='neg_mean_squared_error'
    )

    rf_mse = -rf_mse

    rf_rmse = np.sqrt(rf_mse)

    rf_mae = cross_val_score(
        rf_model,
        X_scaled,
        y_reg,
        cv=kf,
        scoring='neg_mean_absolute_error'
    )

    rf_mae = -rf_mae

    return {
        'classifiers': classifiers,
        'regression': {
            'r2_mean': float(np.nanmean(rf_r2)),
            'r2_std': float(np.nanstd(rf_r2)),
            'rmse_mean': float(np.nanmean(rf_rmse)),
            'rmse_std': float(np.nanstd(rf_rmse)),
            'mae_mean': float(np.nanmean(rf_mae)),
            'mae_std': float(np.nanstd(rf_mae)),
        },
    }


metrics = evaluate_models()


@st.cache_resource
def fit_aux_classifiers():
    """ฟิต XGBoost / SVM เต็มชุดสำหรับทำนายแบบเรียลไทม์"""
    xgb = XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
    )
    xgb.fit(X_scaled, y_class)
    svm = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42
    )
    svm.fit(X_scaled, y_class)
    return xgb, svm


_xgb_live, _svm_live = fit_aux_classifiers()

# =========================================================
# SESSION
# =========================================================

if 'page' not in st.session_state:
    st.session_state.page = 1

def next_page():
    st.session_state.page += 1

def previous_page():
    st.session_state.page = 2

# =========================================================
# PREDICT FUNCTION
# =========================================================

def predict(features):

    features_scaled = scaler.transform([features])

    lr_pred = int(model.predict(features_scaled)[0])
    lr_proba = float(np.max(model.predict_proba(features_scaled)[0]))

    xgb_pred = int(_xgb_live.predict(features_scaled)[0])
    xgb_proba = float(np.max(_xgb_live.predict_proba(features_scaled)[0]))

    svm_pred = int(_svm_live.predict(features_scaled)[0])
    svm_proba = float(np.max(_svm_live.predict_proba(features_scaled)[0]))

    prediction_month = float(model2.predict(features_scaled)[0])

    return {
        'logistic': (lr_pred, lr_proba),
        'xgboost': (xgb_pred, xgb_proba),
        'svm': (svm_pred, svm_proba),
        'months': prediction_month,
    }

# =========================================================
# TITLE
# =========================================================

st.title('🎓 GradPredict Recommendation System')

# =========================================================
# MAPPING
# =========================================================

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

sex_mapping = {
    'ชาย': 0,
    'หญิง': 1,
    'เพศทางเลือก': 2
}

status_mapping = {
    'โสด': 0,
    'สมรส': 1
}

time_mapping = {
    'ต้องการ': 2,
    'ไม่แน่ใจ': 1,
    'ไม่ต้องการ': 0
}

work_mapping = {
    'ปฏิบัติ': 1,
    'ไม่ปฏิบัติ': 0
}

# =========================================================
# QUESTIONS — ลำดับให้ตรงกับฟีเจอร์คอลัมน์ที่ 5–42 ของชุดข้อมูลที่เทรน (รวม 38 ข้อ)
# =========================================================

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

        ('ความพร้อมในการเป็นอาจารย์ที่ปรึกษา เช่นความพร้อมด้านวิชาการ, ความพร้อมด้านการให้คำปรึกษา, ความพร้อมด้านจริยธรรมทางวิชาการและวิจัย เป็นต้น', 'advisor_availability'),

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

        ('การสอบป้องกันวิทยานิพนธ์', 'thesis_defense'),

        ('การส่งรูปเล่มวิทยานิพนธ์', 'thesis_submission'),

        ('รูปแบบการเผยแพร่ผลงานวิทยานิพนธ์ เช่น การเผยแพร่ในรูปของบทความวารสาร, การเผยแพร่ในการประชุมวิชาการ เป็นต้น', 'thesis_publication')
    ],

    "4. ความพร้อมของนักศึกษา": [

        ('ท่านมีความรู้ความเข้าใจในกฎระเบียบและข้อกำหนดเกี่ยวกับวิทยานิพนธ์', 'rules_comprehension'),

        ('มีวินัยในตนเอง', 'self_discipline'),

        ('มีความใฝ่รู้ใฝ่เรียน', 'curiosity'),

        ('ท่านมีการเข้าพบอาจารย์ที่ปรึกษา หรือติดต่อประสานงานกับอาจารย์ที่ปรึกษา', 'advisor_meeting'),

        ('ท่านมีการวางแผนการเรียนระหว่างนักศึกษากับอาจารย์', 'study_planning'),

        ('ท่านมีความรู้และทักษะในการวิจัย เช่น การวางแผน/เก็บตัวอย่าง/ทำปฏิบัติการ/ทดสอบระบบ', 'research_skills'),

        ('มีความสามารถในการสืบค้นข้อมูลในการทำวิทยานิพนธ์ และแหล่งเรียนรู้ต่างๆ', 'information_retrieval'),

        ('ท่านมีทักษะการเขียน การวิเคราะห์ สรุปผล', 'writing_skills')
    ],

    "5. ปัจจัยแวดล้อมที่มีผลต่อการสำเร็จการศึกษาของนักศึกษาระดับบัณฑิตศึกษา": [

        ('เมื่อท่านสำเร็จการศึกษาระดับปริญญาตรี ท่านมีความประสงค์จะศึกษาต่อระดับบัณฑิตศึกษา', 'postgraduate_interest'),

        ('มีความต้องการหาประสบการณ์และหาความรู้เพิ่มเติม', 'knowledge_seeking'),

        ('มีความเป็นไปได้มากน้อยเพียงใดที่คุณจะแนะนำเราให้กับเพื่อนหรือผู้ร่วมงาน', 'recommendation_likelihood'),

        ('แรงผลักดันจากครอบครัว', 'family_support'),

        ('สภาพคล่องด้านการเงิน', 'financial_situation')
    ]
}

# =========================================================
# PAGE 1
# =========================================================

if st.session_state.page == 1:

    st.header('📋 ข้อมูลส่วนตัว')

    course = st.selectbox(
        'นักศึกษาสังกัดหลักสูตร',
        list(course_mapping.keys())
    )

    sex = st.selectbox(
        'เพศ',
        list(sex_mapping.keys())
    )

    age = st.number_input(
        'อายุ',
        min_value=0,
        max_value=80,
        step=1
    )

    status = st.selectbox(
        'สถานภาพสมรส',
        list(status_mapping.keys())
    )

    time = st.selectbox(
        'ความต้องการสำเร็จตามเวลา',
        list(time_mapping.keys())
    )

    work = st.selectbox(
        'มีงานประจำหรือไม่',
        list(work_mapping.keys())
    )

    if st.button('ถัดไป'):

        st.session_state.course_num = course_mapping[course]
        st.session_state.sex_num = sex_mapping[sex]
        st.session_state.age = age
        st.session_state.status_num = status_mapping[status]
        st.session_state.time_num = time_mapping[time]
        st.session_state.work_num = work_mapping[work]

        next_page()

# =========================================================
# PAGE 2
# =========================================================

elif st.session_state.page == 2:

    st.header("📝 แบบสอบถาม")

    options = ["น้อย", "ปานกลาง", "มาก"]

    with st.form("survey_form"):

        responses = {}

        for section_title, questions in sections.items():

            st.subheader(section_title)

            for q_text, key in questions:

                responses[key] = st.radio(
                    q_text,
                    options,
                    horizontal=True,
                    key=key
                )

        submitted = st.form_submit_button("ทำนายผล")

        if submitted:

            for k, v in responses.items():
                st.session_state[f"ans_{k}"] = v

            next_page()

# =========================================================
# PAGE 3
# =========================================================

elif st.session_state.page == 3:

    st.header("📊 ผลการทำนาย")

    answer_list = []

    for title, questions in sections.items():

        for q_text, key in questions:

            answer = st.session_state.get(
                f"ans_{key}",
                "ปานกลาง"
            )

            answer_list.append(answer)

    # =====================================================
    # MAP VALUE
    # =====================================================

    def map_value(val):

        return {
            'น้อย': 3,
            'ปานกลาง': 4,
            'มาก': 5
        }.get(val, 4)

    # =====================================================
    # FEATURE
    # =====================================================

    features = [

        st.session_state.course_num,

        st.session_state.sex_num,

        st.session_state.age,

        st.session_state.status_num,

        st.session_state.time_num,

        *[
            map_value(v)
            for v in answer_list
        ],

        st.session_state.work_num
    ]

    # =====================================================
    # SHOW MODEL EVALUATION
    # =====================================================

    st.subheader("📈 เปรียบเทียบโมเดลจำแนก (Stratified 5-fold CV)")

    def _pct_pm(mean_v, std_v):
        m = mean_v * 100
        s = std_v * 100
        if np.isnan(s):
            return f"{m:.2f}%"
        return f"{m:.2f}% ± {s:.2f}%"

    compare_rows = []
    for name, d in metrics['classifiers'].items():
        compare_rows.append({
            'โมเดล': name,
            'Accuracy': _pct_pm(d['accuracy_mean'], d['accuracy_std']),
            'Precision': _pct_pm(d['precision_mean'], d['precision_std']),
            'Recall': _pct_pm(d['recall_mean'], d['recall_std']),
            'F1': _pct_pm(d['f1_mean'], d['f1_std']),
        })
    st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)

    reg = metrics['regression']

    st.subheader("🌲 การถดถอย — Random Forest (5-fold CV)")

    st.write(
        f"**R²** {reg['r2_mean']:.3f} (±{reg['r2_std']:.3f}) "
        f"**RMSE** {reg['rmse_mean']:.2f} (±{reg['rmse_std']:.2f}) "
        f"**MAE** {reg['mae_mean']:.2f} (±{reg['mae_std']:.2f})"
    )

    st.divider()

    # =====================================================
    # PREDICT
    # =====================================================

    pred_pack = predict(features)

    def format_class_label(binary_val):
        return (
            '🎓 จบภายในระยะเวลาที่กำหนด'
            if binary_val == 1
            else '⏳ จบช้ากว่าระยะเวลาที่กำหนด'
        )

    st.markdown("### ผลทำนายจำแนก (แต่ละโมเดล)")

    c_lr, c_xgb, c_svm = st.columns(3)
    with c_lr:
        p, cf = pred_pack['logistic']
        st.metric("Logistic Regression", format_class_label(p), f"ความมั่นใจ {cf * 100:.1f}%")
    with c_xgb:
        p, cf = pred_pack['xgboost']
        st.metric("XGBoost", format_class_label(p), f"ความมั่นใจ {cf * 100:.1f}%")
    with c_svm:
        p, cf = pred_pack['svm']
        st.metric("SVM (RBF)", format_class_label(p), f"ความมั่นใจ {cf * 100:.1f}%")

    months = pred_pack['months']

    # =====================================================
    # CONVERT MONTH
    # =====================================================

    years = math.floor(months / 12)

    remaining_months = months % 12

    months_only = math.floor(
        remaining_months
    )

    days = round(
        (remaining_months - months_only)
        * 30
    )

    st.write(
        f"⏱ คาดว่าจะจบภายใน "
        f"{years} ปี "
        f"{months_only} เดือน "
        f"{days} วัน"
    )

    if st.button("ย้อนกลับ"):

        previous_page()