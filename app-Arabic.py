# app2.py
import streamlit as st
import pickle
import numpy as np
import time
import os

# إعداد الصفحة
st.set_page_config(page_title="📧 مصنّف الإيميلات", page_icon="📧", layout="centered")

# ===== تنسيقات CSS =====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
* { font-family: 'Tajawal', sans-serif !important; }

.stApp {
  background: linear-gradient(135deg, #0f172a, #1e293b, #0f172a);
  background-size: 400% 400%;
  animation: gradientBG 18s ease infinite;
  color: #F8FAFC !important;
}
@keyframes gradientBG {
  0% {background-position:0% 50%}
  50% {background-position:100% 50%}
  100% {background-position:0% 50%}
}

h1,h2,h3,h4,p,span,li,strong,em { color:#F8FAFC !important; }
label, .stTextArea label, .stTextInput label { color:#F8FAFC !important; opacity:1 !important; }

h1 {
  text-align:center; font-size:38px; font-weight:700;
  color:#60a5fa !important; margin-bottom:20px;
  text-shadow:0 0 14px #3b82f6;
}

.stTextArea textarea{
  background:#223043 !important;
  color:#ffffff !important;
  border:1px solid #7b8aa0 !important;
  border-radius:12px !important;
  font-size:17px !important; padding:12px !important;
}
.stTextArea textarea::placeholder{
  color:#cfe1ff !important; opacity:1 !important;
}

.stButton>button{
  background: linear-gradient(90deg, #3b82f6, #06b6d4);
  color:#ffffff !important; border:none; border-radius:25px;
  padding:12px 30px; font-size:18px; font-weight:700; transition:.3s;
}
.stButton>button:hover{ transform:scale(1.05); box-shadow:0 0 20px rgba(59,130,246,.6); }

.result-card{
  border-radius:18px; padding:24px; margin-top:25px;
  font-size:22px; font-weight:800; text-align:center;
  animation:fadeInUp 0.9s ease-in-out;
}
.success{ background: linear-gradient(135deg, #10b981, #34d399); color:#ffffff !important; box-shadow:0 0 18px rgba(16,185,129,.35); }
.danger { background: linear-gradient(135deg, #f43f5e, #ef4444); color:#ffffff !important; box-shadow:0 0 18px rgba(239,68,68,.35); }
@keyframes fadeInUp{ from{opacity:0; transform:translateY(25px)} to{opacity:1; transform:translateY(0)} }

div[role="alert"]{
  background: rgba(148,163,184,0.12) !important;
  color:#F8FAFC !important;
  border:1px solid #7b8aa0 !important;
}
</style>
""", unsafe_allow_html=True)

# ===== تحميل الموديل والـ vectorizer =====
vectorizer, model = None, None

try:
    # جرّب إذا الملف فيه dict
    with open("spam_classifier.pkl", "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "vectorizer" in data and "model" in data:
        vectorizer, model = data["vectorizer"], data["model"]
    else:
        # إذا ما كان dict → حمّل كل واحد لحاله
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        model = data  # الملف spam_classifier.pkl فيه الموديل فقط
except Exception as e:
    st.error(f"⚠️ خطأ في تحميل الملفات: {e}")

# ===== الواجهة =====
st.markdown("<h1>🚀 مصنّف الإيميلات (احتيالي / عادي)</h1>", unsafe_allow_html=True)

email_text = st.text_area("✍️ أدخل نص الإيميل:", height=180, placeholder="اكتب الإيميل المراد تحليله...")

if st.button("🔮 تصنيف الإيميل"):
    if not email_text.strip():
        st.warning("⚠️ رجاءً اكتب نص الإيميل أولاً")
    elif vectorizer is None or model is None:
        st.error("❌ لم يتم تحميل الموديل أو الـ vectorizer بشكل صحيح")
    else:
        X = vectorizer.transform([email_text])

        # حساب الاحتمالية
        if hasattr(model, "predict_proba"):
            prob_phishing = model.predict_proba(X)[0][1]
        else:
            score = model.decision_function(X)
            prob_phishing = 1 / (1 + np.exp(-score[0]))

        pred = model.predict(X)[0]  # 1 = احتيالي، 0 = عادي

        with st.spinner("⏳ جاري تحليل الإيميل..."):
            time.sleep(1.0)

        # عرض النتيجة
        if pred == 1:
            st.markdown(
                f"<div class='result-card danger'>🚨 احتيالي<br>نسبة الثقة: {prob_phishing*100:.1f}%</div>",
                unsafe_allow_html=True
            )
            st.info("📌 السبب: الموديل اكتشف كلمات أو تراكيب مشبوهة "
                    "مثل الروابط الغريبة أو العبارات التسويقية القوية.")
        else:
            conf_legit = (1 - prob_phishing) * 100
            st.markdown(
                f"<div class='result-card success'>✅ عادي<br>نسبة الثقة: {conf_legit:.1f}%</div>",
                unsafe_allow_html=True
            )
            st.info("📌 السبب: النص لا يحتوي على مؤشرات قوية للاحتيال.")

# ===== التذييل =====
st.markdown(
    """
    <hr style="margin-top:30px; margin-bottom:10px; border: 1px solid #334155;" />
    <div style="text-align:center; font-size:14px; color:#94A3B8;">
        👩‍💻 تم التطوير بواسطة: <b>هدى، ليان، ريماس، لينا</b>
    </div>
    """,
    unsafe_allow_html=True
)
