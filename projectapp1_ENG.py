# app_spam_en.py
import streamlit as st
import numpy as np
import re
import joblib

# ========== PAGE SETUP ==========
st.set_page_config(page_title="📧 Email Classifier", page_icon="📧", layout="centered")

# ===== CSS Styling =====
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
hr { border: 1px solid #334155; }
</style>
""", unsafe_allow_html=True)

# ========== LOAD MODEL & VECTORIZER ==========
model = joblib.load("phishing_model_layan.pkl")
vectorizer = joblib.load("tfidf_vectorizer_layan.pkl")

# ========== TEXT UTILITIES ==========
# (اختياري) Stemming – يعمل إن وُجدت NLTK، وإلا يتجاهله
try:
    from nltk.stem import PorterStemmer
    _ps = PorterStemmer()
    def _stem(w): return _ps.stem(w)
except Exception:
    def _stem(w): return w

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def preprocess(text: str) -> str:
    """تنظيف بسيط مشابه لـ clean_text في التدريب."""
    t = text.lower()
    t = re.sub(r'(http\S+|www\.\S+|\S+@\S+)', ' ', t)  # شيل روابط/إيميلات
    t = re.sub(r'[^a-z\s]', ' ', t)                    # حروف إنجليزية فقط
    tokens = [_stem(w) for w in t.split() if w and w not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)

def segment_text(text: str):
    """تقسيم الإيميل لأسطر/جُمل قصيرة."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    segs = []
    for ln in lines:
        parts = re.split(r'(?<=[.!?])\s+|[.!?]{2,}', ln)
        for p in parts:
            p = p.strip()
            if p:
                segs.append(p)
    return segs or [text.strip()]

SUSPICIOUS_REGEXES = [
    re.compile(r'\b(bank|iban|account|verify|verification|password|otp|login|update|confirm|security)\b', re.I),
    re.compile(r'\b(click|tap|open)\s+(here|link)\b', re.I),
    re.compile(r'\b(win|winner|prize|jackpot|lottery|reward|gift)\b', re.I),
    re.compile(r'\burgent|immediately|act now|limited time|final notice\b', re.I),
    re.compile(r'http[s]?://', re.I),
    re.compile(r'\$\s?\d[\d,]*', re.I),
    re.compile(r'\bwire|transfer\b', re.I),
    re.compile(r'\bbitcoin|crypto\b', re.I),
    re.compile(r'\bssn|social\s+security\b', re.I),
    re.compile(r'!{3,}')
]
def heuristic_hits(text: str):
    hits = []
    for rx in SUSPICIOUS_REGEXES:
        if rx.search(text):
            hits.append(rx.pattern)
    letters = sum(c.isalpha() for c in text)
    caps = sum(c.isupper() for c in text)
    if letters >= 12 and caps / max(letters, 1) > 0.6:
        hits.append("HIGH_CAPS")
    return hits

def topk_mean(probs, k=2):
    probs = np.asarray(probs)
    k = int(np.clip(k, 1, len(probs)))
    return float(np.mean(np.sort(probs)[-k:]))

# ثوابت داخلية (بدون عناصر تحكّم)
BASE_THRESHOLD = 0.25   # قرار سبام
TOPK = 2                # متوسط أعلى مقطعين
HIGH_ALERT = 0.85       # إنذار فوري لو أي مقطع ≥ هذا
RULE_MIN_HITS = 2       # لو ≥ إشارتين من القواعد → Spam

def predict_email(text: str):
    """يُرجع: (is_spam, p_max, p_topk, most_suspicious_line, flags)"""
    segments = segment_text(text)

    pairs = []
    for seg in segments:
        clean = preprocess(seg)
        if len(clean.split()) >= 3:   # تجاهل المقاطع الصغيرة جدًا
            pairs.append((seg, clean))

    if not pairs:
        return False, 0.0, 0.0, "", []

    # احتمالات لكل مقطع عبر الموديل
    X = vectorizer.transform([c for (_orig, c) in pairs])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        score = model.decision_function(X)
        probs = 1 / (1 + np.exp(-score))

    p_max = float(np.max(probs))
    p_topk = topk_mean(probs, k=TOPK)
    worst_idx = int(np.argmax(probs))
    suspicious_line = pairs[worst_idx][0]

    # قواعد بسيطة على النص الخام
    flags = set()
    for seg in segments:
        flags.update(heuristic_hits(seg))
    flags.update(heuristic_hits(text))

    is_spam = (p_topk >= BASE_THRESHOLD) or \
              (p_max >= max(BASE_THRESHOLD, HIGH_ALERT)) or \
              (len(flags) >= RULE_MIN_HITS)

    return is_spam, p_max, p_topk, suspicious_line, sorted(flags)

# ========== UI ==========
st.markdown("<h1>🚀 Email Classifier (Spam / Not Spam)</h1>", unsafe_allow_html=True)

email_text = st.text_area(
    "✍️ Enter the email text here:",
    height=180,
    placeholder="Write or paste the email to classify..."
)

if st.button("🔮 Classify Email"):
    if not email_text.strip():
        st.warning("⚠️ Please enter the email text first")
    else:
        with st.spinner("⏳ Analyzing email..."):
            is_spam, p_max, p_topk, suspicious_line, flags = predict_email(email_text)

        conf_spam = p_topk * 100
        conf_not = (1 - p_topk) * 100

        if is_spam:
            st.markdown(
                f"<div class='result-card danger'>🚨 Spam<br>Confidence: {conf_spam:.1f}%</div>",
                unsafe_allow_html=True
            )
            if suspicious_line:
                st.write("Most suspicious line:")
                st.code(suspicious_line[:300])
            if flags:
                st.info("Heuristic flags: " + ", ".join(flags))
        else:
            st.markdown(
                f"<div class='result-card success'>✅ Not Spam<br>Confidence: {conf_not:.1f}%</div>",
                unsafe_allow_html=True
            )

        st.caption(f"Details • Max segment: {p_max:.2f}  |  Top-2 mean: {p_topk:.2f}  |  Threshold: {BASE_THRESHOLD:.2f}")

# ===== Footer =====
st.markdown(
    """
    <hr style="margin-top:30px; margin-bottom:10px;" />
    <div style="text-align:center; font-size:14px; color:#94A3B8;">
        👩‍💻 Developed by: <b>Huda, Layan, Rimas, Leena</b>
    </div>
    """,
    unsafe_allow_html=True
)
