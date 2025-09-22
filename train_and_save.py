import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- إعداد البيانات ---
CSV_PATH = "ArabicPhishingEmails_clean.csv"
TEXT_COL = "Final Text Email"
LABEL_COL = "email type"

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

# تنظيف الليبل
def to_bin(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"1","spam","true","yes"}: return 1
    if s in {"0","ham","not spam","false","no"}: return 0
    try:
        return int(float(s))
    except:
        return None

df[LABEL_COL] = df[LABEL_COL].apply(to_bin)
df = df.dropna(subset=[LABEL_COL])
df[LABEL_COL] = df[LABEL_COL].astype(int)

# تنظيف النص
df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
df = df[df[TEXT_COL].str.len() > 0]

# X و y
X = df[TEXT_COL].values
y = df[LABEL_COL].values

# --- تقسيم ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- TF-IDF + Logistic Regression ---
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_vec, y_train)

# --- حفظ الموديل ---
with open("spam_classifier.pkl", "wb") as f:
    pickle.dump({"vectorizer": vectorizer, "model": model}, f)

print("✅ تم إنشاء الملف spam_classifier.pkl بنجاح")
