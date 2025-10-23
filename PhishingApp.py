# phishing_app.py
import streamlit as st
import joblib
import re
from bs4 import BeautifulSoup
import tldextract
import nltk
# ---- NLTK data: ensure tokenizer & stopwords are available on the server ----
import os
import nltk

# Put NLTK data inside project folder so Streamlit can access it reliably
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

# Make sure nltk knows to look here
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

# Download the common tokenizers + stopwords (quietly). punkt_tab sometimes requested on servers;
# we'll try both punkt and punkt_tab to be safe.
nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)

# Some environments request punkt_tab — try downloading it too (if it exists in your NLTK version)
try:
    nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)
except Exception:
    # fallback: punt if punkt_tab is not available in this NLTK distribution
    pass
# -------------------------------------------------------------------------------

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from scipy.sparse import hstack, csr_matrix

# Download NLTK data quietly (first run only)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Load model artifacts
ARTIFACT_PATH = "model_artifacts/phish_detector_v1.joblib"
art = joblib.load(ARTIFACT_PATH)
model = art['model']
vectorizer = art['vectorizer']
scaler = art['scaler']

# Text cleaning & feature prep (same as training)
def clean_text_for_app(text):
    text = BeautifulSoup(str(text), "html.parser").get_text()
    links = re.findall(r'http[s]?://\S+|www\.\S+', text)
    text_no_url = re.sub(r'http\S+|www\.\S+', ' ', text)
    text_alpha = re.sub(r'[^A-Za-z\s]', ' ', text_no_url)
    tokens = [w.lower() for w in word_tokenize(text_alpha) if w.isalpha() and w.lower() not in stop_words]
    return " ".join(tokens), len(links)

def suspicious_count(text):
    suspicious_words = [
        'verify','account','password','limited','urgent','claim','winner','update',
        'confirm','suspend','billing','payment','secure','reset','otp','win','prize','selected'
    ]
    t = text.lower()
    return sum(1 for w in suspicious_words if w in t)

def prepare_features(text, sender=""):
    clean, num_links = clean_text_for_app(text)
    X_text = vectorizer.transform([clean])
    sender_domain = tldextract.extract(sender).domain if sender else ""
    sender_known_provider = 1 if sender_domain in ('gmail','yahoo','hotmail','outlook','icloud') else 0
    X_num = np.array([[num_links, suspicious_count(text), sender_known_provider]])
    X_num_scaled = scaler.transform(X_num)
    return hstack([X_text, csr_matrix(X_num_scaled)])

# UI
st.set_page_config(page_title="Phishing Detector", page_icon="🔍", layout="centered")
st.title("🔍 Phishing Email Detector by, TECHTITANS")
st.markdown("Paste the email subject and body below and press **Check**.")

subject = st.text_input("Email subject (optional)")
body = st.text_area("Email body/text", height=220)
sender = st.text_input("Sender email (optional)")


if st.button("Check"):
    if not (body.strip() or subject.strip()):
        st.warning("Please enter an email subject or body.")
    else:
        text_to_check = (subject + " " + body).strip()
        X = prepare_features(text_to_check, sender=sender)
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None
        if pred == 1:
            st.error(f"🚨 PHISHING (prob {prob*100:.2f}%)" if prob is not None else "🚨 PHISHING")
        else:
            st.success(f"✅ SAFE (phishing prob {prob*100:.2f}%)" if prob is not None else "✅ SAFE")

        # show brief features
        st.write("**Detected features:**")
        st.write(f"- Links found: {re.findall(r'http[s]?://\\S+|www\\.\\S+', text_to_check)}")
        st.write(f"- Suspicious keyword count: {suspicious_count(text_to_check)}")
