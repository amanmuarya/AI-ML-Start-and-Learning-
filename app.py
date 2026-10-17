from pathlib import Path
import html
import re

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "spam.csv"

st.set_page_config(
    page_title="MailGuard | Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Visual design ----------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
      :root { --ink:#172033; --muted:#68738a; --line:#e4e9f2; --blue:#3659e3; }
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .stApp { background: linear-gradient(180deg,#f7f9ff 0%,#f7f8fb 48%,#f3f6fb 100%); color:var(--ink); }
      [data-testid="stHeader"] { background:rgba(247,249,255,.82); }
      .block-container { max-width: 1420px; padding-top: 1.7rem; padding-bottom: 3rem; }
      h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; color:#172033; }
      section[data-testid="stSidebar"] { background:#111a2e; }
      section[data-testid="stSidebar"] * { color:#f4f6ff; }
      section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#c5cee2; }
      .brand-row { display:flex; align-items:center; gap:12px; margin-bottom:14px; }
      .brand-icon { width:46px; height:46px; border-radius:15px; display:flex; align-items:center; justify-content:center; background:linear-gradient(135deg,#4565f5,#7c55e8); color:white; font-size:24px; box-shadow:0 8px 22px rgba(65,89,230,.25); }
      .brand-name { font-family:'Manrope',sans-serif; font-weight:800; font-size:1.55rem; color:#172033; letter-spacing:-.05em; }
      .brand-name span { color:#4e62df; }
      .eyebrow { font-size:.72rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; color:#596cda; margin-bottom:8px; }
      .hero { border-radius:24px; padding:30px 32px; background:radial-gradient(circle at 88% 8%,rgba(123,98,255,.42),transparent 26%),linear-gradient(125deg,#16223d,#273c72 70%,#3d4bb0); color:white; box-shadow:0 18px 45px rgba(31,50,100,.15); margin-bottom:24px; }
      .hero h1 { color:white !important; font-size:2.25rem !important; line-height:1.12; margin:0 0 10px 0; }
      .hero p { color:#dce4ff; max-width:760px; font-size:1.02rem; margin:0; }
      .hero-pill { display:inline-block; border:1px solid rgba(255,255,255,.25); background:rgba(255,255,255,.1); padding:6px 10px; border-radius:999px; color:#f7f8ff; font-size:.76rem; margin-bottom:16px; }
      .section-kicker { color:#69758d; font-size:.78rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; margin:0 0 8px 0; }
      .panel-heading { font-family:'Manrope',sans-serif; font-weight:800; font-size:1.25rem; color:#172033; margin-bottom:4px; }
      .muted { color:#718096; font-size:.9rem; }
      .mini-card { background:white; border:1px solid #e5eaf3; border-radius:17px; padding:17px 18px; height:100%; box-shadow:0 5px 18px rgba(23,37,70,.035); }
      .mini-label { color:#778198; font-size:.76rem; font-weight:700; letter-spacing:.04em; text-transform:uppercase; }
      .mini-value { font-family:'Manrope',sans-serif; color:#19243d; font-size:1.5rem; font-weight:800; margin-top:4px; }
      .status-box { padding:20px 22px; border-radius:18px; border:1px solid #e5eaf3; background:white; box-shadow:0 8px 22px rgba(23,37,70,.035); }
      .status-spam { border-color:#ffcaca; background:linear-gradient(135deg,#fff7f6,#fff); }
      .status-safe { border-color:#bce8d2; background:linear-gradient(135deg,#f2fff8,#fff); }
      .status-title { font-family:'Manrope',sans-serif; font-size:1.45rem; font-weight:800; margin-bottom:4px; }
      .signal { padding:11px 13px; border-radius:12px; background:#fff; border:1px solid #e6ebf4; margin:7px 0; color:#344054; }
      .signal-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:9px; background:#ed9c42; }
      .signal-good .signal-dot { background:#2aab77; }
      div[data-testid="stForm"] { background:#fff; border:1px solid #e4e9f2; border-radius:20px; padding:22px 22px 12px 22px; box-shadow:0 8px 28px rgba(23,37,70,.04); }
      div.stButton > button, div[data-testid="stFormSubmitButton"] button { border-radius:12px; min-height:2.8rem; font-weight:700; transition:all .15s ease; }
      div[data-testid="stFormSubmitButton"] button { background:linear-gradient(110deg,#405ce8,#7556e8); color:white; border:0; }
      div[data-testid="stFormSubmitButton"] button:hover { background:linear-gradient(110deg,#314bce,#6645d2); color:white; border:0; }
      div.stButton > button { border:1px solid #dfe5f0; background:white; color:#2f3b57; }
      div.stButton > button:hover { border-color:#aab8ff; color:#3f55cf; }
      div[data-baseweb="input"] > div, div[data-baseweb="textarea"] > div, div[data-baseweb="select"] > div { border-radius:11px; }
      .fine-print { color:#778198; font-size:.78rem; line-height:1.5; }
      .history-row { background:#fff; border:1px solid #e5eaf3; padding:12px 14px; border-radius:12px; margin-bottom:8px; }
      @media(max-width:760px) { .hero{padding:22px;} .hero h1{font-size:1.8rem!important;} .block-container{padding-left:1rem;padding-right:1rem;} }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Dataset se AI model train ho raha hai…")
def load_model():
    """Train a simple TF-IDF + Naive Bayes model from the project's spam.csv."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"spam.csv nahi mili. Is file ko app.py ke same folder mein rakho: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH, encoding="latin-1")
    if not {"v1", "v2"}.issubset(data.columns):
        raise ValueError("spam.csv mein v1 (label) aur v2 (message) columns hone chahiye.")

    data = data[["v1", "v2"]].dropna().copy()
    data["v1"] = data["v1"].astype(str).str.strip().str.lower()
    data["v2"] = data["v2"].astype(str).str.strip()
    data = data[data["v1"].isin(["ham", "spam"]) & data["v2"].ne("")]
    if data.empty or data["v1"].nunique() < 2:
        raise ValueError("Dataset mein ham aur spam dono tarah ke messages hone chahiye.")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True, max_features=60000)),
        ("classifier", MultinomialNB(alpha=0.35)),
    ])
    model.fit(data["v2"], data["v1"])
    return model, len(data), int((data["v1"] == "spam").sum()), int((data["v1"] == "ham").sum())


def find_signals(text, sender=""):
    """Simple visible text checks; these are hints, not the ML model's explanation."""
    checks = [
        ("Prize / reward wording", r"\b(won|winner|prize|lottery|reward|claim your|free gift|cash bonus)\b"),
        ("Urgent or pressure wording", r"\b(urgent|act now|limited time|immediately|expires today|last chance)\b"),
        ("Sensitive information request", r"\b(password|otp|one[- ]time password|bank details|card number|verify your account|login to verify)\b"),
        ("Payment or money request", r"\b(wire transfer|send money|processing fee|pay now|payment required|refund)\b"),
    ]
    results = []
    for label, pattern in checks:
        if re.search(pattern, text, flags=re.IGNORECASE):
            results.append(label)

    urls = re.findall(r"https?://\S+|www\.\S+", text, flags=re.IGNORECASE)
    if urls:
        results.append(f"Contains link ({len(urls)} detected)")
    if re.search(r"\b(bit\.ly|tinyurl\.com|t\.co|goo\.gl|cutt\.ly)\b", text, flags=re.IGNORECASE):
        results.append("Contains a shortened link")

    letters = [char for char in text if char.isalpha()]
    if len(letters) >= 25 and sum(char.isupper() for char in letters) / max(len(letters), 1) > 0.55:
        results.append("Unusually high use of capital letters")

    if sender.strip() and not re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", sender.strip()):
        results.append("Sender field does not look like a complete email address")
    return results


def safe_text(value, fallback="Not provided"):
    value = (value or "").strip()
    return html.escape(value) if value else html.escape(fallback)


try:
    model, total_rows, spam_rows, ham_rows = load_model()
except Exception as exc:
    st.error(f"Model load nahi ho saka: {exc}")
    st.info("Check karo ki app.py aur spam.csv same folder mein hain aur dependencies install hain.")
    st.stop()

# Session state defaults
for key, default in {
    "message_mode": "Email",
    "sender_input": "",
    "subject_input": "",
    "body_input": "",
    "scan_history": [],
    "last_result": None,
}.items():
    st.session_state.setdefault(key, default)

# Sidebar
with st.sidebar:
    st.markdown("<div style='font-size:.72rem;letter-spacing:.16em;font-weight:800;color:#97a8ff'>MAILGUARD / AI LAB</div>", unsafe_allow_html=True)
    st.markdown("# 🛡️ MailGuard")
    st.markdown("Smart message screening")
    st.divider()
    st.markdown("### Model overview")
    st.markdown(f"**{total_rows:,}** training messages")
    st.markdown(f"🟢 Normal / ham: **{ham_rows:,}**")
    st.markdown(f"🔴 Spam: **{spam_rows:,}**")
    st.divider()
    st.markdown("### How to use")
    st.markdown("1. Add the email subject and body.\n2. Optionally add the sender address.\n3. Select **Analyze message**.\n4. Review the result and warning signals.")
    st.divider()
    st.markdown("### Model note")
    st.markdown("The included `spam.csv` is an SMS-style spam dataset. Email results are **indicative**, not a production email-security verdict.")
    st.markdown("<div class='fine-print'>The sender address is checked only for basic formatting; the ML score uses the subject and body text.</div>", unsafe_allow_html=True)

# Header
st.markdown(
    """
    <div class="brand-row"><div class="brand-icon">✉</div><div class="brand-name">Mail<span>Guard</span></div></div>
    <div class="hero">
      <div class="hero-pill">✦ AI-POWERED MESSAGE CHECK</div>
      <h1>Know what’s hiding<br>in your inbox.</h1>
      <p>Check a suspicious email or text before you click, reply, or share personal information. Get a clear prediction and easy-to-understand warning signals.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick examples are outside the form so that clicking them populates the form fields.
st.markdown('<div class="section-kicker">Start with a sample</div>', unsafe_allow_html=True)
sample_a, sample_b, sample_c = st.columns([1, 1, 2])
with sample_a:
    if st.button("🚨 Load spam example", use_container_width=True):
        st.session_state["message_mode"] = "Email"
        st.session_state["sender_input"] = "rewards@prize-notice.example"
        st.session_state["subject_input"] = "Congratulations — claim your prize!"
        st.session_state["body_input"] = "You are a winner! Claim your free cash reward now. Click https://bit.ly/example to claim your prize. Limited time offer."
        st.session_state["last_result"] = None
with sample_b:
    if st.button("✅ Load normal example", use_container_width=True):
        st.session_state["message_mode"] = "Email"
        st.session_state["sender_input"] = "teammate@example.com"
        st.session_state["subject_input"] = "Meeting notes for tomorrow"
        st.session_state["body_input"] = "Hi, I shared the meeting notes in the folder. Please review them before tomorrow's discussion. Let me know if you have questions."
        st.session_state["last_result"] = None
with sample_c:
    st.markdown("<div style='padding:8px 4px;color:#778198;font-size:.86rem'>Tip: Paste the full subject and body for a more useful check.</div>", unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
left_col, right_col = st.columns([1.3, 0.7], gap="large")

with left_col:
    st.markdown('<div class="section-kicker">Message details</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-heading">What would you like to check?</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Your text stays in this local app session and is used for the prediction.</div>', unsafe_allow_html=True)

    with st.form("message_scan_form", clear_on_submit=False, border=False):
        mode = st.selectbox("Message type", ["Email", "SMS / text message"], key="message_mode")
        sender = ""
        subject = ""
        if mode == "Email":
            sender = st.text_input("From / Sender email (optional)", key="sender_input", placeholder="name@example.com")
            subject = st.text_input("Email subject", key="subject_input", placeholder="e.g. Your account needs attention")
        body_label = "Email body" if mode == "Email" else "Text message"
        body = st.text_area(
            body_label,
            key="body_input",
            height=210,
            placeholder="Paste the message content here…",
            help="Do not paste passwords, OTPs, or other private information into a demo tool.",
        )
        submitted = st.form_submit_button("🔍  Analyze message", use_container_width=True)

    if submitted:
        combined = " ".join(part.strip() for part in [subject, body] if part and part.strip())
        if not combined:
            st.warning("Pehle subject ya message body likho, phir Analyze message dabao.")
        else:
            predicted = str(model.predict([combined])[0]).lower()
            probabilities = model.predict_proba([combined])[0]
            spam_idx = list(model.classes_).index("spam")
            spam_probability = float(probabilities[spam_idx])
            signals = find_signals(combined, sender if mode == "Email" else "")
            result = {
                "mode": mode,
                "sender": sender.strip(),
                "subject": subject.strip(),
                "body": body.strip(),
                "prediction": predicted,
                "spam_probability": spam_probability,
                "signals": signals,
            }
            st.session_state["last_result"] = result
            st.session_state["scan_history"].insert(0, {
                "prediction": predicted,
                "spam_probability": spam_probability,
                "subject": subject.strip() if mode == "Email" else "Text message",
                "preview": combined[:105].replace("\n", " "),
            })
            st.session_state["scan_history"] = st.session_state["scan_history"][:5]

with right_col:
    st.markdown('<div class="section-kicker">Before you trust it</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-heading">Quick safety checklist</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="mini-card" style="margin-bottom:12px">
          <div style="font-size:1.4rem;margin-bottom:8px">🔗</div>
          <b>Check links first</b><br><span class="muted">A link can look familiar while leading somewhere else.</span>
        </div>
        <div class="mini-card" style="margin-bottom:12px">
          <div style="font-size:1.4rem;margin-bottom:8px">🔐</div>
          <b>Keep private details private</b><br><span class="muted">Never send your password, OTP, or bank PIN in reply to a message.</span>
        </div>
        <div class="mini-card" style="margin-bottom:12px">
          <div style="font-size:1.4rem;margin-bottom:8px">⏳</div>
          <b>Pause under pressure</b><br><span class="muted">Urgency, threats, and unexpected prizes deserve a closer look.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Prediction result section
result = st.session_state.get("last_result")
if result:
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">Analysis report</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-heading">Your message scan</div>', unsafe_allow_html=True)
    spam_pct = result["spam_probability"] * 100
    spam_flag = result["prediction"] == "spam"

    r1, r2 = st.columns([1.1, 0.9], gap="large")
    with r1:
        if spam_flag:
            st.markdown(
                f"<div class='status-box status-spam'><div style='font-size:.75rem;font-weight:800;letter-spacing:.12em;color:#bb3434'>RESULT · REVIEW CAREFULLY</div><div class='status-title' style='color:#b42318'>🚨 Likely SPAM</div><div style='color:#6e4a4a'>The text model found patterns commonly associated with spam. Avoid clicking links or sharing information until verified.</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='status-box status-safe'><div style='font-size:.75rem;font-weight:800;letter-spacing:.12em;color:#18835b'>RESULT · LOWER SPAM SIGNAL</div><div class='status-title' style='color:#137653'>✅ Likely NOT SPAM</div><div style='color:#496b5c'>The model classified this as a normal message. This is not a guarantee that the sender or links are safe.</div></div>",
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown("**Message preview**")
        if result["mode"] == "Email":
            st.markdown(f"<div class='mini-card'><div class='mini-label'>From</div><div style='margin:3px 0 10px'>{safe_text(result['sender'])}</div><div class='mini-label'>Subject</div><div style='margin:3px 0 10px;font-weight:700'>{safe_text(result['subject'])}</div><div class='mini-label'>Body</div><div style='white-space:pre-wrap;overflow-wrap:anywhere'>{safe_text(result['body'])}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='mini-card' style='white-space:pre-wrap;overflow-wrap:anywhere'>{safe_text(result['body'])}</div>", unsafe_allow_html=True)

    with r2:
        st.markdown("<div class='mini-card'>", unsafe_allow_html=True)
        st.markdown("<div class='mini-label'>Model spam probability estimate</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='mini-value' style='font-size:2.25rem'>{spam_pct:.1f}<span style='font-size:1.1rem'>%</span></div>", unsafe_allow_html=True)
        st.progress(int(round(spam_pct)))
        if spam_pct >= 70:
            st.markdown("<div style='color:#bd3434;font-weight:700;font-size:.88rem'>High spam signal</div>", unsafe_allow_html=True)
        elif spam_pct >= 35:
            st.markdown("<div style='color:#ad7518;font-weight:700;font-size:.88rem'>Medium spam signal</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#23845e;font-weight:700;font-size:.88rem'>Low spam signal</div>", unsafe_allow_html=True)
        st.markdown("<hr style='border:none;border-top:1px solid #e8edf5;margin:18px 0'>", unsafe_allow_html=True)
        st.markdown("<div class='mini-label'>Visible warning signals</div>", unsafe_allow_html=True)
        if result["signals"]:
            for signal in result["signals"]:
                st.markdown(f"<div class='signal'><span class='signal-dot'></span>{signal}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='signal signal-good'><span class='signal-dot'></span>No common warning phrase detected by the basic checks.</div>", unsafe_allow_html=True)
        st.markdown("<div class='fine-print' style='margin-top:13px'>Warning signals are simple text checks, separate from the ML model. They may miss risks or flag harmless messages.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='fine-print' style='margin-top:14px'>Important: The score is a model estimate, not a guarantee. This project uses an SMS-style dataset and is for learning/demo purposes; verify unexpected messages through the sender's official website or app.</div>", unsafe_allow_html=True)

# Recent scan history
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
history_col, clear_col = st.columns([4, 1])
with history_col:
    st.markdown('<div class="section-kicker">This session</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-heading">Recent checks</div>', unsafe_allow_html=True)
with clear_col:
    if st.session_state["scan_history"] and st.button("Clear history", use_container_width=True):
        st.session_state["scan_history"] = []
        st.rerun()

history = st.session_state.get("scan_history", [])
if history:
    for item in history:
        label = "SPAM" if item["prediction"] == "spam" else "NOT SPAM"
        color = "#b42318" if item["prediction"] == "spam" else "#137653"
        title = html.escape(item["subject"] or item["preview"])
        preview = html.escape(item["preview"])
        st.markdown(
            f"<div class='history-row'><div style='display:flex;justify-content:space-between;gap:14px;align-items:center'><div style='min-width:0'><div style='font-weight:700;color:#26324a;white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{title}</div><div style='font-size:.82rem;color:#7b8498;white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{preview}</div></div><div style='flex-shrink:0;text-align:right'><div style='color:{color};font-size:.73rem;font-weight:800'>{label}</div><div style='font-weight:800;color:#26324a'>{item['spam_probability']*100:.1f}%</div></div></div></div>",
            unsafe_allow_html=True,
        )
else:
    st.markdown("<div class='mini-card'><span class='muted'>Your analyzed messages will appear here during this session.</span></div>", unsafe_allow_html=True)

st.markdown("<div style='height:24px'></div><div class='fine-print' style='text-align:center'>MAILGUARD · Educational Spam Detection Project · Keep your data safe</div>", unsafe_allow_html=True)
