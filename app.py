import streamlit as st

st.set_page_config(page_title="SecureBank AI | Fraud Risk Screening", page_icon="🛡️", layout="centered")

HIGH_RISK_PHRASES = [
    "otp", "one time password", "share your pin", "share pin", "password",
    "urgent transfer", "verify your account", "account blocked", "click this link",
    "claim your prize", "gift card", "crypto wallet", "unknown beneficiary",
    "unrecognized transaction", "unauthorized transaction", "remote access",
    "screen sharing", "kyc update link",
]
MEDIUM_RISK_PHRASES = [
    "new beneficiary", "international transfer", "unusual payment",
    "unknown merchant", "online purchase", "cash withdrawal", "wire transfer",
    "late night", "first transaction",
]

def screen_message(amount: float, message: str):
    text = message.lower()
    high_hits = [p for p in HIGH_RISK_PHRASES if p in text]
    medium_hits = [p for p in MEDIUM_RISK_PHRASES if p in text]
    score, reasons = 0, []

    if high_hits:
        score += min(60, 25 * len(high_hits))
        reasons.append("Potentially suspicious wording: " + ", ".join(high_hits[:4]))
    if medium_hits:
        score += min(30, 12 * len(medium_hits))
        reasons.append("Transaction patterns to review: " + ", ".join(medium_hits[:4]))

    if amount >= 100000:
        score += 25
        reasons.append("High amount (₹1,00,000 or more); verify against your usual activity.")
    elif amount >= 50000:
        score += 15
        reasons.append("Relatively high amount (₹50,000 or more); consider confirming it.")
    elif amount >= 20000:
        score += 7
        reasons.append("Amount is ₹20,000 or more; amount alone does not establish fraud.")

    score = min(score, 100)
    if score >= 50:
        label, advice = "Higher risk indicators", "Do not click links or share OTP/PIN/password. Verify through your bank's official app or phone number."
    elif score >= 20:
        label, advice = "Review recommended", "Check the recipient and transaction in your official banking app."
    else:
        label, advice = "Few risk indicators detected", "This does not prove the transaction is safe. Confirm it through your bank's official channel."
    if not reasons:
        reasons.append("No configured warning phrases or amount thresholds were detected.")
    return label, score, reasons, advice

st.markdown("<div style='text-align:center'><div style='font-size:2.5rem'>🛡️</div><h1>SecureBank AI</h1><p>Transaction Message Risk Screening</p></div>", unsafe_allow_html=True)
st.info("Demo project: enter the amount and a bank-style message. This screen uses transparent demo rules, not a trained message-based fraud model.")

with st.form("transaction_form"):
    amount = st.number_input("Transaction amount (₹)", min_value=0.0, value=45000.0, step=500.0, format="%.2f")
    message = st.text_area("Transaction reason / bank message", value="Transferred to a new beneficiary for an online purchase", placeholder="Example: ₹20,000 paid to an unknown merchant...", height=120, max_chars=2000)
    submitted = st.form_submit_button("🔎 Analyze Transaction", use_container_width=True)

if submitted:
    if not message.strip():
        st.warning("Please enter the transaction message.")
    else:
        label, score, reasons, advice = screen_message(float(amount), message)
        st.divider()
        st.subheader("Screening result")
        if score >= 50:
            st.error("⚠️ " + label)
        elif score >= 20:
            st.warning("🟠 " + label)
        else:
            st.success("🟢 " + label)
        st.metric("Demo risk indicator", f"{score}/100")
        st.caption("This score is a rule-based demo indicator, not a calibrated probability of fraud.")
        st.markdown("**Why this result?**")
        for reason in reasons:
            st.write("- " + reason)
        st.info(advice)

st.divider()
st.caption("Never enter real card numbers, CVV, PIN, passwords, or OTPs. A message and amount alone cannot reliably prove fraud.")
with st.expander("Where is the trained ML model?"):
    st.write("train_model.py separately trains Logistic Regression and Random Forest using creditcard.csv. That dataset contains Time, Amount, V1–V28 and Class, but no message text. Its numerical model cannot honestly predict from only an amount and message. A genuine combined model requires labeled transactions with descriptions/messages.")
