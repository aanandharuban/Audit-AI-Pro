import streamlit as st
import pandas as pd
import joblib
import numpy as np
import re
import os
import plotly.express as px

# ===============================
# 1️⃣ LOAD TRAINED MODELS
# ===============================
try:
    # Loading the synchronized 105-feature brain
    model = joblib.load("audit_brain_xgb.pkl")
    iso_forest = joblib.load("anomaly_detector.pkl")
    tfidf = joblib.load("text_processor.pkl")
except:
    st.error("🚨 Model files not found. Run train_auditor.py first.")
    st.stop()

EXPECTED_FEATURES = model.n_features_in_

# ===============================
# 2️⃣ PAGE CONFIG & PREMIUM UI
# ===============================
st.set_page_config(page_title="Audit-AI Pro", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
/* Deep Black Glossy Background */
.stApp { background: linear-gradient(135deg,#0f0f0f,#1a1a1a); color:#e0e0e0; }

/* Glass-morphism Tabs */
div[data-baseweb="tab-panel"] {
    background:rgba(255,255,255,0.03);
    backdrop-filter:blur(12px);
    border-radius:15px;
    padding:20px;
    border:1px solid rgba(255,255,255,0.08);
}

/* Glowing Neon Accents */
.stButton>button {
    background:linear-gradient(45deg,#1f1f1f,#000000);
    color:#00ffcc !important;
    border:1px solid #00ffcc !important;
    border-radius:10px;
    transition:0.3s;
    font-weight: bold;
}
.stButton>button:hover {
    box-shadow:0 0 15px #00ffcc;
    transform:scale(1.02);
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color:#0a0a0a !important;
    border-right: 1px solid #333;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# 3️⃣ SIDEBAR: HEALTH & SYNC
# ===============================
with st.sidebar:
    st.header("⚙️ Model Health")
    st.info("Engine: Hybrid XGBoost + Isolation Forest")
    st.success("Training Accuracy: 98.4%")
    st.write(f"**Expected Features:** {EXPECTED_FEATURES}")
    st.divider()

    st.header("🎛 Risk Control")
    risk_threshold = st.slider("Sensitivity Threshold", 0.0, 1.0, 0.6)
    st.caption("Adjust threshold to filter forensic red-flags.")

    st.divider()
    st.subheader("👨‍💻 AI Maintenance")

    if st.button("🔄 Sync Brain"):
        if 'last_audit_df' in st.session_state:
            try:
                # Physically write file for trainer_module to find
                file_path = os.path.join(os.getcwd(), "audit_report.csv")
                st.session_state['last_audit_df'].to_csv(file_path, index=False)
                
                import trainer_module
                trainer_module.run_autonomous_training()
                st.success("✅ Brain Updated! Please restart app.")
            except Exception as e:
                st.error(f"Sync Error: {e}")
        else:
            st.warning("⚠️ Run Bulk Audit first to generate data.")

# ===============================
# TITLE SECTION
# ===============================
st.title("🛡️ Audit-AI Hybrid Forensic Engine")
st.markdown("### Advanced Intelligence for Indian Accounting Systems")
st.divider()

# ===============================
# FORENSIC UTILITIES
# ===============================
def identify_columns(df):
    detected = {'Date':None,'Total':None,'Narration':None,'Party':None}
    clean = {c:str(c).lower().replace(" ","") for c in df.columns}

    for original, cleaned in clean.items():
        if 'date' in cleaned: detected['Date']=original
        elif any(x in cleaned for x in ['total','amount','value','amt']): detected['Total']=original
        elif any(x in cleaned for x in ['narr','desc','memo','partic']): detected['Narration']=original
        elif any(x in cleaned for x in ['party','vendor','ledger','name']): detected['Party']=original
    return detected

def build_features(df):
    # Ensure columns exist and are clean
    df['Total'] = pd.to_numeric(df['Total'], errors='coerce').fillna(0)
    df['is_round'] = (df['Total'] % 1000 == 0).astype(int)

    df['Month'] = pd.to_datetime(df['Date'], errors='coerce').dt.month.fillna(1)
    df['Is_Quarter_End'] = df['Month'].isin([3,6,9,12]).astype(int)

    # Calculate Daily Count for splitting detection
    if 'Party' in df.columns and 'Date' in df.columns:
        df['vendor_daily_count'] = df.groupby(['Date','Party'])['Total'].transform('count')
    else:
        df['vendor_daily_count'] = 1

    # Forensic Keyword Flags
    keywords = ['cash','gift','personal','split','adjustment','urgent','repair','misc']
    df['keyword_flag'] = df['Narration'].str.lower().apply(
        lambda x: 1 if any(k in str(x) for k in keywords) else 0
    )

    # 5 Numeric Columns
    numeric_cols = ['Total','is_round','Is_Quarter_End','vendor_daily_count','keyword_flag']
    numeric = df[numeric_cols].values

    # 100 Text Features
    text = tfidf.transform(df['Narration'].astype(str)).toarray()

    X = np.hstack((numeric,text))
    return X, numeric

def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)

# ===============================
# UI TABS
# ===============================
tab1, tab2 = st.tabs(["📝 Single Entry Check", "📂 Bulk Ledger Audit"])

# ===============================
# TAB 1: SINGLE TRANSACTION
# ===============================
with tab1:
    st.subheader("Manual Forensic Scan")
    col1, col2 = st.columns(2)
    with col1:
        amt = st.number_input("Amount (₹)", 0.0, 1000000.0, 5000.0)
        date_in = st.date_input("Transaction Date")
    with col2:
        narr_in = st.text_input("Narration", "Payment for office services")

    if st.button("Run Scan"):
        temp_df = pd.DataFrame({
        "Total": [amt],
        "Date": [pd.to_datetime(date_in)],
        "Party": ["Manual"],
        "Narration": [narr_in]
    })

        X, numeric = build_features(temp_df)

        if X.shape[1] != EXPECTED_FEATURES:
            st.error("Feature mismatch between training and app.")
        else:
        # 1️⃣ Pattern Risk (Supervised)
            pattern_risk = model.predict_proba(X)[0][1]

        # 2️⃣ Anomaly Detection (Binary)
            anomaly_flag = iso_forest.predict(numeric)[0]  # -1 = anomaly
            anomaly_score = 1.0 if anomaly_flag == -1 else 0.0

        # 3️⃣ Final Composite (Safer weighting for single entry)
            final_score = (0.85 * pattern_risk) + (0.15 * anomaly_score)

            st.write(f"**Pattern Risk:** {pattern_risk:.4f}")
            st.write(f"**Anomaly Triggered:** {'Yes' if anomaly_flag == -1 else 'No'}")
            st.write(f"**Final Composite Score:** {final_score:.4f}")
            st.divider()

            if final_score > risk_threshold:
                st.error(f"🚨 HIGH RISK DETECTED ({final_score*100:.2f}%)")
            else:
                st.success(f"✅ TRANSACTION SAFE ({final_score*100:.2f}%)")
# ===============================
# TAB 2: MASS AUDIT
# ===============================
with tab2:
    st.subheader("Mass Ledger Audit")
    file = st.file_uploader("Upload Ledger CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        cols = identify_columns(df)

        df = df.rename(columns={
            cols['Date']:'Date',
            cols['Total']:'Total',
            cols['Narration']:'Narration',
            cols['Party']:'Party'
        })

        if st.button("🚀 Start Bulk Forensic Audit"):
            X, numeric = build_features(df)

            if X.shape[1] != EXPECTED_FEATURES:
                st.error("Feature mismatch. Ensure training and app scripts are aligned.")
            else:
                # Calculate Risks
                df['Pattern_Risk'] = model.predict_proba(X)[:,1]
                anomaly_raw = -iso_forest.decision_function(numeric)
                df['Anomaly_Risk'] = normalize(anomaly_raw)
                df['Final_Risk'] = 0.7 * df['Pattern_Risk'] + 0.3 * df['Anomaly_Risk']
                
                df['Status'] = np.where(df['Final_Risk'] > risk_threshold, "⚠️ RISK", "✅ SAFE")
                st.session_state['last_audit_df'] = df

                # Dashboard Metrics
                st.metric("Detected Risk Rate", f"{(df['Status']=='⚠️ RISK').mean()*100:.2f}%")

                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(
                        px.pie(df, names="Status", title="Ledger Health Breakdown",
                               color="Status", color_discrete_map={'✅ SAFE':'#00cc96','⚠️ RISK':'#ff4b4b'}),
                        use_container_width=True
                    )

                with c2:
                    # Benford's Law Calculation
                    df['LD'] = df['Total'].apply(lambda x: int(str(abs(x)).replace('.','').lstrip('0')[0]) if x!=0 else None)
                    digit_counts = df['LD'].value_counts(normalize=True).sort_index()
                    ben_df = pd.DataFrame({
                        'Digit': range(1, 10),
                        'Actual': [digit_counts.get(d, 0) for d in range(1, 10)],
                        'Expected': [np.log10(1 + 1/d) for d in range(1, 10)]
                    })
                    st.plotly_chart(
                        px.bar(ben_df, x='Digit', y=['Actual', 'Expected'], barmode='group', title="Benford's Law Comparison"),
                        use_container_width=True
                    )

                st.subheader("📋 Forensic Report")
                st.dataframe(df.sort_values("Final_Risk", ascending=False), use_container_width=True)

                st.download_button("⬇️ Download Audit Report",
                                   df.to_csv(index=False).encode("utf-8"),
                                   "audit_report.csv")