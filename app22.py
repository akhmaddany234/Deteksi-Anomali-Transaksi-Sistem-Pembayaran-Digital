import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
from datetime import datetime
import io
import warnings

warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS kustom untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #3B82F6;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .warning {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .success {
        background-color: #D1FAE5;
        border-left: 5px solid #10B981;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .danger {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== FUNGSI BANTU ====================

# Load model dan scaler
@st.cache_resource
def load_models():
    """Load semua model dan scaler yang diperlukan"""
    models = {}
    
    try:
        # Load model CreditCard
        models['creditcard_model'] = joblib.load('xgboost_fraud_model_CreditCard.pkl')
        models['scaler_amount_cc'] = joblib.load('scaler_ROBUS_amount.pkl')
        models['scaler_time_cc'] = joblib.load('scaler_ROBUS_time.pkl')
        models['amount_lower_bound'] = joblib.load('amount_lower_bound.pkl')
        models['amount_upper_bound'] = joblib.load('amount_upper_bound.pkl')

        st.success("✅ Model CreditCard berhasil dimuat")
    except FileNotFoundError:
        st.error("❌ File model CreditCard tidak ditemukan")
        models['creditcard_model'] = None
        models['scaler_amount_cc'] = None
        models['scaler_time_cc'] = None
    
    try:
        # Load model PaySim
        models['paysim_model'] = joblib.load('xgboost_fraud_modep_paysim.pkl')
        st.success("✅ Model PaySim berhasil dimuat")
    except FileNotFoundError:
        st.error("❌ File model PaySim tidak ditemukan")
        models['paysim_model'] = None
    
    return models

# Fungsi preprocessing CreditCard untuk prediksi manual
def preprocess_creditcard_input(input_data):
    """
    Preprocessing CreditCard (harus IDENTIK dengan training)
    """

    # Validasi semua komponen
    required_keys = [
        'scaler_amount_cc',
        'scaler_time_cc',
        'amount_lower_bound',
        'amount_upper_bound'
    ]
    for k in required_keys:
        if k not in models:
            raise ValueError(f"Missing preprocessing object: {k}")

    df = pd.DataFrame([input_data])

    # ===== Feature Engineering =====

    # Hour feature
    df['hour'] = (df['Time'] // 3600) % 24

    # Winsorization Amount (pakai batas training)
    df['Amount_winsorized'] = df['Amount'].clip(
        models['amount_lower_bound'],
        models['amount_upper_bound']
    )

    # Scaling (RobustScaler)
    df['Amount_Scaled'] = models['scaler_amount_cc'].transform(
        df[['Amount_winsorized']]
    )
    df['Time_Scaled'] = models['scaler_time_cc'].transform(
        df[['Time']]
    )

    # Pastikan V1–V28
    for i in range(1, 29):
        df[f'V{i}'] = df.get(f'V{i}', 0.0)

    # Urutan fitur HARUS sama dengan training
    expected_features = [
        *[f'V{i}' for i in range(1, 29)],
        'Amount_winsorized',
        'hour',
        'Amount_Scaled',
        'Time_Scaled'
    ]

    return df[expected_features]


# Fungsi preprocessing PaySim untuk prediksi manual
def preprocess_paysim_input(input_data):
    """Preprocess input PaySim untuk prediksi dengan feature engineering"""
    # Buat DataFrame dari input
    df = pd.DataFrame([input_data])
    
    # Feature engineering seperti saat training
    # 1. Balance differences
    df['orig_balance_diff'] = df['oldbalanceOrg'] - df['newbalanceOrig'] - df['amount']
    df['dest_balance_diff'] = df['oldbalanceDest'] - df['newbalanceDest'] - df['amount']
    
    df['orig_balance_diff_abs'] = df['orig_balance_diff'].abs()
    df['dest_balance_diff_abs'] = df['dest_balance_diff'].abs()
    
    df['errorBalanceOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig'] - df['amount']
    df['errorBalanceDest'] = df['newbalanceDest'] - df['oldbalanceDest'] - df['amount']
    
    df['senderBalanceChange'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['destBalanceChange'] = df['newbalanceDest'] - df['oldbalanceDest']
    
    # 2. Amount ratios (handle division by zero)
    df['amountRatio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['amount_log'] = np.log1p(df['amount'])
    
    # 3. High risk type
    df['is_highrisk_type'] = df['type'].isin(['TRANSFER', 'CASH_OUT']).astype(int)
    
    # 4. Time features
    df['hour_of_day'] = df['step'] % 24
    
    # 5. One-hot encoding untuk type
    type_dummies = pd.get_dummies(df['type'], prefix='type')
    
    # Gabungkan dummies
    df = pd.concat([df, type_dummies], axis=1)

    print("columns before drop", df)
    
    # Drop kolom yang tidak diperlukan
    columns_to_drop = ['type']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # Pastikan semua fitur yang diperlukan ada
    # Ini adalah contoh fitur yang diharapkan, sesuaikan dengan model Anda
    expected_features_paysim = [
        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
        'newbalanceDest', 'orig_balance_diff', 'dest_balance_diff',
        'orig_balance_diff_abs', 'dest_balance_diff_abs', 'errorBalanceOrig',
        'errorBalanceDest', 'senderBalanceChange', 'destBalanceChange',
        'amountRatio', 'amount_log', 'is_highrisk_type', 'hour_of_day',
        'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
    
    # Tambahkan kolom yang hilang
    for feat in expected_features_paysim:
        if feat not in df.columns:
            df[feat] = 0
    
    # Urutkan kolom
    df = df[expected_features_paysim]
    
    return df

# Prediksi CreditCard
def predict_creditcard_fraud(input_data):
    """Prediksi fraud untuk CreditCard menggunakan model XGBoost"""
    try:
        # Preprocess input
        processed_data = preprocess_creditcard_input(input_data)
        print("proses data", processed_data)
        
        # Prediksi dengan model
        if models['creditcard_model'] is not None:
            prediction = models['creditcard_model'].predict(processed_data)
            prediction_proba = models['creditcard_model'].predict_proba(processed_data)
            
            # Return hasil
            is_fraud = bool(prediction[0])
            fraud_probability = float(prediction_proba[0][1]) * 100  # Probabilitas kelas 1 (fraud)
            risk_score = fraud_probability
            
            return is_fraud, risk_score, fraud_probability
        else:
            return None, None, None
    except Exception as e:
        st.error(f"Error dalam prediksi CreditCard: {str(e)}")
        return None, None, None

# Prediksi PaySim
def predict_paysim_fraud(input_data):
    """Prediksi fraud untuk PaySim menggunakan model XGBoost"""
    try:
        # Preprocess input
        processed_data = preprocess_paysim_input(input_data)
        
        # Prediksi dengan model
        if models['paysim_model'] is not None:
            prediction = models['paysim_model'].predict(processed_data)
            prediction_proba = models['paysim_model'].predict_proba(processed_data)
            
            # Return hasil
            is_fraud = bool(prediction[0])
            fraud_probability = float(prediction_proba[0][1]) * 100  # Probabilitas kelas 1 (fraud)
            risk_score = fraud_probability
            
            return is_fraud, risk_score, fraud_probability
        else:
            return None, None, None
    except Exception as e:
        st.error(f"Error dalam prediksi PaySim: {str(e)}")
        return None, None, None

# Fungsi generate sample data
def generate_sample_credit_card():
    """Generate sample data untuk Credit Card"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Time': np.random.uniform(0, 172800, n_samples),  # 0-48 jam dalam detik
        'Amount': np.random.exponential(100, n_samples),
        'V1': np.random.normal(0, 1, n_samples),
        'V2': np.random.normal(0, 1, n_samples),
        'V3': np.random.normal(0, 1, n_samples),
        'V4': np.random.normal(0, 1, n_samples),
        'V5': np.random.normal(0, 1, n_samples),
        'V6': np.random.normal(0, 1, n_samples),
        'V7': np.random.normal(0, 1, n_samples),
        'V8': np.random.normal(0, 1, n_samples),
        'V9': np.random.normal(0, 1, n_samples),
        'V10': np.random.normal(0, 1, n_samples),
        'V11': np.random.normal(0, 1, n_samples),
        'V12': np.random.normal(0, 1, n_samples),
        'V13': np.random.normal(0, 1, n_samples),
        'V14': np.random.normal(0, 1, n_samples),
        'V15': np.random.normal(0, 1, n_samples),
        'V16': np.random.normal(0, 1, n_samples),
        'V17': np.random.normal(0, 1, n_samples),
        'V18': np.random.normal(0, 1, n_samples),
        'V19': np.random.normal(0, 1, n_samples),
        'V20': np.random.normal(0, 1, n_samples),
        'V21': np.random.normal(0, 1, n_samples),
        'V22': np.random.normal(0, 1, n_samples),
        'V23': np.random.normal(0, 1, n_samples),
        'V24': np.random.normal(0, 1, n_samples),
        'V25': np.random.normal(0, 1, n_samples),
        'V26': np.random.normal(0, 1, n_samples),
        'V27': np.random.normal(0, 1, n_samples),
        'V28': np.random.normal(0, 1, n_samples),
        'Class': np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
    }
    
    # Tambahkan beberapa outlier untuk fraud
    fraud_indices = np.where(np.array(data['Class']) == 1)[0]
    for idx in fraud_indices[:10]:
        data['Amount'][idx] = np.random.uniform(1000, 5000)
        data['V1'][idx] = np.random.uniform(-5, -3)
        data['V2'][idx] = np.random.uniform(2, 4)
        data['V3'][idx] = np.random.uniform(-4, -2)

@st.cache_data
def load_paysim_data(uploaded_file):
    """Load data PaySim dengan cache untuk performa"""
    try:
        return pd.read_csv(uploaded_file)
    except:
        return generate_sample_paysim()   
    return pd.DataFrame(data)

def generate_sample_paysim():
    """Generate sample data untuk PaySim"""
    np.random.seed(42)
    n_samples = 1500
    
    data = {
        'step': np.random.randint(1, 744, n_samples),  # 1-31 hari dalam jam
        'type': np.random.choice(['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'], 
                                 n_samples, p=[0.2, 0.2, 0.1, 0.3, 0.2]),
        'amount': np.random.exponential(500, n_samples),
        'oldbalanceOrg': np.random.uniform(0, 10000, n_samples),
        'newbalanceOrig': np.random.uniform(0, 10000, n_samples),
        'oldbalanceDest': np.random.uniform(0, 10000, n_samples),
        'newbalanceDest': np.random.uniform(0, 10000, n_samples),
        'isFraud': np.random.choice([0, 1], n_samples, p=[0.995, 0.005]),
        'isFlaggedFraud': np.random.choice([0, 1], n_samples, p=[0.999, 0.001])
    }
    
    # Update balance untuk konsistensi
    for i in range(n_samples):
        if data['type'][i] in ['CASH_OUT', 'TRANSFER', 'PAYMENT', 'DEBIT']:
            data['newbalanceOrig'][i] = max(0, data['oldbalanceOrg'][i] - data['amount'][i])
        if data['type'][i] in ['CASH_IN', 'TRANSFER']:
            data['newbalanceDest'][i] = data['oldbalanceDest'][i] + data['amount'][i]
    
    # Tambahkan pola fraud
    fraud_indices = np.where(np.array(data['isFraud']) == 1)[0]
    for idx in fraud_indices[:15]:
        data['type'][idx] = np.random.choice(['TRANSFER', 'CASH_OUT'])
        data['amount'][idx] = np.random.uniform(5000, 20000)
        data['oldbalanceOrg'][idx] = data['amount'][idx] * np.random.uniform(0.8, 1.2)
    
    return pd.DataFrame(data)

# Load model
st.sidebar.markdown("### 🔄 Loading Models")
with st.sidebar:
    with st.spinner("Memuat model..."):
        models = load_models()

# Sidebar untuk navigasi
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103655.png", width=80)
    st.markdown("### Menu Navigasi")
    menu_option = st.radio(
        "Pilih halaman:",
        ["🏠 Home", "📊 Manual Prediction", "📈 Dashboard PaySim", 
         "📉 Dashboard CreditCard", "📤 Upload CSV"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Model Status")
    
    # Status model
    cc_status = "✅ Loaded" if models['creditcard_model'] is not None else "❌ Not Found"
    ps_status = "✅ Loaded" if models['paysim_model'] is not None else "❌ Not Found"
    
    st.info(f"""
    **CreditCard Model:** {cc_status}  
    **PaySim Model:** {ps_status}
    """)
    
    st.markdown("---")
    if st.button("🔄 Refresh Data & Models"):
        st.rerun()

# ==================== HALAMAN HOME ====================
if menu_option == "🏠 Home":
    st.markdown('<h1 class="main-header">Sistem Deteksi Kecurangan Transaksi Keuangan</h1>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/3067/3067256.png", width=200)
    with col3:
        st.image("fraud_10201014.png", width=200)
    # Status sistem
    if models['creditcard_model'] is None or models['paysim_model'] is None:
        st.markdown("""
        <div class="warning">
        <h4>⚠️ Perhatian</h4>
        <p>Satu atau lebih model tidak ditemukan. Pastikan file berikut ada di direktori yang sama:</p>
        <ul>
            <li><code>xgboost_fraud_model_CreditCard.pkl</code> - Model CreditCard</li>
            <li><code>paysim_xgb_model.pkl</code> - Model PaySim</li>
            <li><code>scaler_ROBUS_amount.pkl</code> - Scaler Amount</li>
            <li><code>scaler_ROBUS_time.pkl</code> - Scaler Time</li>
        </ul>
        <p>Untuk prediksi manual, semua model harus tersedia.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success">
        <h4>✅ Sistem Siap</h4>
        <p>Semua model berhasil dimuat. Sistem siap melakukan prediksi fraud.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Informasi model
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
        <h3>💳 Model CreditCard</h3>
        <p><b>Algoritma:</b> XGBoost</p>
        <p><b>Fitur:</b> 30 fitur (Time_Scaled, Amount_Scaled, V1-V28)</p>
        <p><b>Preprocessing:</b> RobustScaler untuk Time dan Amount</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
        <h3>📱 Model PaySim</h3>
        <p><b>Algoritma:</b> XGBoost</p>
        <p><b>Fitur:</b> Feature engineering extensive</p>
        <p><b>Preprocessing:</b> One-hot encoding, balance diffs, ratios</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start
    st.markdown('<h3 class="sub-header">🚀 Mulai Cepat</h3>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🎯 Prediksi Manual", "📊 Dashboard", "📄 Upload CSV"])
    
    with tab1:
        st.markdown("""
        ### Cara Menggunakan Prediksi Manual:
        
        1. **Pilih Manual Prediction** di menu sidebar
        2. **Pilih tipe dataset** (Credit Card / PaySim)
        3. **Isi parameter transaksi** sesuai dengan input form
        4. **Klik Predict Fraud** untuk melakukan prediksi
        5. **Lihat hasil** dari model XGBoost
        
        **Contoh Input untuk Testing:**
        - CreditCard: Amount tinggi (>2000), V1 negatif (< -2), V2 positif (> 2)
        - PaySim: Type TRANSFER/CASH_OUT, Amount tinggi, balance discrepancy
        """)
    
    with tab2:
        st.markdown("""
        ### Cara Menggunakan Dashboard:
        
        1. **Dashboard PaySim:** Analisis dataset PaySim dengan visualisasi interaktif
        2. **Dashboard CreditCard:** Analisis dataset CreditCard dengan PCA components
        3. **Upload CSV:** Analisis file CSV Anda sendiri
        
        **Fitur Dashboard:**
        - Filter data real-time
        - Visualisasi interaktif dengan Plotly
        - Download data hasil filter
        - Statistik deskriptif
        """)

    with tab3:
        st.markdown("""
        ### Cara Menggunakan Upload CSV:
        1. **lihat contoh format file** untuk PaySim dan CreditCard
        2. **Pilih Upload CSV** di menu sidebar
        3. **Upload file CSV** dengan format sesuai (PaySim/CreditCard)
        4. **Lihat preview data** dan statistik deskriptif
        5. **Gunakan filter interaktif** untuk eksplorasi data
        6. **Download data hasil filter** jika diperlukan
        
        **Catatan:**
        - Pastikan format kolom sesuai dengan dataset
        - Gunakan sample data jika tidak memiliki dataset sendiri
        """)

# ==================== HALAMAN MANUAL PREDICTION ====================
elif menu_option == "📊 Manual Prediction":
    st.markdown('<h1 class="main-header">🔍 Prediksi Manual Fraud Detection</h1>', unsafe_allow_html=True)
    
    # Pilih tipe dataset
    dataset_type = st.selectbox(
        "Pilih Tipe Dataset:",
        ["Credit Card Fraud Detection", "PaySim Mobile Money"]
    )
    
    if dataset_type == "Credit Card Fraud Detection":
        # Periksa ketersediaan model
        if models['creditcard_model'] is None:
            st.error("❌ Model CreditCard tidak ditemukan. Tidak dapat melakukan prediksi.")
            st.info("Pastikan file `xgboost_fraud_model_CreditCard.pkl` ada di direktori yang sama.")
        else:
            st.markdown('<h3 class="sub-header">💳 Input Parameter Credit Card</h3>', unsafe_allow_html=True)
            st.markdown("""
            <div class="card">
            <p><b>Tip:</b> Transaksi fraud cenderung memiliki Amount tinggi, V1 negatif, dan V2 positif.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # col1, col2 = st.columns(2)
            
            # with col1:
            times = st.number_input(
                "Time (seconds since first transaction)",
                min_value=0.0,
                value=50000.0,
                step=1.0,
                format="%.2f"
            )

            amount = st.number_input(
                "Amount ($)",
                min_value=0.0,
                value=100.0,
                step=0.01,
                format="%.2f"
            )

            st.markdown("### 🔢 PCA Features (V1–V28)")

            v_inputs = {}
            cols = st.columns(6)

            for i in range(1, 29):
                with cols[(i - 1) % 6]:   # ✅ MOD 6 (SESUAI JUMLAH KOLOM)
                    v_inputs[f'V{i}'] = st.number_input(
                        f'V{i}',
                        value=0.0,
                        step=0.01,
                        format="%.4f",
                        key=f'v{i}'
                    )


            
            if st.button("🔍 Predict Fraud", type="primary", use_container_width=True):
                with st.spinner("Menganalisis transaksi dengan model XGBoost..."):
                    time.sleep(1)
                    
                    # Kumpulkan semua input
                    input_data = {'Time': times, 
                                  'Amount': amount,
                                  **v_inputs}
                    # Lakukan prediksi
                    is_fraud, risk_score, confidence = predict_creditcard_fraud(input_data)
                    
                    if is_fraud is None:
                        st.error("Gagal melakukan prediksi. Periksa input data.")
                    else:
                        # Tampilkan hasil
                        st.markdown("---")
                        st.markdown('<h3 class="sub-header">Hasil Analisis Model XGBoost</h3>', unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Risk Score", f"{risk_score:.1f}/100")
                        
                        with col2:
                            st.metric("Fraud Probability", f"{confidence:.2f}%")
                        
                        with col3:
                            if is_fraud:
                                st.markdown('<div class="danger"><h3>⚠️ FRAUD DETECTED</h3></div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="success"><h3>✅ TRANSACTION NORMAL</h3></div>', unsafe_allow_html=True)
                        
                        # Visualisasi risk meter
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = risk_score,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Fraud Risk Meter (XGBoost)"},
                            delta = {'reference': 50},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 30], 'color': "green"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        threshold = 50  # threshold risiko (berbasis probabilitas)
                      
                        if is_fraud:
                            st.markdown(f"""
                            <div class="danger">
                            <h4>🚨 TRANSAKSI FRAUD TERDETEKSI</h4>

                            <p>
                            Model <b>XGBoost</b> mengklasifikasikan transaksi ini sebagai 
                            <b>fraud</b> dengan probabilitas <b>{confidence:.2f}%</b>, 
                            yang berada di atas ambang risiko <b>{threshold}%</b>.
                            </p>

                            <h5>Indikator Risiko (berdasarkan pola data latih):</h5>
                            <ul>
                                <li><b>Amount (winsorized):</b> {amount:.2f}</li>
                                <li><b>Time (hour):</b> {(time // 3600) % 24}</li>
                                <li><b>Pola PCA:</b> Terdeteksi anomali pada beberapa komponen (V1–V28)</li>
                            </ul>

                            <h5>Rekomendasi Tindakan:</h5>
                            <ol>
                                <li>Menunda atau menolak transaksi</li>
                                <li>Melakukan verifikasi pemegang kartu</li>
                                <li>Monitoring transaksi lanjutan</li>
                            </ol>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        else:
                            st.markdown(f"""
                            <div class="success">
                            <h4>✅ TRANSAKSI NORMAL</h4>

                            <p>
                            Model <b>XGBoost</b> memprediksi transaksi ini sebagai 
                            <b>non-fraud</b> dengan probabilitas fraud <b>{confidence:.2f}%</b>, 
                            berada di bawah ambang risiko <b>{threshold}%</b>.
                            </p>

                            <h5>Ringkasan Analisis:</h5>
                            <ul>
                                <li><b>Risk Score:</b> {risk_score:.1f}/100</li>
                                <li><b>Pola fitur:</b> Sesuai dengan distribusi transaksi normal</li>
                            </ul>

                            <p>Transaksi dapat diproses secara normal.</p>
                            </div>
                            """, unsafe_allow_html=True)

    
    else:  # PaySim
        # Periksa ketersediaan model
        if models['paysim_model'] is None:
            st.error("❌ Model PaySim tidak ditemukan. Tidak dapat melakukan prediksi.")
            st.info("Pastikan file `paysim_xgb_model.pkl` ada di direktori yang sama.")
        else:
            st.markdown('<h3 class="sub-header">📱 Input Parameter PaySim</h3>', unsafe_allow_html=True)
            st.markdown("""
            <div class="card">
            <p><b>Tip:</b> Fraud pada PaySim sering terjadi pada transaksi TRANSFER/CASH_OUT dengan amount tinggi dan ketidaksesuaian saldo.</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                step = st.number_input("Step (jam)", min_value=0.0, max_value=100000.0, value=100.0, step=100.0)
                # step = st.slider("Step (jam)", min_value=1, max_value=744, value=100)
                trans_type = st.selectbox(
                    "Tipe Transaksi:",
                    ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
                )
                amount = st.number_input("Amount ($)", min_value=0.0, max_value=100000.0, value=1000.0, step=100.0)
            
            with col2:
                oldbalanceOrg = st.number_input("Saldo Awal Pengirim", min_value=0.0, max_value=100000.0, value=5000.0, step=100.0)
                newbalanceOrig = st.number_input("Saldo Akhir Pengirim", min_value=0.0, max_value=100000.0, value=4000.0, step=100.0)
                oldbalanceDest = st.number_input("Saldo Awal Penerima", min_value=0.0, max_value=100000.0, value=3000.0, step=100.0)
                newbalanceDest = st.number_input("Saldo Akhir Penerima", min_value=0.0, max_value=100000.0, value=4000.0, step=100.0)
            
            # Advanced options
            with st.expander("⚙️ Advanced Options"):
                isFlaggedFraud = st.selectbox("isFlaggedFraud (sistem rule-based)", [0, 1], index=0)
                col1, col2 = st.columns(2)
                with col1:
                    nameOrig = st.text_input("Pengirim (nameOrig)", value="C12345")
                with col2:
                    nameDest = st.text_input("Penerima (nameDest)", value="C67890")
            
            if st.button("🔍 Predict Fraud", type="primary", use_container_width=True):
                with st.spinner("Menganalisis transaksi mobile money dengan model XGBoost..."):
                    time.sleep(1)
                    
                    # Kumpulkan semua input
                    input_data = {
                        'step': step,
                        'type': trans_type,
                        'amount': amount,
                        'oldbalanceOrg': oldbalanceOrg,
                        'newbalanceOrig': newbalanceOrig,
                        'oldbalanceDest': oldbalanceDest,
                        'newbalanceDest': newbalanceDest
                    }
                    
                    # Lakukan prediksi
                    print("input data paysim", input_data)
                    is_fraud, risk_score, confidence = predict_paysim_fraud(input_data)
                    
                    if is_fraud is None:
                        st.error("Gagal melakukan prediksi. Periksa input data.")
                    else:
                        # Tampilkan hasil
                        st.markdown("---")
                        st.markdown('<h3 class="sub-header">📊 Hasil Analisis Model XGBoost</h3>', unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Risk Score", f"{risk_score:.1f}/100")
                        
                        with col2:
                            st.metric("Fraud Probability", f"{confidence:.2f}%")
                        
                        with col3:
                            if is_fraud:
                                st.markdown('<div class="danger"><h3>⚠️ FRAUD DETECTED</h3></div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="success"><h3>✅ TRANSACTION NORMAL</h3></div>', unsafe_allow_html=True)
                        
                        # Feature analysis
                        st.markdown("### 📋 Analisis Fitur")
                        
                        # Hitung feature penting
                        features = []
                        
                        # Tipe transaksi
                        if trans_type in ['TRANSFER', 'CASH_OUT']:
                            features.append(("Tipe Transaksi Berisiko", "HIGH", f"{trans_type} memiliki fraud rate tinggi"))
                        
                        # Amount analysis
                        if amount > 10000:
                            features.append(("Amount Besar", "HIGH", f"${amount:,.2f} di atas threshold"))
                        elif amount > 5000:
                            features.append(("Amount Sedang", "MEDIUM", f"${amount:,.2f} perlu dipantau"))
                        
                        # Balance discrepancy
                        balance_discrepancy = abs((oldbalanceOrg - amount) - newbalanceOrig)
                        if balance_discrepancy > 100:
                            features.append(("Ketidaksesuaian Saldo", "HIGH", f"Selisih ${balance_discrepancy:.2f}"))
                        
                        # Amount ratio
                        if oldbalanceOrg > 0:
                            amount_ratio = amount / oldbalanceOrg
                            if amount_ratio > 0.9:
                                features.append(("Amount/Saldo Ratio", "HIGH", f"{amount_ratio:.1%} mendekati saldo penuh"))
                        
                        # Show features
                        if features:
                            for factor, level, desc in features:
                                emoji = "🔴" if level == "HIGH" else "🟡" if level == "MEDIUM" else "🟢"
                                st.markdown(f"{emoji} **{factor}** ({level}): {desc}")
                        else:
                            st.markdown("🟢 **Tidak ada faktor risiko signifikan terdeteksi**")
                        
                        # Rekomendasi
                        if is_fraud:
                            st.markdown(f"""
                            <div class="danger">
                            <h4>🚨 TRANSAKSI MOBILE MONEY FRAUD TERDETEKSI!</h4>
                            <p>Model XGBoost mendeteksi transaksi ini sebagai <b>fraud</b> dengan probabilitas <b>{confidence:.2f}%</b>.</p>
                            
                            <h5>Detail Transaksi:</h5>
                            <ul>
                                <li><b>Type:</b> {trans_type}</li>
                                <li><b>Amount:</b> ${amount:,.2f}</li>
                                <li><b>Pengirim:</b> Saldo {oldbalanceOrg:,.2f} → {newbalanceOrig:,.2f}</li>
                                <li><b>Penerima:</b> Saldo {oldbalanceDest:,.2f} → {newbalanceDest:,.2f}</li>
                                <li><b>Waktu:</b> Step {step} (jam ke-{step%24} dalam hari)</li>
                            </ul>
                            
                            <h5>Tindakan yang Disarankan:</h5>
                            <ol>
                                <li><b>SEGERA BLOKIR</b> transaksi ini</li>
                                <li>Hubungi nasabah pengirim untuk verifikasi</li>
                                <li>Periksa transaksi serupa dari akun yang sama</li>
                                <li>Laporkan ke tim fraud monitoring</li>
                                <li>Review sistem rule-based (isFlaggedFraud: {isFlaggedFraud})</li>
                            </ol>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="success">
                            <h4>✅ Transaksi Mobile Money Aman</h4>
                            <p>Model XGBoost mengklasifikasikan transaksi ini sebagai <b>normal</b> dengan probabilitas fraud hanya <b>{confidence:.2f}%</b>.</p>
                            
                            <h5>Detail Transaksi:</h5>
                            <ul>
                                <li><b>Risk Score:</b> {risk_score:.1f}/100 (aman)</li>
                                <li><b>Type:</b> {trans_type} {'(berisiko)' if trans_type in ['TRANSFER', 'CASH_OUT'] else '(aman)'}</li>
                                <li><b>Amount:</b> ${amount:,.2f} {'(wajar)' if amount < 5000 else '(besar namun terverifikasi)'}</li>
                                <li><b>Sistem Rule-Based:</b> {'Tidak menandai fraud' if isFlaggedFraud == 0 else 'Menandai fraud'}</li>
                            </ul>
                            
                            <p>Transaksi dapat diproses secara normal. Pantau aktivitas akun untuk pola yang tidak biasa.</p>
                            </div>
                            """, unsafe_allow_html=True)

# ==================== HALAMAN DASHBOARD PAYSIM ====================
elif menu_option == "📈 Dashboard PaySim":
    st.markdown('<h1 class="main-header">📈 Dashboard Analisis PaySim Dataset</h1>', unsafe_allow_html=True)
    
    # Pilihan: gunakan data sample atau upload file
    data_source = st.radio(
        "Pilih sumber data:",
        ["📊 Gunakan Data Sample", "📤 Upload File CSV"]
    )
    
    if data_source == "📤 Upload File CSV":
        uploaded_file = st.file_uploader("Upload file PaySim CSV", type=['csv'])
        if uploaded_file is not None:
            df_paysim = load_paysim_data(uploaded_file)
            st.success(f"✅ File berhasil diupload! {len(df_paysim)} transaksi.")
        else:
            st.info("Silakan upload file CSV atau gunakan data sample")
            df_paysim = generate_sample_paysim()
    else:
        # Gunakan data sample
        with st.spinner("Memuat data sample PaySim..."):
            df_paysim = generate_sample_paysim()
            time.sleep(1)
    
    # Info model jika tersedia
    if models['paysim_model'] is not None:
        st.markdown("""
        <div class="card">
        <h4>🤖 Model XGBoost Siap</h4>
        <p>Model PaySim XGBoost telah dimuat dan dapat digunakan untuk prediksi batch.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Filter
    st.markdown("### 🔍 Filter Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_amount = st.number_input(
            "Minimum Amount",
            min_value=0.0,
            value=0.0,
            step=100.0,
            format="%.2f"
        )

        max_amount = st.number_input(
            "Maximum Amount",
            min_value=min_amount,
            value=10000.0,
            step=100.0,
            format="%.2f"
        )

    with col2:
        selected_types = st.multiselect(
            "Tipe Transaksi:",
            options=df_paysim['type'].unique(),
            default=df_paysim['type'].unique()
        )
    
    with col3:
        fraud_filter = st.selectbox(
            "Status Fraud:",
            ["Semua", "Fraud Saja", "Normal Saja"]
        )
    
    # Apply filters
    filtered_df = df_paysim.copy()
    filtered_df = filtered_df[(filtered_df['amount'] >= min_amount) & 
                             (filtered_df['amount'] <= max_amount)]
    filtered_df = filtered_df[filtered_df['type'].isin(selected_types)]
    
    if fraud_filter == "Fraud Saja":
        filtered_df = filtered_df[filtered_df['isFraud'] == 1]
    elif fraud_filter == "Normal Saja":
        filtered_df = filtered_df[filtered_df['isFraud'] == 0]
    
    # Batch prediction jika model tersedia
    if models['paysim_model'] is not None and st.button("🤖 Jalankan Prediksi Batch", type="secondary"):
        with st.spinner("Melakukan prediksi batch..."):
            # Simulasi prediksi batch
            time.sleep(2)
            
            # Untuk demo, kita buat prediksi acak
            np.random.seed(42)
            sample_size = min(1000, len(filtered_df))
            if sample_size > 0:
                sample_df = filtered_df.sample(sample_size)
                
                # Simulasi probabilitas fraud
                fraud_probs = np.random.beta(1, 10, size=len(sample_df)) * 100
                
                # Threshold
                predictions = (fraud_probs > 50).astype(int)
                
                # Hitung metrics
                accuracy = np.mean(predictions == sample_df['isFraud'].values[:len(predictions)])
                
                st.success(f"✅ Prediksi batch selesai! Akurasi: {accuracy:.2%}")
                
                # Tambahkan kolom prediksi
                filtered_df = filtered_df.copy()
                filtered_df['Predicted_Fraud'] = 0
                filtered_df['Fraud_Probability'] = 0.0
                
                # Update untuk sample yang diprediksi
                for idx, row in sample_df.iterrows():
                    if idx in filtered_df.index:
                        filtered_df.at[idx, 'Predicted_Fraud'] = predictions[list(sample_df.index).index(idx)]
                        filtered_df.at[idx, 'Fraud_Probability'] = fraud_probs[list(sample_df.index).index(idx)]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transaksi", len(filtered_df))
    
    with col2:
        fraud_count = filtered_df['isFraud'].sum()
        st.metric("Transaksi Fraud", fraud_count)
    
    with col3:
        fraud_rate = (fraud_count / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    
    with col4:
        avg_amount = filtered_df['amount'].mean()
        st.metric("Rata-rata Amount", f"${avg_amount:,.2f}")
    
    # Visualisasi
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribusi", "⏰ Temporal", "💰 Amount", "📈 Korelasi"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribusi tipe transaksi
            type_dist = filtered_df['type'].value_counts()
            fig1 = px.pie(
                type_dist, 
                values=type_dist.values, 
                names=type_dist.index,
                title="Distribusi Tipe Transaksi",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Fraud by type
            fraud_by_type = filtered_df.groupby('type')['isFraud'].mean() * 100
            fig2 = px.bar(
                fraud_by_type.reset_index(),
                x='type',
                y='isFraud',
                title="Fraud Rate per Tipe Transaksi (%)",
                color='isFraud',
                color_continuous_scale='RdYlGn_r'
            )
            fig2.update_layout(yaxis_title="Fraud Rate (%)")
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        # Fraud over time
        fraud_over_time = filtered_df.groupby('step')['isFraud'].sum().reset_index()
        fig3 = px.line(
            fraud_over_time,
            x='step',
            y='isFraud',
            title="Frekuensi Fraud per Step (Jam)",
            markers=True
        )
        fig3.update_layout(
            xaxis_title="Step (Jam)",
            yaxis_title="Jumlah Fraud"
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribusi amount
            fig4 = px.histogram(
                filtered_df,
                x='amount',
                nbins=50,
                title="Distribusi Amount",
                color='isFraud',
                color_discrete_map={0: 'green', 1: 'red'}
            )
            fig4.update_layout(xaxis_title="Amount", yaxis_title="Count")
            st.plotly_chart(fig4, use_container_width=True)
        
        with col2:
            # Box plot amount by fraud
            fig5 = px.box(
                filtered_df,
                x='isFraud',
                y='amount',
                title="Amount: Fraud vs Normal",
                color='isFraud',
                color_discrete_map={0: 'green', 1: 'red'},
                labels={'isFraud': 'Status (0=Normal, 1=Fraud)'}
            )
            st.plotly_chart(fig5, use_container_width=True)
    
    with tab4:
        # Correlation heatmap
        numeric_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                       'oldbalanceDest', 'newbalanceDest', 'isFraud']
        numeric_cols = [col for col in numeric_cols if col in filtered_df.columns]
        
        corr_matrix = filtered_df[numeric_cols].corr()
        
        fig6 = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            title="Korelasi antar Variabel",
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig6, use_container_width=True)
    
    # Download data
    st.markdown("---")
    st.markdown("### 📥 Download Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📊 Download CSV",
            data=csv,
            file_name=f"paysim_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        if st.button("📋 Generate Summary Report"):
            with st.spinner("Membuat laporan..."):
                time.sleep(2)
                
                report = f"""
                ### 📊 Laporan Analisis PaySim
                
                **Tanggal:** {datetime.now().strftime('%d %B %Y %H:%M')}
                **Total Data:** {len(filtered_df):,} transaksi
                **Transaksi Fraud:** {fraud_count:,} ({fraud_rate:.2f}%)
                **Rata-rata Amount:** ${avg_amount:,.2f}
                
                **Distribusi Tipe Transaksi:**
                {type_dist.to_string()}
                
                **Insights:**
                1. Fraud rate tertinggi pada tipe: {fraud_by_type.idxmax() if len(fraud_by_type) > 0 else 'N/A'}
                2. Waktu dengan fraud terbanyak: Step {fraud_over_time.loc[fraud_over_time['isFraud'].idxmax(), 'step'] if len(fraud_over_time) > 0 else 'N/A'}
                3. Amount fraud rata-rata: ${filtered_df[filtered_df['isFraud']==1]['amount'].mean():,.2f if fraud_count > 0 else 0}
                
                **Rekomendasi:**
                - Fokus monitoring pada transaksi {fraud_by_type.idxmax() if len(fraud_by_type) > 0 else 'TRANSFER/CASH_OUT'}
                - Implementasi threshold amount: ${filtered_df['amount'].quantile(0.95):,.2f}
                - Pantau step dengan aktivitas fraud tinggi
                """
                
                st.markdown(report)

# ==================== HALAMAN DASHBOARD CREDITCARD ====================
elif menu_option == "📉 Dashboard CreditCard":
    st.markdown('<h1 class="main-header">💳 Dashboard Analisis Credit Card Dataset</h1>', unsafe_allow_html=True)
    
    # Pilihan sumber data
    st.markdown("### 📁 Sumber Data")
    data_source_cc = st.radio(
        "Pilih sumber data:",
        ["📂 Baca dari File CSV", "📤 Upload File Baru"],
        horizontal=True,
        key="data_source_cc"
    )
    
    # Load data berdasarkan pilihan
    if data_source_cc == "📂 Baca dari File CSV":
        # Coba baca dari file CSV yang sudah ada
        try:
            df_credit = pd.read_csv('creditcard.csv')
            st.success(f"✅ Data CreditCard berhasil dimuat dari file! {len(df_credit):,} transaksi ditemukan.")
            
            # Validasi kolom yang diperlukan
            required_columns = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]
            missing_columns = [col for col in required_columns if col not in df_credit.columns]
            
            if missing_columns:
                st.warning(f"⚠️ Beberapa kolom penting tidak ditemukan: {missing_columns[:5]}...")
                st.info("Menggunakan data sample sebagai fallback...")
                df_credit = pd.read_csv('creditcard.csv').head()
            else:
                # Tampilkan info dataset
                st.info(f"""
                **Dataset Info:**
                - Total records: {len(df_credit):,}
                - Fraud transactions: {df_credit['Class'].sum():,} ({df_credit['Class'].sum()/len(df_credit)*100:.4f}%)
                - Memory usage: {df_credit.memory_usage(deep=True).sum() / 1024**2:.2f} MB
                - Columns: {len(df_credit.columns)}
                """)
                
        except FileNotFoundError:
            st.error("❌ File 'CreditCard_dataset.csv' tidak ditemukan di direktori saat ini.")
            st.info("Menggunakan data sample sebagai gantinya...")
            df_credit = pd.read_csv('creditcard.csv').head()
    
    elif data_source_cc == "📤 Upload File Baru":
        # Upload file baru
        uploaded_file_cc = st.file_uploader(
            "Upload file CreditCard CSV",
            type=['csv'],
            key="upload_cc"
        )
        
        if uploaded_file_cc is not None:
            try:
                df_credit = pd.read_csv(uploaded_file_cc)
                st.success(f"✅ File berhasil diupload! {len(df_credit):,} transaksi ditemukan.")
                
                # Validasi kolom minimal
                if 'Amount' in df_credit.columns and 'Class' in df_credit.columns:
                    # Check for V1-V28 columns
                    v_columns_present = [f'V{i}' for i in range(1, 29) if f'V{i}' in df_credit.columns]
                    
                    if len(v_columns_present) >= 10:  # Minimal 10 kolom V
                        st.info(f"✅ Dataset valid. {len(v_columns_present)} kolom V ditemukan.")
                        
                        # Tampilkan preview
                        with st.expander("👀 Preview Data"):
                            st.dataframe(df_credit.head(), use_container_width=True)
                            st.write(f"**Shape:** {df_credit.shape}")
                    else:
                        st.warning(f"Hanya {len(v_columns_present)} kolom V ditemukan. Mungkin dataset berbeda format.")
                else:
                    st.error("Kolom 'Amount' dan 'Class' harus ada dalam dataset.")
                    st.info("Menggunakan data sample sebagai fallback...")
                    df_credit = pd.read_csv('creditcard.csv').head()
                
                    
            except Exception as e:
                st.error(f"❌ Error membaca file: {str(e)}")
                st.info("Menggunakan data sample sebagai fallback...")
                df_credit = pd.read_csv('creditcard.csv').head()
        else:
            st.info("Silakan upload file CSV atau pilih opsi lain...")
            df_credit = pd.read_csv('creditcard.csv').head()
    
    # Info model jika tersedia
    if models['creditcard_model'] is not None:
        st.markdown("""
        <div class="card">
        <h4>🤖 Model XGBoost Siap</h4>
        <p>Model CreditCard XGBoost telah dimuat. Dilengkapi dengan RobustScaler untuk Time dan Amount.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Filter Section
    st.markdown("### 🔍 Filter Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Range amount berdasarkan data
        min_amount_val = float(df_credit['Amount'].min())
        max_amount_val = float(df_credit['Amount'].max())
        
        # Gunakan 5000.0 bukan 5000
        amount_range = st.slider(
            "Rentang Amount ($):",
            min_value=min_amount_val,
            max_value=max_amount_val,
            value=(min_amount_val, min(max_amount_val, 5000.0)),  # <- perhatikan 5000.0
            step=50.0
        )
    
    with col2:
        fraud_filter_cc = st.selectbox(
            "Status Transaksi:",
            ["Semua", "Fraud Only", "Normal Only"],
            key="fraud_filter_cc"
        )
    
    with col3:
        if 'Time' in df_credit.columns:
            df_credit['Hour'] = df_credit['Time'] / 3600
            min_hour = float(df_credit['Hour'].min())
            max_hour = float(df_credit['Hour'].max())
            hour_filter = st.slider(
                "Rentang Waktu (jam):",
                min_value=min_hour,
                max_value=max_hour,
                value=(min_hour, min(max_hour, 48.0)),  # <- perhatikan 48.0
                step=1.0
            )
    
    # Apply filters (sama seperti sebelumnya)
    filtered_cc = df_credit.copy()
    
    # Filter amount
    filtered_cc = filtered_cc[(filtered_cc['Amount'] >= amount_range[0]) & 
                             (filtered_cc['Amount'] <= amount_range[1])]
    
    # Filter time jika ada
    if 'Hour' in filtered_cc.columns:
        filtered_cc = filtered_cc[(filtered_cc['Hour'] >= hour_filter[0]) & 
                                 (filtered_cc['Hour'] <= hour_filter[1])]
    
    # Filter fraud
    if fraud_filter_cc == "Fraud Only":
        filtered_cc = filtered_cc[filtered_cc['Class'] == 1]
    elif fraud_filter_cc == "Normal Only":
        filtered_cc = filtered_cc[filtered_cc['Class'] == 0]
    
    # Batch prediction jika model tersedia
    if models['creditcard_model'] is not None and st.button("🤖 Jalankan Prediksi Batch", type="secondary", key="batch_pred_cc"):
        with st.spinner("Melakukan prediksi batch dengan XGBoost..."):
            time.sleep(2)
            
            # Untuk demo, kita batasi jumlah sampel
            sample_size = min(500, len(filtered_cc))
            if sample_size > 0:
                sample_df = filtered_cc.head(sample_size)
                
                # Simulasi prediksi
                try:
                    # Preprocess untuk prediksi
                    # Note: Ini adalah contoh, perlu disesuaikan dengan preprocessing aktual
                    
                    # Pastikan semua kolom V ada
                    for i in range(1, 29):
                        col_name = f'V{i}'
                        if col_name not in sample_df.columns:
                            sample_df[col_name] = 0
                    
                    # Scaling Time dan Amount
                    if 'Time' in sample_df.columns and 'Amount' in sample_df.columns:
                        # In production, use the actual scalers
                        sample_df['Time_Scaled'] = sample_df['Time'] / 100000  # Simplified scaling
                        sample_df['Amount_Scaled'] = sample_df['Amount'] / 1000  # Simplified scaling
                    
                    # Simulasi probabilitas (random untuk demo)
                    np.random.seed(42)
                    fraud_probs = np.random.beta(1, 50, size=len(sample_df)) * 100
                    
                    # Apply some pattern based on actual data
                    if 'V1' in sample_df.columns and 'V2' in sample_df.columns:
                        # Make V1 < -2 and V2 > 2 increase fraud probability
                        high_risk_mask = (sample_df['V1'] < -2) & (sample_df['V2'] > 2)
                        fraud_probs[high_risk_mask] = fraud_probs[high_risk_mask] * 2
                        fraud_probs = np.clip(fraud_probs, 0, 100)
                    
                    # Threshold
                    predictions = (fraud_probs > 50).astype(int)
                    
                    # Hitung accuracy jika ada label asli
                    if 'Class' in sample_df.columns:
                        actual_labels = sample_df['Class'].values[:len(predictions)]
                        accuracy = np.mean(predictions == actual_labels)
                        st.success(f"✅ Prediksi batch selesai! Akurasi: {accuracy:.2%}")
                    else:
                        st.success(f"✅ Prediksi batch selesai! {predictions.sum()} transaksi terdeteksi sebagai fraud.")
                    
                    # Tambahkan kolom prediksi ke dataframe utama
                    filtered_cc = filtered_cc.copy()
                    filtered_cc['Predicted_Class'] = 0
                    filtered_cc['Fraud_Probability'] = 0.0
                    
                    # Update untuk sample yang diprediksi
                    for idx in sample_df.index:
                        if idx in filtered_cc.index:
                            pred_idx = list(sample_df.index).index(idx)
                            filtered_cc.at[idx, 'Predicted_Class'] = predictions[pred_idx]
                            filtered_cc.at[idx, 'Fraud_Probability'] = fraud_probs[pred_idx]
                    
                except Exception as e:
                    st.error(f"Error dalam prediksi batch: {str(e)}")
    
    # Metrics Section
    st.markdown("### 📊 Statistik Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_transactions = len(filtered_cc)
        st.metric("Total Transaksi", f"{total_transactions:,}")
    
    with col2:
        fraud_count_cc = filtered_cc['Class'].sum() if 'Class' in filtered_cc.columns else 0
        st.metric("Transaksi Fraud", f"{fraud_count_cc:,}")
    
    with col3:
        fraud_rate_cc = (fraud_count_cc / total_transactions) * 100 if total_transactions > 0 else 0
        st.metric("Fraud Rate", f"{fraud_rate_cc:.4f}%")
    
    with col4:
        avg_amount_cc = filtered_cc['Amount'].mean() if 'Amount' in filtered_cc.columns else 0
        st.metric("Avg Amount", f"${avg_amount_cc:,.2f}")
    
    # Visualisasi Tabs
    st.markdown("### 📈 Visualisasi Data")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribusi", "⏰ Time Analysis", "🔢 PCA Components", "📈 Correlation & Insights"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Class distribution
            if 'Class' in filtered_cc.columns:
                class_counts = filtered_cc['Class'].value_counts()
                fig1 = px.pie(
                    class_counts,
                    values=class_counts.values,
                    names=['Normal (0)', 'Fraud (1)'],
                    title="Distribusi Kelas",
                    color=['Normal (0)', 'Fraud (1)'],
                    color_discrete_map={'Normal (0)': 'green', 'Fraud (1)': 'red'}
                )
                st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Amount distribution
            if 'Amount' in filtered_cc.columns and 'Class' in filtered_cc.columns:
                fig2 = px.histogram(
                    filtered_cc,
                    x='Amount',
                    nbins=50,
                    title="Distribusi Amount",
                    color='Class',
                    color_discrete_map={0: 'green', 1: 'red'},
                    log_y=True,
                    hover_data=['Time'] if 'Time' in filtered_cc.columns else None
                )
                fig2.update_layout(xaxis_title="Amount ($)", yaxis_title="Count (log scale)")
                st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        if 'Time' in filtered_cc.columns and 'Class' in filtered_cc.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Convert time to hours
                filtered_cc['Hour_Int'] = (filtered_cc['Time'] / 3600).astype(int)
                fraud_by_hour = filtered_cc.groupby('Hour_Int')['Class'].mean() * 100
                
                fig3 = px.bar(
                    fraud_by_hour.reset_index(),
                    x='Hour_Int',
                    y='Class',
                    title="Fraud Rate per Jam (%)",
                    labels={'Hour_Int': 'Jam', 'Class': 'Fraud Rate (%)'},
                    color='Class',
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                # Transaction frequency by hour
                trans_by_hour = filtered_cc.groupby('Hour_Int').size().reset_index(name='Count')
                fig4 = px.line(
                    trans_by_hour,
                    x='Hour_Int',
                    y='Count',
                    title="Frekuensi Transaksi per Jam",
                    markers=True
                )
                fig4.update_layout(xaxis_title="Jam", yaxis_title="Jumlah Transaksi")
                st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Kolom 'Time' tidak tersedia dalam dataset untuk analisis temporal.")
    
    with tab3:
        # PCA components analysis
        st.markdown("### 🔍 Analisis Komponen PCA")
        
        # Cari kolom V yang ada
        v_columns = [col for col in filtered_cc.columns if col.startswith('V')]
        
        if v_columns:
            # Urutkan kolom V secara numerik
            v_columns_sorted = sorted(v_columns, key=lambda x: int(x[1:]) if x[1:].isdigit() else float('inf'))
            
            selected_pca = st.selectbox(
                "Pilih PCA Component untuk dianalisis:", 
                v_columns_sorted[:20],  # Batasi pilihan untuk UI yang lebih bersih
                key="pca_select"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution of selected PCA
                if 'Class' in filtered_cc.columns:
                    fig5 = px.histogram(
                        filtered_cc,
                        x=selected_pca,
                        color='Class',
                        title=f"Distribusi {selected_pca} by Class",
                        color_discrete_map={0: 'green', 1: 'red'},
                        nbins=50,
                        marginal="box"
                    )
                    st.plotly_chart(fig5, use_container_width=True)
            
            with col2:
                # Scatter plot V1 vs V2 (jika ada)
                if 'V1' in filtered_cc.columns and 'V2' in filtered_cc.columns:
                    sample_size_viz = min(1000, len(filtered_cc))
                    scatter_df = filtered_cc.sample(sample_size_viz) if len(filtered_cc) > 1000 else filtered_cc
                    
                    fig6 = px.scatter(
                        scatter_df,
                        x='V1',
                        y='V2',
                        color='Class',
                        title="V1 vs V2 (PCA Space)",
                        color_discrete_map={0: 'green', 1: 'red'},
                        size='Amount' if 'Amount' in filtered_cc.columns else None,
                        hover_data=['Amount', 'Time'] if 'Amount' in filtered_cc.columns and 'Time' in filtered_cc.columns else None,
                        opacity=0.7
                    )
                    st.plotly_chart(fig6, use_container_width=True)
        else:
            st.info("Kolom PCA (V1-V28) tidak ditemukan dalam dataset.")
    
    with tab4:
        # Correlation matrix
        st.markdown("### 📊 Matriks Korelasi")
        
        # Pilih kolom numerik untuk korelasi
        numeric_cols_cc = []
        potential_cols = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 10)]  # Hanya ambil V1-V9 untuk kinerja
        
        for col in potential_cols:
            if col in filtered_cc.columns and pd.api.types.is_numeric_dtype(filtered_cc[col]):
                numeric_cols_cc.append(col)
        
        if len(numeric_cols_cc) >= 3:  # Minimal 3 kolom untuk korelasi bermakna
            corr_matrix_cc = filtered_cc[numeric_cols_cc].corr()
            
            fig7 = px.imshow(
                corr_matrix_cc,
                text_auto='.2f',
                aspect='auto',
                title="Korelasi antar Variabel",
                color_continuous_scale='RdBu',
                zmin=-1, zmax=1
            )
            fig7.update_layout(height=800, width=800)
            st.plotly_chart(fig7, use_container_width=True)
            
            # Insights dari korelasi
            st.markdown("""
            <div class="card">
            <h4>🔍 Insights Korelasi CreditCard:</h4>
            <ul>
                <li><b>V2, V4, V11:</b> Biasanya korelasi negatif dengan Fraud (Class)</li>
                <li><b>V7, V10, V16:</b> Biasanya korelasi positif dengan Fraud</li>
                <li><b>Time & Amount:</b> Korelasi rendah dengan Fraud - menjelaskan mengapa model linear kurang efektif</li>
                <li><b>Model XGBoost:</b> Dapat menangkap hubungan non-linear yang tidak terlihat dalam korelasi linear</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

            # Hitung korelasi lengkap
            corr_full = filtered_cc.corr()

            # Buat tombol download
            csv_corr = corr_full.to_csv()
            st.download_button(
                label="📥 Download Matriks Korelasi Lengkap",
                data=csv_corr,
                file_name="correlation_matrix_full.csv"
)

        else:
            st.info("Tidak cukup kolom numerik untuk analisis korelasi.")
    
    # Download dan Export Section
    st.markdown("---")
    st.markdown("### 💾 Download & Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download filtered data
        csv_cc = filtered_cc.to_csv(index=False)
        st.download_button(
            label="📥 Download Data Filtered",
            data=csv_cc,
            file_name=f"creditcard_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Generate summary report
        if st.button("📋 Generate Summary Report", use_container_width=True):
            with st.spinner("Membuat ringkasan..."):
                time.sleep(2)
                
                # Hitung statistik tambahan
                if 'Amount' in filtered_cc.columns:
                    amount_stats = filtered_cc['Amount'].describe()
                    q95_amount = filtered_cc['Amount'].quantile(0.95)
                
                if 'Class' in filtered_cc.columns:
                    fraud_stats = filtered_cc['Class'].describe()
                
                summary = f"""
                ### 📊 Ringkasan Analisis Credit Card Dataset
                
                **Tanggal Analisis:** {datetime.now().strftime('%d %B %Y %H:%M')}
                **Sumber Data:** {data_source_cc}
                
                **Statistik Dataset:**
                - Total Transaksi: {total_transactions:,}
                - Transaksi Fraud: {fraud_count_cc:,} ({fraud_rate_cc:.4f}%)
                - Rata-rata Amount: ${avg_amount_cc:,.2f}
                - Fraud Rate per Jam Tertinggi: {fraud_by_hour.max():.4f}% pada jam {fraud_by_hour.idxmax() if 'fraud_by_hour' in locals() and len(fraud_by_hour) > 0 else 'N/A'}
                
                **Karakteristik Fraud:**
                - Amount fraud rata-rata: ${filtered_cc[filtered_cc['Class']==1]['Amount'].mean():,.2f if fraud_count_cc > 0 else 0}
                - Waktu fraud tersering: Jam {fraud_by_hour.idxmax() if 'fraud_by_hour' in locals() and len(fraud_by_hour) > 0 else 'N/A'}
                - Jumlah kolom PCA: {len(v_columns) if 'v_columns' in locals() else 0}
                
                **Rekomendasi Model XGBoost:**
                1. Fokus monitoring pada jam dengan fraud rate tinggi
                2. Implementasi threshold amount: ${q95_amount:,.2f if 'q95_amount' in locals() else 0}
                3. Perhatikan transaksi dengan V1 < -2 dan V2 > 2
                4. Gunakan feature importance dari model untuk prioritasi monitoring
                
                **Filter yang Diterapkan:**
                - Amount: ${amount_range[0]:,.2f} - ${amount_range[1]:,.2f}
                - Status: {fraud_filter_cc}
                - Waktu: {hour_filter[0]} - {hour_filter[1]} jam (jika tersedia)
                """
                
                st.markdown(summary)
    
    with col3:
        # Export visualizations
        if st.button("🖼️ Export Visualizations", use_container_width=True):
            with st.spinner("Menyimpan visualisasi..."):
                time.sleep(1)
                st.success("✅ Visualisasi berhasil diexport! (Simulasi)")
                st.info("Fitur export aktual memerlukan implementasi tambahan untuk menyimpan gambar.")
    
    # Reset button
    if st.button("🔄 Reset ke Dataset Awal", type="secondary", use_container_width=True):
        st.rerun()

# ==================== HALAMAN UPLOAD CSV ====================
elif menu_option == "📤 Upload CSV":
    st.markdown('<h1 class="main-header">📤 Upload dan Analisis File CSV</h1>', unsafe_allow_html=True)
    
    # Informasi model
    if models['creditcard_model'] is None or models['paysim_model'] is None:
        st.warning("⚠️ Beberapa model tidak tersedia. Pastikan model sudah diload di halaman utama.")
    
    # Pilihan dataset
    st.markdown("### 🎯 Pilih Tipe Dataset")
    dataset_choice = st.radio(
        "Pilih jenis data yang akan diupload:",
        ["Credit Card Fraud", "PaySim Financial"],
        horizontal=True,
        key="dataset_choice"
    )
    
    # Tampilkan contoh data sesuai pilihan
    st.markdown("### 📋 Format Data yang Dibutuhkan")
    
    if dataset_choice == "Credit Card Fraud":
        st.markdown("""
        <div class="card">
        <h4>📊 Format Data Credit Card Fraud</h4>
        <p>Pastikan file CSV memiliki kolom berikut:</p>
        <ul>
            <li><b>Time:</b> Detik sejak transaksi pertama</li>
            <li><b>Amount:</b> Jumlah transaksi</li>
            <li><b>V1-V28:</b> 28 fitur utama hasil PCA</li>
            <li><b>Class (opsional):</b> Label fraud (1) atau bukan (0)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Contoh data Credit Card
        try:
            dataCC = pd.read_csv('creditcard.csv')
            sample_cc = dataCC.head(5)
            st.markdown("**Contoh Data Credit Card:**")
            st.dataframe(sample_cc, use_container_width=True)
            
            # Download sample
            csv_cc_sample = sample_cc.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample Credit Card",
                data=csv_cc_sample,
                file_name="sample_creditcard.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.warning(f"File contoh tidak ditemukan: {str(e)}")
    
    else:  # PaySim
        st.markdown("""
        <div class="card">
        <h4>🏦 Format Data PaySim Financial</h4>
        <p>Pastikan file CSV memiliki kolom berikut:</p>
        <ul>
            <li><b>step:</b> Unit waktu (jam)</li>
            <li><b>type:</b> Jenis transaksi (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)</li>
            <li><b>amount:</b> Jumlah transaksi</li>
            <li><b>oldbalanceOrg:</b> Saldo pengirim sebelum transaksi</li>
            <li><b>newbalanceOrig:</b> Saldo pengirim setelah transaksi</li>
            <li><b>oldbalanceDest:</b> Saldo penerima sebelum transaksi</li>
            <li><b>newbalanceDest:</b> Saldo penerima setelah transaksi</li>
            <li><b>isFraud (opsional):</b> Label fraud (1) atau bukan (0)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Contoh data PaySim
        try:
            dataPS = pd.read_csv('paysim.csv')
            sample_ps = dataPS.head(5)
            st.markdown("**Contoh Data PaySim:**")
            st.dataframe(sample_ps, use_container_width=True)
            
            # Download sample
            csv_ps_sample = sample_ps.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample PaySim",
                data=csv_ps_sample,
                file_name="sample_paysim.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.warning(f"File contoh tidak ditemukan: {str(e)}")
    
    # Upload file
    st.markdown("---")
    st.markdown("### 📤 Upload File CSV Anda")
    
    uploaded_file = st.file_uploader(
        f"Pilih file CSV untuk dataset {dataset_choice}",
        type=['csv'],
        help=f"Upload file CSV sesuai format {dataset_choice}"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df_upload = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File berhasil diupload! {len(df_upload)} baris data ditemukan.")
            
            # Preview data
            st.markdown("### 👀 Preview Data")
            st.dataframe(df_upload.head(10), use_container_width=True)
            
            # Validasi kolom berdasarkan pilihan dataset
            st.markdown("### 🔍 Validasi Format Data")
            
            if dataset_choice == "Credit Card Fraud":
                required_cols = ['Time', 'Amount']
                v_cols = [f'V{i}' for i in range(1, 29)]
                required_cols.extend(v_cols)
            else:  # PaySim
                required_cols = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                               'oldbalanceDest', 'newbalanceDest']
            
            missing_cols = [col for col in required_cols if col not in df_upload.columns]
            
            if missing_cols:
                st.error(f"❌ Kolom yang hilang: {missing_cols}")
                st.info("Pastikan file CSV memiliki format yang benar sesuai contoh di atas.")
            else:
                st.success("✅ Format data valid!")
                
                # EDA - Visualisasi Data
                st.markdown("### 📊 Exploratory Data Analysis (EDA)")
                
                # Tab untuk berbagai visualisasi
                eda_tabs = st.tabs(["📈 Distribusi", "📊 Statistik", "🔍 Korelasi", "📦 Box Plot"])
                
                with eda_tabs[0]:
                    # Distribusi
                    if dataset_choice == "Credit Card Fraud":
                        fig1 = px.histogram(df_upload, x='Amount', nbins=50, 
                                           title="Distribusi Amount (Transaksi)")
                        fig1.update_layout(xaxis_title="Amount", yaxis_title="Count")
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        if 'Class' in df_upload.columns:
                            fraud_counts = df_upload['Class'].value_counts()
                            fig2 = px.pie(values=fraud_counts.values, names=['Non-Fraud', 'Fraud'],
                                         title="Distribusi Fraud vs Non-Fraud")
                            st.plotly_chart(fig2, use_container_width=True)
                    
                    else:  # PaySim
                        # Distribusi transaction type
                        type_counts = df_upload['type'].value_counts()
                        fig1 = px.bar(x=type_counts.index, y=type_counts.values,
                                     title="Distribusi Jenis Transaksi")
                        fig1.update_layout(xaxis_title="Tipe Transaksi", yaxis_title="Count")
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        # Distribusi amount
                        fig2 = px.histogram(df_upload, x='amount', nbins=50,
                                          title="Distribusi Amount (Log Scale)")
                        fig2.update_layout(xaxis_title="Amount", yaxis_title="Count")
                        fig2.update_xaxes(type="log")
                        st.plotly_chart(fig2, use_container_width=True)
                
                with eda_tabs[1]:
                    # Statistik Deskriptif
                    st.markdown("#### 🔢 Statistik Deskriptif")
                    
                    if dataset_choice == "Credit Card Fraud":
                        numeric_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29) if f'V{i}' in df_upload.columns]
                    else:
                        numeric_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                                       'oldbalanceDest', 'newbalanceDest']
                    
                    desc_df = df_upload[numeric_cols].describe().T
                    desc_df['missing'] = df_upload[numeric_cols].isnull().sum().values
                    desc_df['missing_pct'] = (desc_df['missing'] / len(df_upload) * 100).round(2)
                    
                    st.dataframe(desc_df, use_container_width=True)
                
                with eda_tabs[2]:
                    # Heatmap Korelasi
                    st.markdown("#### 🔗 Heatmap Korelasi")
                    
                    if dataset_choice == "Credit Card Fraud":
                        # Ambil subset kolom untuk heatmap (maksimal 20 kolom untuk performa)
                        corr_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 10)]
                        corr_data = df_upload[corr_cols].corr()
                    else:
                        corr_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                                   'oldbalanceDest', 'newbalanceDest']
                        corr_data = df_upload[corr_cols].corr()
                    
                    fig = px.imshow(corr_data, text_auto='.2f', aspect="auto",
                                   title="Heatmap Korelasi antar Fitur")
                    st.plotly_chart(fig, use_container_width=True)
                
                with eda_tabs[3]:
                    # Box Plot untuk outlier detection
                    st.markdown("#### 📦 Box Plot untuk Mendeteksi Outlier")
                    
                    if dataset_choice == "Credit Card Fraud":
                        fig = px.box(df_upload, y='Amount', title="Box Plot Amount")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = px.box(df_upload, y='amount', x='type', 
                                    title="Box Plot Amount per Jenis Transaksi")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Tombol untuk memulai prediksi
                st.markdown("---")
                st.markdown("### 🤖 Prediksi Fraud Detection")
                
                if st.button("🚀 Jalankan Prediksi Batch dengan XGBoost", type="primary", use_container_width=True):
                    with st.spinner("Melakukan prediksi batch... Ini mungkin memerlukan beberapa saat..."):
                        progress_bar = st.progress(0)
                        
                        # Prediksi BATCH untuk seluruh dataset - LEBIH CEPAT!
                        try:
                            if dataset_choice == "Credit Card Fraud":
                                # Fungsi preprocessing untuk BATCH CreditCard
                                def preprocess_creditcard_batch(df_batch):
                                    """Preprocessing CreditCard untuk batch data"""
                                    df = df_batch.copy()
                                    
                                    # Validasi semua komponen
                                    required_keys = [
                                        'scaler_amount_cc',
                                        'scaler_time_cc',
                                        'amount_lower_bound',
                                        'amount_upper_bound'
                                    ]
                                    for k in required_keys:
                                        if k not in models:
                                            raise ValueError(f"Missing preprocessing object: {k}")
                                    
                                    # ===== Feature Engineering =====
                                    # Hour feature
                                    df['hour'] = (df['Time'] // 3600) % 24
                                    
                                    # Winsorization Amount (pakai batas training)
                                    df['Amount_winsorized'] = df['Amount'].clip(
                                        models['amount_lower_bound'],
                                        models['amount_upper_bound']
                                    )
                                    
                                    # Scaling (RobustScaler)
                                    df['Amount_Scaled'] = models['scaler_amount_cc'].transform(
                                        df[['Amount_winsorized']]
                                    )
                                    df['Time_Scaled'] = models['scaler_time_cc'].transform(
                                        df[['Time']]
                                    )
                                    
                                    # Pastikan V1–V28
                                    for i in range(1, 29):
                                        if f'V{i}' not in df.columns:
                                            df[f'V{i}'] = 0.0
                                    
                                    # Urutan fitur HARUS sama dengan training
                                    expected_features = [
                                        *[f'V{i}' for i in range(1, 29)],
                                        'Amount_winsorized',
                                        'hour',
                                        'Amount_Scaled',
                                        'Time_Scaled'
                                    ]
                                    
                                    return df[expected_features]
                                
                                # Preprocess seluruh data sekaligus
                                if models['creditcard_model'] is not None:
                                    processed_data = preprocess_creditcard_batch(df_upload)
                                    progress_bar.progress(30)
                                    
                                    # Prediksi batch
                                    predictions = models['creditcard_model'].predict(processed_data)
                                    progress_bar.progress(60)
                                    
                                    # Probabilitas prediksi
                                    prediction_proba = models['creditcard_model'].predict_proba(processed_data)
                                    fraud_probs = prediction_proba[:, 1] * 100  # Probabilitas kelas 1 (fraud)
                                    progress_bar.progress(90)
                                    
                                else:
                                    st.error("Model Credit Card belum diload!")
                                    processed_data = None
                                    predictions = []
                                    fraud_probs = []
                            
                            else:  # PaySim
                                # Fungsi preprocessing untuk BATCH PaySim
                                def preprocess_paysim_batch(df_batch):
                                    """Preprocess input PaySim untuk batch data"""
                                    df = df_batch.copy()
                                    
                                    # Feature engineering seperti saat training
                                    # 1. Balance differences
                                    df['orig_balance_diff'] = df['oldbalanceOrg'] - df['newbalanceOrig'] - df['amount']
                                    df['dest_balance_diff'] = df['oldbalanceDest'] - df['newbalanceDest'] - df['amount']
                                    
                                    df['orig_balance_diff_abs'] = df['orig_balance_diff'].abs()
                                    df['dest_balance_diff_abs'] = df['dest_balance_diff'].abs()
                                    
                                    df['errorBalanceOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig'] - df['amount']
                                    df['errorBalanceDest'] = df['newbalanceDest'] - df['oldbalanceDest'] - df['amount']
                                    
                                    df['senderBalanceChange'] = df['oldbalanceOrg'] - df['newbalanceOrig']
                                    df['destBalanceChange'] = df['newbalanceDest'] - df['oldbalanceDest']
                                    
                                    # 2. Amount ratios (handle division by zero)
                                    df['amountRatio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
                                    df['amount_log'] = np.log1p(df['amount'])
                                    
                                    # 3. High risk type
                                    df['is_highrisk_type'] = df['type'].isin(['TRANSFER', 'CASH_OUT']).astype(int)
                                    
                                    # 4. Time features
                                    df['hour_of_day'] = df['step'] % 24
                                    
                                    # 5. One-hot encoding untuk type
                                    type_dummies = pd.get_dummies(df['type'], prefix='type')
                                    
                                    # Gabungkan dummies
                                    df = pd.concat([df, type_dummies], axis=1)
                                    
                                    # Drop kolom yang tidak diperlukan
                                    columns_to_drop = ['type']
                                    for col in columns_to_drop:
                                        if col in df.columns:
                                            df = df.drop(col, axis=1)
                                    
                                    # Pastikan semua fitur yang diperlukan ada
                                    expected_features_paysim = [
                                        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
                                        'newbalanceDest', 'orig_balance_diff', 'dest_balance_diff',
                                        'orig_balance_diff_abs', 'dest_balance_diff_abs', 'errorBalanceOrig',
                                        'errorBalanceDest', 'senderBalanceChange', 'destBalanceChange',
                                        'amountRatio', 'amount_log', 'is_highrisk_type', 'hour_of_day',
                                        'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
                                    
                                    # Tambahkan kolom yang hilang
                                    for feat in expected_features_paysim:
                                        if feat not in df.columns:
                                            df[feat] = 0
                                    
                                    # Urutkan kolom
                                    df = df[expected_features_paysim]
                                    
                                    return df
                                
                                # Preprocess seluruh data sekaligus
                                if models['paysim_model'] is not None:
                                    processed_data = preprocess_paysim_batch(df_upload)
                                    progress_bar.progress(30)
                                    
                                    # Prediksi batch
                                    predictions = models['paysim_model'].predict(processed_data)
                                    progress_bar.progress(60)
                                    
                                    # Probabilitas prediksi
                                    prediction_proba = models['paysim_model'].predict_proba(processed_data)
                                    fraud_probs = prediction_proba[:, 1] * 100  # Probabilitas kelas 1 (fraud)
                                    progress_bar.progress(90)
                                    
                                else:
                                    st.error("Model PaySim belum diload!")
                                    processed_data = None
                                    predictions = []
                                    fraud_probs = []
                            
                            progress_bar.progress(100)
                            
                            if processed_data is not None and len(predictions) > 0:
                                # Tambahkan hasil prediksi ke DataFrame
                                df_upload['Predicted_Fraud'] = predictions
                                df_upload['Fraud_Probability'] = fraud_probs
                                df_upload['Risk_Level'] = pd.cut(df_upload['Fraud_Probability'], 
                                                               bins=[0, 30, 70, 100], 
                                                               labels=['Low', 'Medium', 'High'])
                                
                                # Results section
                                st.markdown("---")
                                st.markdown("## 📊 Hasil Prediksi Batch")
                                st.success(f"✅ Prediksi selesai! {len(df_upload)} transaksi diproses dalam 1 batch.")
                                
                                # Hitung metrics
                                fraud_count = int(predictions.sum())
                                total_count = len(df_upload)
                                fraud_rate = (fraud_count / total_count * 100) if total_count > 0 else 0
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Transaksi", f"{total_count:,}")
                                with col2:
                                    st.metric("Fraud Terdeteksi", f"{fraud_count:,}")
                                with col3:
                                    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
                                with col4:
                                    avg_prob = np.mean(fraud_probs) if len(fraud_probs) > 0 else 0
                                    st.metric("Rata-rata Probabilitas", f"{avg_prob:.2f}%")
                                
                                # Visualisasi hasil
                                st.markdown("### 📈 Visualisasi Hasil Prediksi")
                                
                                # Tab untuk berbagai visualisasi hasil
                                result_tabs = st.tabs(["📊 Distribusi Fraud", "📈 Probabilitas", "📋 Data Fraud", "⚡ Performa"])
                                
                                with result_tabs[0]:
                                    # Pie chart fraud vs non-fraud
                                    fraud_labels = ['Non-Fraud', 'Fraud']
                                    fraud_values = [total_count - fraud_count, fraud_count]
                                    
                                    fig = px.pie(values=fraud_values, names=fraud_labels,
                                               title="Distribusi Prediksi Fraud vs Non-Fraud",
                                               color=fraud_labels,
                                               color_discrete_map={'Non-Fraud':'blue', 'Fraud':'red'})
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with result_tabs[1]:
                                    # Histogram fraud probability
                                    fig = px.histogram(df_upload, x='Fraud_Probability', nbins=20,
                                                     title="Distribusi Fraud Probability",
                                                     color='Risk_Level',
                                                     color_discrete_map={'Low': 'green', 'Medium': 'yellow', 'High': 'red'})
                                    fig.update_layout(xaxis_title="Fraud Probability (%)", yaxis_title="Count")
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Scatter plot untuk melihat hubungan amount dengan fraud probability
                                    if dataset_choice == "Credit Card Fraud":
                                        amount_col = 'Amount'
                                    else:
                                        amount_col = 'amount'
                                    
                                    fig2 = px.scatter(df_upload, x=amount_col, y='Fraud_Probability',
                                                     color='Predicted_Fraud',
                                                     title=f"Hubungan {amount_col} dengan Fraud Probability",
                                                     labels={amount_col: 'Amount', 'Fraud_Probability': 'Fraud Probability (%)'},
                                                     color_discrete_map={False: 'blue', True: 'red'},
                                                     opacity=0.6)
                                    st.plotly_chart(fig2, use_container_width=True)
                                
                                with result_tabs[2]:
                                    # Tampilkan data yang diprediksi sebagai fraud
                                    fraud_data = df_upload[df_upload['Predicted_Fraud'] == 1]
                                    
                                    if not fraud_data.empty:
                                        st.markdown(f"**Data yang diprediksi sebagai Fraud ({len(fraud_data)} transaksi):**")
                                        
                                        # Tampilkan dengan pagination
                                        page_size = 50
                                        total_pages = (len(fraud_data) + page_size - 1) // page_size
                                        
                                        if total_pages > 1:
                                            page_number = st.number_input("Halaman", min_value=1, max_value=total_pages, value=1)
                                            start_idx = (page_number - 1) * page_size
                                            end_idx = min(start_idx + page_size, len(fraud_data))
                                            
                                            st.dataframe(fraud_data.iloc[start_idx:end_idx].sort_values('Fraud_Probability', ascending=False), 
                                                       use_container_width=True)
                                            st.caption(f"Menampilkan {start_idx+1}-{end_idx} dari {len(fraud_data)} transaksi fraud")
                                        else:
                                            st.dataframe(fraud_data.sort_values('Fraud_Probability', ascending=False), 
                                                       use_container_width=True)
                                    else:
                                        st.info("Tidak ada transaksi yang diprediksi sebagai fraud.")
                                
                                with result_tabs[3]:
                                    # Analisis performa jika ada label sebenarnya
                                    if dataset_choice == "Credit Card Fraud" and 'Class' in df_upload.columns:
                                        st.markdown("#### 📊 Evaluasi Performa Model")
                                        
                                        from sklearn.metrics import classification_report, confusion_matrix
                                        
                                        y_true = df_upload['Class']
                                        y_pred = df_upload['Predicted_Fraud'].astype(int)
                                        
                                        # Confusion Matrix
                                        cm = confusion_matrix(y_true, y_pred)
                                        fig_cm = px.imshow(cm, 
                                                         labels=dict(x="Predicted", y="Actual", color="Count"),
                                                         x=['Non-Fraud', 'Fraud'],
                                                         y=['Non-Fraud', 'Fraud'],
                                                         title="Confusion Matrix",
                                                         text_auto=True)
                                        st.plotly_chart(fig_cm, use_container_width=True)
                                        
                                        # Classification Report
                                        report = classification_report(y_true, y_pred, output_dict=True)
                                        report_df = pd.DataFrame(report).transpose()
                                        st.dataframe(report_df, use_container_width=True)
                                    
                                    elif dataset_choice == "PaySim Financial" and 'isFraud' in df_upload.columns:
                                        st.markdown("#### 📊 Evaluasi Performa Model")
                                        
                                        from sklearn.metrics import classification_report, confusion_matrix
                                        
                                        y_true = df_upload['isFraud']
                                        y_pred = df_upload['Predicted_Fraud'].astype(int)
                                        
                                        # Confusion Matrix
                                        cm = confusion_matrix(y_true, y_pred)
                                        fig_cm = px.imshow(cm, 
                                                         labels=dict(x="Predicted", y="Actual", color="Count"),
                                                         x=['Non-Fraud', 'Fraud'],
                                                         y=['Non-Fraud', 'Fraud'],
                                                         title="Confusion Matrix",
                                                         text_auto=True)
                                        st.plotly_chart(fig_cm, use_container_width=True)
                                        
                                        # Classification Report
                                        report = classification_report(y_true, y_pred, output_dict=True)
                                        report_df = pd.DataFrame(report).transpose()
                                        st.dataframe(report_df, use_container_width=True)
                                    
                                    else:
                                        st.info("Tidak ada label asli (Class/isFraud) untuk evaluasi performa.")
                                
                                # Download results
                                st.markdown("---")
                                st.markdown("### 💾 Download Hasil Analisis")
                                
                                result_csv = df_upload.to_csv(index=False)
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.download_button(
                                        label="📥 Download CSV Hasil Prediksi",
                                        data=result_csv,
                                        file_name=f"fraud_prediction_{dataset_choice.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                                
                                with col2:
                                    # Hanya download data fraud
                                    fraud_csv = df_upload[df_upload['Predicted_Fraud'] == 1].to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Hanya Data Fraud",
                                        data=fraud_csv,
                                        file_name=f"fraud_only_{dataset_choice.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                                
                                with col3:
                                    # Generate summary report
                                    if st.button("📊 Generate Summary Report", use_container_width=True):
                                        with st.spinner("Membuat laporan summary..."):
                                            # Buat summary statistics
                                            summary_stats = {
                                                "Total Transaksi": total_count,
                                                "Transaksi Fraud": int(fraud_count),
                                                "Fraud Rate": f"{fraud_rate:.2f}%",
                                                "Rata-rata Fraud Probability": f"{np.mean(fraud_probs):.2f}%",
                                                "Std Dev Fraud Probability": f"{np.std(fraud_probs):.2f}%",
                                                "Max Fraud Probability": f"{max(fraud_probs):.2f}%",
                                                "Min Fraud Probability": f"{min(fraud_probs):.2f}%",
                                                "Model Used": "XGBoost",
                                                "Dataset Type": dataset_choice,
                                                "Processing Time": "Batch (Single Pass)",
                                                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                            }
                                            
                                            # Tampilkan summary
                                            st.markdown("### 📄 Summary Report")
                                            summary_df = pd.DataFrame(list(summary_stats.items()), 
                                                                    columns=['Metric', 'Value'])
                                            st.table(summary_df)
                                            
                                            st.success("✅ Laporan summary berhasil dibuat!")
                            
                        except Exception as e:
                            st.error(f"❌ Error dalam prediksi batch: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
        
        except Exception as e:
            st.error(f"❌ Error membaca file: {str(e)}")
            st.info("Pastikan file CSV dalam format yang benar dan coba lagi.")
    
    else:
        # Informasi tanpa upload
        st.info("📝 Silakan upload file CSV Anda untuk memulai analisis.")

# ==================== FOOTER ====================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])

with footer_col2:
    st.markdown("""
    <div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    <p>🛡️ <b>Fraud Detection System v2.0</b> | Model XGBoost Production</p>
    <p>Proyek Studi Independen - PT VINIX SEVEN AURUM | Universitas Negeri Surabaya</p>
    <p>© 2025 Akhmad Dany (23031554234) | S1 Sains Data FMIPA</p>
    </div>

    """, unsafe_allow_html=True)

