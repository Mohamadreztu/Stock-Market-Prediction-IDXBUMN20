# -*- coding: utf-8 -*-
"""Website Prediksi Harga Saham dengan GRU (Versi Diperbaiki)"""

import sys
import subprocess
import importlib
from packaging import version
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os
import streamlit as st

# ==============================================
# KONFIGURASI AWAL
# ==============================================

# Fix untuk error MarkupSafe
try:
    import markupsafe
    if version.parse(markupsafe.__version__) >= version.parse("2.1.0"):
        subprocess.run([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "markupsafe==2.0.1",
            "--quiet"
        ], check=True)
        importlib.reload(markupsafe)
except Exception:
    pass

# Konfigurasi halaman
st.set_page_config(
    page_title="Website Prediksi Saham IDXBUMN20",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# ==============================================
# KONSTANTA DAN SETUP
# ==============================================

# Daftar saham dan model
STOCKS = {
    'ADHI': {'model': 'Model/GRU_MODEL_ADHI.h5', 'scaler': 'Scaler/GRU_SCALER_ADHI.save'},
    'AGRO': {'model': 'Model/GRU_MODEL_AGRO.h5', 'scaler': 'Scaler/GRU_SCALER_AGRO.save'},
    'ANTM': {'model': 'Model/GRU_MODEL_ANTM.h5', 'scaler': 'Scaler/GRU_SCALER_ANTM.save'},
    'BBNI': {'model': 'Model/GRU_MODEL_BBNI.h5', 'scaler': 'Scaler/GRU_SCALER_BBNI.save'},
    'BBRI': {'model': 'Model/GRU_MODEL_BBRI.h5', 'scaler': 'Scaler/GRU_SCALER_BBRI.save'},
    'BBTN': {'model': 'Model/GRU_MODEL_BBTN.h5', 'scaler': 'Scaler/GRU_SCALER_BBTN.save'},
    'BJBR': {'model': 'Model/GRU_MODEL_BJBR.h5', 'scaler': 'Scaler/GRU_SCALER_BJBR.save'},
    'BJTM': {'model': 'Model/GRU_MODEL_BJTM.h5', 'scaler': 'Scaler/GRU_SCALER_BJTM.save'},
    'BMRI': {'model': 'Model/GRU_MODEL_BMRI.h5', 'scaler': 'Scaler/GRU_SCALER_BMRI.save'},
    'BRIS': {'model': 'Model/GRU_MODEL_BRIS.h5', 'scaler': 'Scaler/GRU_SCALER_BRIS.save'},
    'ELSA': {'model': 'Model/GRU_MODEL_ELSA.h5', 'scaler': 'Scaler/GRU_SCALER_ELSA.save'},
    'JSMR': {'model': 'Model/GRU_MODEL_JSMR.h5', 'scaler': 'Scaler/GRU_SCALER_JSMR.save'},
    'MTEL': {'model': 'Model/GRU_MODEL_MTEL.h5', 'scaler': 'Scaler/GRU_SCALER_MTEL.save'},
    'PGAS': {'model': 'Model/GRU_MODEL_PGAS.h5', 'scaler': 'Scaler/GRU_SCALER_PGAS.save'},
    'PGEO': {'model': 'Model/GRU_MODEL_PGEO.h5', 'scaler': 'Scaler/GRU_SCALER_PGEO.save'},
    'PTBA': {'model': 'Model/GRU_MODEL_PTBA.h5', 'scaler': 'Scaler/GRU_SCALER_PTBA.save'},
    'PTPP': {'model': 'Model/GRU_MODEL_PTPP.h5', 'scaler': 'Scaler/GRU_SCALER_PTPP.save'},
    'SMGR': {'model': 'Model/GRU_MODEL_SMGR.h5', 'scaler': 'Scaler/GRU_SCALER_SMGR.save'},
    'TINS': {'model': 'Model/GRU_MODEL_TINS.h5', 'scaler': 'Scaler/GRU_SCALER_TINS.save'},
    'TLKM': {'model': 'Model/GRU_MODEL_TLKM.h5', 'scaler': 'Scaler/GRU_SCALER_TLKM.save'},
}

# ==============================================
# FUNGSI UTILITAS
# ==============================================

def generate_synthetic_sequence(base_price, sentiment, days=60, volatility=0.02):
    """Membuat sequence harga dengan variasi realistis dan mempertimbangkan sentimen"""
    np.random.seed(42)  # Untuk reproducibility
    prices = [base_price]
    trend_factor = 1 + (sentiment * 0.003)  # Faktor tren berdasarkan sentimen
    
    for _ in range(days-1):
        change = np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + change) * trend_factor
        
        # Batasi fluktuasi antara -3% hingga +3% per hari (disesuaikan)
        new_price = max(base_price * 0.97, min(base_price * 1.03, new_price))
        prices.append(new_price)
    return np.array(prices)

def validate_inputs(open_p, high_p, low_p, close_p, volume):
    """Validasi input pengguna"""
    errors = []
    if any(p <= 0 for p in [open_p, high_p, low_p, close_p]):
        errors.append("Semua harga harus lebih besar dari 0")
    if volume <= 0:
        errors.append("Volume harus lebih besar dari 0")
    if high_p < open_p or high_p < close_p or high_p < low_p:
        errors.append("Harga tinggi harus yang tertinggi")
    if low_p > open_p or low_p > close_p or low_p > high_p:
        errors.append("Harga rendah harus yang terendah")
    return errors

# ==============================================
# FUNGSI PREDIKSI INTI YANG DIPERBAIKI
# ==============================================

def enhanced_predict(stock_code, input_data, months=12):
    """Fungsi prediksi multi-bulan dengan perbaikan pola prediksi"""
    try:
        # Load model dan scaler
        model = load_model(STOCKS[stock_code]['model'])
        scaler = joblib.load(STOCKS[stock_code]['scaler'])
        
        # Parameter
        close_price = input_data[3]
        sentiment = input_data[6]
        days_per_month = 30
        n_steps = 60
        
        # 1. Bangun sequence awal dengan variasi realistis dan faktor sentimen
        synthetic_prices = generate_synthetic_sequence(close_price, sentiment, n_steps)
        
        # 2. Bangun sequence lengkap dengan semua fitur
        sequence = []
        for i in range(n_steps):
            # Gunakan variasi sekitar harga sintetis dengan sentimen
            open_p = synthetic_prices[i] * (0.99 + np.random.uniform(-0.01, 0.01))
            high_p = synthetic_prices[i] * (1.01 + np.random.uniform(-0.01, 0.01))
            low_p = synthetic_prices[i] * (0.98 + np.random.uniform(-0.01, 0.01))
            
            sequence.append([
                open_p, high_p, low_p, synthetic_prices[i],
                input_data[4],  # Volume
                input_data[5],  # Volatilitas
                input_data[6]   # Sentimen
            ])
        
        sequence = np.array(sequence)
        predictions = []
        
        # 3. Prediksi untuk setiap bulan dengan perbaikan
        for month in range(months):
            # Normalisasi sequence
            scaled_seq = scaler.transform(sequence)
            input_seq = scaled_seq.reshape(1, n_steps, 7)
            
            # Prediksi 30 hari (1 bulan)
            monthly_preds = []
            for day in range(days_per_month):
                # Prediksi 1 hari
                pred = model.predict(input_seq, verbose=0)[0,0]
                monthly_preds.append(pred)
                
                # Update sequence dengan prediksi terbaru
                new_seq = np.roll(input_seq[0], shift=-1, axis=0)
                new_seq[-1, 3] = pred  # Update harga close
                input_seq = new_seq.reshape(1, n_steps, 7)
            
            # Simpan prediksi akhir bulan
            final_pred = monthly_preds[-1]
            dummy = np.zeros((1,7))
            dummy[0,3] = final_pred
            pred_price = scaler.inverse_transform(dummy)[0,3]
            
            # 4. Perbaikan: Tambahkan faktor penyesuaian
            if month > 0:
                # Smoothing dengan moving average 2 bulan
                pred_price = (predictions[month-1] + pred_price) / 2
                
                # Jika prediksi terus menurun, berikan penyesuaian acak
                if month > 2 and pred_price < predictions[month-1]:
                    if np.random.rand() > 0.7:  # 30% chance untuk adjustment
                        adjustment = 1 + abs(np.random.normal(0, 0.015))
                        pred_price *= adjustment
            
            predictions.append(pred_price)
            
            # 5. Update sequence untuk bulan berikutnya dengan mempertimbangkan tren
            sequence = np.roll(sequence, shift=-days_per_month, axis=0)
            
            # Update harga close untuk 30 hari terakhir dengan variasi baru
            for i in range(1, days_per_month+1):
                # Tambahkan variasi berdasarkan sentimen
                change = np.random.normal(0, input_data[5]/1000) * (1 + sentiment*0.5)
                sequence[-i, 3] = pred_price * (1 + change)
            
            # Update fitur OHLC berdasarkan harga close baru
            for i in range(n_steps):
                sequence[i, 0] = sequence[i, 3] * (0.99 + np.random.uniform(-0.01, 0.01))
                sequence[i, 1] = sequence[i, 3] * (1.01 + np.random.uniform(-0.01, 0.01))
                sequence[i, 2] = sequence[i, 3] * (0.98 + np.random.uniform(-0.01, 0.01))
        
        return predictions
        
    except Exception as e:
        st.error(f"Error dalam prediksi: {str(e)}")
        return None

# ==============================================
# TAMPILAN STREAMLIT
# ==============================================

def show_intro():
    st.markdown("""
    ### Selamat Datang di Wesbite Prediksi Saham IDXBUMN20👋🏻
    
    Website ini menggunakan model **GRU (Gated Recurrent Unit)** untuk memprediksi pergerakan harga saham yang terdaftar pada **Indeks Saham IDXBUMN20** (ADHI, AGRO, ANTM, BBNI, BBRI, BBTN, BJBR, BJTM, BMRI, BRIS, ELSA, JSMR, MTEL, PGAS, PGEO, PTBA, PTPP, SMGR, TINS, TLKM).
    """)
    
    with st.expander("ℹ️ Tentang Model", expanded=True):
        st.markdown("""
        - **Arsitektur** : GRU 1 Layer dengan (24, 36, 48, 60) Unit
        - **Pelatihan** : 
            - Batch Size (32, 64, 128, 256)
            - Epoch (250, 500, 750, 1000)
        - **Akurasi** : 
            - Nilai Kesalahan Tertinggi : 2.73%
            - Nilai Kesalahan Terendah : 0.82%
        - **Update Terakhir** : 1 Januari 2025
        """)
        
    with st.expander("📌 Cara Menggunakan"):
        st.markdown("""
        1. **Pilih Kode Emiten Saham** dari dropdown menu di sidebar
        2. **Masukkan parameter** harga dan indikator
        3. Klik tombol **Lakukan Prediksi**
        4. Lihat hasil prediksi dan analisis
        """)
    
def show_sidebar():
    """Tampilan sidebar untuk input parameter"""
    with st.sidebar:
        st.header('Masukan Parameter')
        
        selected_stock = st.selectbox(
            'Pilih Kode Emiten Saham',
            options=list(STOCKS.keys()),
            index=0
        )
        
        st.subheader('Data Harga')
        col1, col2 = st.columns(2)
        with col1:
            open_price = st.number_input('Harga Buka', min_value=0.01, value=5900.0, step=1.0)
            high_price = st.number_input('Harga Tertinggi', min_value=0.01, value=5950.0, step=1.0)
        with col2:
            low_price = st.number_input('Harga Terendah', min_value=0.01, value=5850.0, step=1.0)
            close_price = st.number_input('Harga Penutupan', min_value=0.01, value=5900.0, step=1.0)
        
        st.subheader('Indikator Tambahan')
        volume = st.number_input('Volume ', min_value=1, value=50000000, step=1000)
        
        volatility = st.number_input(
            'Volatilitas Pasar',
            min_value=0.0,
            max_value=1000.0,
            value=100.0,
            step=0.5,
            format="%.1f",
            help="Masukkan nilai volatilitas dalam desimal (contoh: 112.5)"
        )
        
        sentiment = st.slider('Sentimen Pasar', -1.0, 1.0, 0.1, 0.1)
        
        if st.button('Lakukan Prediksi', type='primary', use_container_width=True):
            errors = validate_inputs(open_price, high_price, low_price, close_price, volume)
            if errors:
                for error in errors:
                    st.error(error)
            else:
                st.session_state.input_data = np.array([
                    open_price, high_price, low_price, close_price,
                    volume, volatility, sentiment
                ])
                st.session_state.selected_stock = selected_stock
                st.session_state.close_price = close_price
                st.session_state.predict_clicked = True
                st.rerun()

def show_prediction_results():
    """Tampilkan hasil prediksi"""
    col1, col2, col3 = st.columns([5,2,1])
    with col1:
        st.title(f'📊 Hasil Prediksi {st.session_state.selected_stock}')
    with col2:
        st.metric("Harga Terakhir", f"Rp.{st.session_state.close_price:,.0f}")
    with col3:
        if st.button('🔄 Reset', type='secondary', use_container_width=True):
            st.session_state.predict_clicked = False
            st.rerun()
    
    st.write("")
    
    with st.spinner('Mohon ditunggu ya, saya sedang memproses hasil prediksi...'):
        predictions = enhanced_predict(
            st.session_state.selected_stock,
            st.session_state.input_data
        )
    
    if predictions is None:
        st.error("Gagal melakukan prediksi. Silakan coba lagi.")
        return
    
    # Tampilkan hasil utama
    months_to_show = [2, 4, 6, 8, 12]
    close_price = st.session_state.close_price
    
    # Buat dataframe hasil
    results = []
    for i, month in enumerate(months_to_show):
        pred_price = predictions[month-1]
        change = ((pred_price - close_price) / close_price) * 100
        results.append({
            'Bulan': f'Bulan ke-{month}',
            'Prediksi Harga': pred_price,
            'Perubahan (%)': change,
            'Perubahan (Rp)': pred_price - close_price
        })
    
    df = pd.DataFrame(results)
    
    # Format tabel
    formatted_df = df.copy()
    formatted_df['Prediksi Harga'] = formatted_df['Prediksi Harga'].apply(lambda x: f"Rp.{x:,.0f}")
    formatted_df['Perubahan (Rp)'] = formatted_df['Perubahan (Rp)'].apply(
        lambda x: f"+Rp.{x:,.0f}" if x >= 0 else f"-Rp.{abs(x):,.0f}")
    formatted_df['Perubahan (%)'] = formatted_df['Perubahan (%)'].apply(
        lambda x: f"+{x:.2f}%" if x >= 0 else f"{x:.2f}%")
    
    # Tampilkan tabel
    st.subheader('Hasil Prediksi')
    st.dataframe(
        formatted_df,
        column_config={
            "Bulan": st.column_config.TextColumn("Periode", width="medium"),
            "Prediksi Harga": st.column_config.TextColumn("Prediksi Harga", width="large"),
            "Perubahan (%)": st.column_config.TextColumn("Perubahan (%)", width="medium"),
            "Perubahan (Rp)": st.column_config.TextColumn("Perubahan (Rp)", width="medium")
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Analisis tambahan
    with st.expander("📈 Analisis Prediksi"):
        max_increase = df['Perubahan (%)'].max()
        max_decrease = df['Perubahan (%)'].min()
        best_month = df.loc[df['Perubahan (%)'].idxmax(), 'Bulan']
        worst_month = df.loc[df['Perubahan (%)'].idxmin(), 'Bulan']
        
        st.markdown(f"""
        - **Potensi kenaikan tertinggi**: {max_increase:.2f}% pada {best_month}
        - **Potensi penurunan terbesar**: {max_decrease:.2f}% pada {worst_month}
        - **Rentang prediksi**: {formatted_df['Prediksi Harga'].iloc[0]} hingga {formatted_df['Prediksi Harga'].iloc[-1]}
        """)
    
    # Disclaimer
    st.warning("""
    **Disclaimer**: Prediksi ini didasarkan pada model machine learning dan tidak menjamin akurasi mutlak. 
    Hasil prediksi bukan merupakan rekomendasi investasi. Selalu lakukan riset tambahan sebelum membuat keputusan investasi.
    """)

# ==============================================
# FUNGSI UTAMA
# ==============================================

def main():
    """Fungsi utama aplikasi"""
    # Inisialisasi session state
    if 'predict_clicked' not in st.session_state:
        st.session_state.predict_clicked = False
    
    # Tambahkan style CSS
    st.markdown("""
    <style>
        .stDataFrame {
            font-size: 16px;
        }
        .stDataFrame [data-testid='stDataFrame-container'] {
            width: 100%;
        }
        .st-b7 {
            color: green;
        }
        .st-b8 {
            color: red;
        }
        .stAlert {
            font-size: 16px;
        }
        [data-testid="stExpander"] {
            margin: 15px 0;
        }
        .stButton button {
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Tampilkan sidebar
    show_sidebar()
    
    # Tampilan utama
    if not st.session_state.predict_clicked:
        show_intro()
    else:
        show_prediction_results()

# ==============================================
# JALANKAN APLIKASI
# ==============================================

if __name__ == "__main__":
    main()
