# -*- coding: utf-8 -*-
"""Website Prediksi Harga Saham dengan GRU (Versi Lengkap)"""

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

STOCKS = {
    'ADHI': {'model': 'Model/GRU_MODEL_ADHI.h5', 'scaler': 'Scaler/GRU_SCALER_ADHI.save'},
    'AGRO': {'model': 'Model/GRU_MODEL_AGRO.h5', 'scaler': 'Scaler/GRU_SCALER_AGRO.save'},
    # Tambahkan saham lainnya sesuai kebutuhan
}

# ==============================================
# FUNGSI UTILITAS
# ==============================================

def generate_synthetic_sequence(base_price, sentiment, days=60, volatility=0.02):
    np.random.seed(42)
    prices = [base_price]
    trend_factor = 1 + (sentiment * 0.003)
    for _ in range(days-1):
        change = np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + change) * trend_factor
        new_price = max(base_price * 0.97, min(base_price * 1.03, new_price))
        prices.append(new_price)
    return np.array(prices)

def validate_inputs(open_p, high_p, low_p, close_p, volume):
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
# FUNGSI PREDIKSI INTI
# ==============================================

def enhanced_predict(stock_code, input_data, months=12):
    try:
        model = load_model(STOCKS[stock_code]['model'])
        scaler = joblib.load(STOCKS[stock_code]['scaler'])

        close_price = input_data[3]
        sentiment = input_data[6]
        days_per_month = 30
        n_steps = 60

        synthetic_prices = generate_synthetic_sequence(close_price, sentiment, n_steps)
        sequence = []
        for i in range(n_steps):
            open_p = synthetic_prices[i] * (0.99 + np.random.uniform(-0.01, 0.01))
            high_p = synthetic_prices[i] * (1.01 + np.random.uniform(-0.01, 0.01))
            low_p = synthetic_prices[i] * (0.98 + np.random.uniform(-0.01, 0.01))
            sequence.append([
                open_p, high_p, low_p, synthetic_prices[i],
                input_data[4], input_data[5], input_data[6]
            ])
        sequence = np.array(sequence)
        predictions = []

        for month in range(months):
            scaled_seq = scaler.transform(sequence)
            input_seq = scaled_seq.reshape(1, n_steps, 7)
            monthly_preds = []
            for day in range(days_per_month):
                pred = model.predict(input_seq, verbose=0)[0,0]
                monthly_preds.append(pred)
                new_seq = np.roll(input_seq[0], shift=-1, axis=0)
                new_seq[-1, 3] = pred
                input_seq = new_seq.reshape(1, n_steps, 7)
            final_pred = monthly_preds[-1]
            dummy = np.zeros((1,7))
            dummy[0,3] = final_pred
            pred_price = scaler.inverse_transform(dummy)[0,3]

            if month > 0:
                pred_price = (predictions[month-1] + pred_price) / 2
                if month > 2 and pred_price < predictions[month-1]:
                    if np.random.rand() > 0.7:
                        adjustment = 1 + abs(np.random.normal(0, 0.015))
                        pred_price *= adjustment

            predictions.append(pred_price)

            sequence = np.roll(sequence, shift=-days_per_month, axis=0)
            for i in range(1, days_per_month+1):
                change = np.random.normal(0, input_data[5]/1000) * (1 + sentiment*0.5)
                sequence[-i, 3] = pred_price * (1 + change)
            for i in range(n_steps):
                sequence[i, 0] = sequence[i, 3] * (0.99 + np.random.uniform(-0.01, 0.01))
                sequence[i, 1] = sequence[i, 3] * (1.01 + np.random.uniform(-0.01, 0.01))
                sequence[i, 2] = sequence[i, 3] * (0.98 + np.random.uniform(-0.01, 0.01))

        return predictions
    except Exception as e:
        st.error(f"Error dalam prediksi: {str(e)}")
        return None

# ==============================================
# STREAMLIT UI
# ==============================================

def show_intro():
    st.markdown("""
    ### Selamat Datang di Website Prediksi Saham IDXBUMN20
    Website ini menggunakan model **GRU (Gated Recurrent Unit)** untuk memprediksi harga saham bulanan
    berdasarkan data terakhir dan sentimen pasar.
    """)

def main():
    show_intro()
    selected_stock = st.selectbox("Pilih Kode Saham", list(STOCKS.keys()))
    open_price = st.number_input("Harga Open", min_value=0.0)
    high_price = st.number_input("Harga High", min_value=0.0)
    low_price = st.number_input("Harga Low", min_value=0.0)
    close_price = st.number_input("Harga Close", min_value=0.0)
    volume = st.number_input("Volume", min_value=0.0)
    volatility = st.slider("Volatilitas (0.01 - 1.0)", 0.01, 1.0, 0.05)
    sentiment = st.slider("Sentimen Pasar (-1 negatif, +1 positif)", -1.0, 1.0, 0.0)
    months = st.slider("Jumlah Bulan Prediksi", 1, 24, 12)

    if st.button("Prediksi Harga"):
        errors = validate_inputs(open_price, high_price, low_price, close_price, volume)
        if errors:
            for err in errors:
                st.warning(err)
        else:
            input_data = [open_price, high_price, low_price, close_price, volume, volatility, sentiment]
            hasil = enhanced_predict(selected_stock, input_data, months)
            if hasil:
                df_hasil = pd.DataFrame({"Bulan ke-":[i+1 for i in range(months)], "Harga Prediksi": hasil})
                st.line_chart(df_hasil.set_index("Bulan ke-"))
                st.dataframe(df_hasil)

if __name__ == "__main__":
    main()
