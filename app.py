# app.py (versi untuk Random Forest dengan Fitur Penting)

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# --- Fungsi untuk memuat semua objek ---
# Menggunakan cache agar tidak perlu load ulang setiap kali ada interaksi
@st.cache_data
def load_objects():
    """Memuat semua model dan objek preprocessing yang diperlukan."""
    # Tentukan path ke file-file yang dibutuhkan
    paths = {
        "model": "model_rf_final.pkl",
        "imputer": "imputer.pkl",
        "encoder": "label_encoder.pkl",
        "all_features": "feature_names.pkl",
        "important_features": "important_features.pkl", # BARU
        "feature_means": "feature_means.pkl"         # BARU
    }

    # Periksa apakah semua file ada
    if not all(os.path.exists(p) for p in paths.values()):
        return None

    # Muat semua objek
    loaded_objects = {key: joblib.load(path) for key, path in paths.items()}
    return loaded_objects

# Muat semua objek saat aplikasi dimulai
loaded_objects = load_objects()

# --- Konfigurasi Halaman dan Tampilan Utama ---
st.set_page_config(page_title="Prediksi Topik Skripsi", layout="wide")

# Tampilkan error jika file model tidak ditemukan
if loaded_objects is None:
    st.error(
        "❌ File model tidak ditemukan! Pastikan file-file berikut ada di direktori yang sama dengan `app.py`:\n"
        "- `model_rf_final.pkl`\n"
        "- `imputer.pkl`\n"
        "- `label_encoder.pkl`\n"
        "- `feature_names.pkl`\n"
        "- `important_features.pkl` (BARU)\n"
        "- `feature_means.pkl` (BARU)"
    )
    st.stop()

st.title("🎓 Aplikasi Prediksi Topik Skripsi")
st.write(
    "Aplikasi ini menggunakan model **Random Forest** untuk memberikan rekomendasi "
    "topik skripsi berdasarkan **nilai mata kuliah paling berpengaruh**."
)
st.markdown("---")


# --- Form Input Nilai ---
with st.form("prediction_form"):
    st.subheader("📝 Masukkan Nilai Mata Kuliah (Skala 0-100)")
    
    # Ambil daftar fitur penting dari objek yang dimuat
    important_features = loaded_objects['important_features']
    
    # Buat kolom agar form lebih rapi
    num_cols = 3
    cols = st.columns(num_cols)
    
    # Dictionary untuk menampung nilai input dari pengguna
    user_input_data = {}

    # Buat input field hanya untuk fitur yang penting
    for i, feature in enumerate(important_features):
        with cols[i % num_cols]:
            user_input_data[feature] = st.number_input(
                label=f"{feature}", 
                min_value=0.0, 
                max_value=100.0, 
                value=75.0,  # Nilai default
                step=1.0,
                key=feature
            )
    
    # Tombol submit form
    submitted = st.form_submit_button("🚀 Prediksi Topik Saya")

# --- Logika Prediksi dan Tampilan Hasil ---
if submitted:
    try:
        # Ambil daftar semua fitur dan nilai rata-ratanya
        all_features = loaded_objects['all_features']
        feature_means = loaded_objects['feature_means']

        # 1. BUAT VEKTOR FITUR LENGKAP (LANGKAH KUNCI)
        # Mulai dengan nilai rata-rata untuk semua fitur
        full_feature_vector = feature_means.copy()
        # Timpa nilai rata-rata dengan input dari pengguna untuk fitur-fitur penting
        for feature, value in user_input_data.items():
            full_feature_vector[feature] = value
            
        # Urutkan vektor sesuai urutan asli yang digunakan saat training
        input_list = [full_feature_vector[feature] for feature in all_features]
        
        # 2. Konversi ke numpy array 2D
        input_array = np.array(input_list).reshape(1, -1)
        
        # 3. Terapkan imputasi (menggunakan .transform)
        imputed_input = loaded_objects['imputer'].transform(input_array)
        
        # 4. Lakukan prediksi dengan model
        model = loaded_objects['model']
        prediction_code = model.predict(imputed_input)
        prediction_proba = model.predict_proba(imputed_input)
        
        # 5. Ubah kode prediksi kembali ke label asli (nama topik)
        label_encoder = loaded_objects['label_encoder']
        prediction_label = label_encoder.inverse_transform(prediction_code)[0]
        confidence_score = prediction_proba[0][prediction_code[0]] * 100
        
        # Tampilkan hasil dengan gaya
        st.success(f"**Rekomendasi Topik Skripsi:**")
        st.subheader(f"🎯 {prediction_label}")
        st.info(f"**Tingkat Keyakinan Model:** {confidence_score:.2f}%")

        # Tampilkan detail probabilitas (opsional, tapi informatif)
        with st.expander("Lihat Detail Probabilitas untuk Semua Topik"):
            proba_df = pd.DataFrame(
                prediction_proba,
                columns=label_encoder.classes_,
                index=['Probabilitas']
            ).T * 100
            proba_df.rename(columns={'Probabilitas': 'Peluang (%)'}, inplace=True)
            st.dataframe(proba_df.style.format("{:.2f}%").highlight_max(axis=0, color='lightgreen'))

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
