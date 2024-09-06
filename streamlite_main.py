import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import altair as alt
import base64
import time

# Fungsi untuk memuat data dari file yang diunggah
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col="Date")  # Muat file CSV yang diunggah
        return df
    else:
        return None

# Fungsi untuk memberikan link download file CSV
def get_table_download_link(df):
    csv = df.to_csv(index=True)  # Pastikan kolom Date disertakan
    b64 = base64.b64encode(csv.encode()).decode()  # Encode dalam format base64
    href = f'<a href="data:file/csv;base64,{b64}" download="updated_data.csv">Download updated CSV file</a>'
    return href

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard-Prediction Use LSTM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

# Judul halaman dashboard
st.title("Prediction Dashboard")

# Bagian sidebar
with st.sidebar:
    st.title('📈 Dashboard-Prediction Use LSTM')

    # Opsi untuk mengunggah file CSV
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    # Model selection
    model_options = ['stokUsD4.h5', 'stokUsD2.h5', 'stokUsD1.h5', 'stoknew.h5']  # Replace with your model names
    selected_model = st.selectbox("Select a model", model_options)

    # Slider untuk pengaturan
    n_day = st.slider("Days of prediction :", 1, 30)
    sample = st.slider("Sample :", 1, 30)
    check_box = st.checkbox(label="Display Table of Prediction")

# Fungsi untuk melakukan prediksi
def prediction(uploaded_file, selected_model, n_day, sample):
    df = load_data(uploaded_file)
    if df is None:
        st.warning("Please upload a CSV file.")
        return

    data = df.filter(['Close'])
    dataset = data.values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Dapatkan jumlah data untuk pelatihan
    training_data_len = int(np.ceil(len(dataset) * .95))

    # Muat model yang dipilih
    model = load_model(selected_model)

    test_data = scaled_data[training_data_len - 60:, :]

    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Make predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Siapkan data latih dan valid
    train = data[:training_data_len]
    valid = data[training_data_len:].copy()
    valid['Predictions'] = predictions

    # Prediksi untuk hari berikutnya
    last_60_days = scaled_data[-60:]
    for day in range(n_day):
        x_future = np.array([last_60_days])
        x_future = np.reshape(x_future, (x_future.shape[0], x_future.shape[1], 1))

        # Prediksi untuk hari berikutnya
        predicted_value = model.predict(x_future)
        predicted_value = scaler.inverse_transform(predicted_value)

        # Tambahkan prediksi ke dataset valid sebagai prediksi masa depan
        next_date = valid.index[-1] + pd.DateOffset(1)  # Tambah 1 hari
        valid.loc[next_date] = [np.nan, predicted_value[0][0]]  # Append predicted value

        # Update last_60_days dengan prediksi terbaru
        new_value_scaled = scaler.transform(predicted_value)
        last_60_days = np.append(last_60_days[1:], new_value_scaled, axis=0)  # Shift window

    # Plot hasil
    fig = go.Figure()

    # Plot data latih
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train Close', line=dict(color='blue')))

    # Plot data valid
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Valid Close', line=dict(color='red')))

    # Plot prediksi
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Predictions', line=dict(color='green')))

    # Update layout
    fig.update_layout(title='Stock Price Prediction',
                      width=1000,
                      height=500,
                      margin=dict(l=20, r=20, t=40, b=20),  # left, right, top, bottom
                      font=dict(family="Courier New, monospace", size=18, color="RebeccaPurple"),
                      xaxis_title='Date',
                      yaxis_title='Value',
                      legend_title='Legend')

    # Tampilkan grafik di Streamlit
    st.plotly_chart(fig)

    # Tampilkan tabel hasil prediksi
    if check_box:
        st.dataframe(valid)
        st.markdown(get_table_download_link(valid), unsafe_allow_html=True)
    else:
        st.dataframe(df, width=1000, height=500)
        st.markdown(get_table_download_link(df), unsafe_allow_html=True)

# Panggil fungsi prediksi
prediction(uploaded_file, selected_model, n_day, sample)
