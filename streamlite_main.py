import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import altair as alt
import base64

def load_data(file):
    """Function for loading data"""
    if file.name.endswith('.csv'):
        df = pd.read_csv(file, index_col="Date")
    elif file.name.endswith('.xlsx'):
        df = pd.read_excel(file, index_col="Date")
    return df

def get_table_download_link(df):
    csv = df.to_csv(index=True)  # Ensure index (Date) is included in the output file
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="updated_data.csv">Download updated CSV file</a>'
    return href

st.set_page_config(
    page_title="Dashboard-Prediction Use LSTM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# Title of dashboard
st.title("Prediction Dashboard")

with st.sidebar:
    st.title('📈 Dashboard-Prediction Use LSTM')

    file_options = ['AAPL_stock_data.csv']
    uploaded_file = st.selectbox("Select a file", file_options)
    # Model selection
    model_options = ['stokUsD4.h5', 'stokUsD2.h5', 'stokUsD1.h5', 'stoknew.h5']  # Replace with your model names
    selected_model = st.selectbox("Select a model", model_options)

    n_day = st.slider("Days of prediction :", 1, 30)
    sample = st.slider("Sample :", 1, 30)
    check_box = st.checkbox(label="Display Table of Prediction")

def prediction(uploaded_file, selected_model, n_day, sample):
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        data = df.filter(['Close'])
        dataset = data.values

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        # Get the number of rows to train the model on
        training_data_len = int(np.ceil(len(dataset) * .95))

        # Load model
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

        # Prepare train and valid data
        train = data[:training_data_len]
        valid = data[training_data_len:].copy()
        valid['Predictions'] = predictions

        # Continue predicting for the future based on the predictions
        last_60_days = scaled_data[-60:]
        for day in range(n_day):
            x_future = np.array([last_60_days])
            x_future = np.reshape(x_future, (x_future.shape[0], x_future.shape[1], 1))

            # Make prediction for the next day
            predicted_value = model.predict(x_future)
            predicted_value = scaler.inverse_transform(predicted_value)

            # Append this value to the valid dataset as a future prediction
            next_date = valid.index[-1] + pd.DateOffset(1)  # Increment date by 1
            valid.loc[next_date] = [np.nan, predicted_value[0][0]]  # Append predicted value

            # Update last_60_days with the newly predicted value
            new_value_scaled = scaler.transform(predicted_value)
            last_60_days = np.append(last_60_days[1:], new_value_scaled, axis=0)  # Shift window

        # Append the new predictions to the original dataframe
        df = df.append(valid[['Predictions']])

        # Plot the results
        fig = go.Figure()

        # Plot train data
        fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train Close', line=dict(color='blue')))

        # Plot valid data
        fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Valid Close', line=dict(color='red')))

        # Plot predictions
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

        # Display plot in Streamlit
        st.plotly_chart(fig)

        # Display the updated table
        if check_box:
            st.dataframe(valid)
            st.markdown(get_table_download_link(valid), unsafe_allow_html=True)
        else:
            st.dataframe(df, width=1000, height=500)
            st.markdown(get_table_download_link(df), unsafe_allow_html=True) 

# Call prediction function
prediction(uploaded_file, selected_model, n_day, sample)
