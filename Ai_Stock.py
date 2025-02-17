## NOTE: Set yfinance to the following version to get chart working: "pip install yfinance==0.2.40"

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ollama
import tempfile
import base64
import os

# Set up Streamlit app
st.set_page_config(layout="wide")
st.title("AI-Powered Technical Stock Analysis Dashboard")
st.sidebar.header("Configuration")

# Input for stock ticker and date range
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-14"))

# Fetch stock data
if st.sidebar.button("Fetch Data"):
    try:
        yf_data = yf.download(ticker, start=start_date, end=end_date)
        data = pd.DataFrame(yf_data)  # Create a new DataFrame

        # Flatten the DataFrame
        data.columns = data.columns.get_level_values(0)  # Remove MultiIndex
        data = data.reset_index()  # Reset index to make 'Date' a column

        st.session_state["stock_data"] = data  # Store the data in session state
        st.success("Stock data loaded successfully!")
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()  # Stop execution if data fetching fails

# Check if data is available
if "stock_data" in st.session_state:
    data = st.session_state["stock_data"]

    # Check if the DataFrame is empty or contains NaN values
    if data is not None and not data.empty and not data.isnull().values.any():

        st.write(f"Type of data['Open']: {type(data['Open'])}")  # Check type with data.loc
        st.write(f"Type of data: {type(data)}")  # Check type of data

        st.write("First 5 rows of data:")  # Print the first few rows
        st.write(data.head())

        st.write("Data types of each column:")  # Print the data types of each column
        st.write(data.dtypes)

        # Convert data types to numeric (handle potential errors)
        try:
            data['Open'] = pd.to_numeric(data['Open'])  # Use data.loc
            data['High'] = pd.to_numeric(data['High'])  # Use data.loc
            data['Low'] = pd.to_numeric(data['Low'])  # Use data.loc
            data['Close'] = pd.to_numeric(data['Close'])  # Use data.loc
            data['Volume'] = pd.to_numeric(data['Volume'])  # Use data.loc
        except ValueError as e:
            st.error(f"Error converting data to numeric: {e}")
            st.stop()

        # Plot candlestick chart
        fig = go.Figure(data=[
            go.Candlestick(
                x=data['Date'],  # Use 'Date' column for x-axis
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Candlestick"
            )
        ])

        fig.update_layout(
            title=f"Candlestick Chart for {ticker}",
            xaxis_title="Date",
            yaxis_title="Price",
        )

        st.plotly_chart(fig)

        # Sidebar: Select technical indicators
        st.sidebar.subheader("Technical Indicators")
        indicators = st.sidebar.multiselect(
            "Select Indicators:",
            ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"],
            default=["20-Day SMA"]
        )

        # Helper function to add indicators to the chart
        def add_indicator(indicator, data, fig):  # Pass data and fig as arguments
            try:  # Add a try-except block to handle potential errors within the function
                if indicator == "20-Day SMA":
                    sma = data['Close'].rolling(window=20).mean()
                    fig.add_trace(go.Scatter(x=data['Date'], y=sma, mode='lines', name='SMA (20)'))
                elif indicator == "20-Day EMA":
                    ema = data['Close'].ewm(span=20).mean()
                    fig.add_trace(go.Scatter(x=data['Date'], y=ema, mode='lines', name='EMA (20)'))
                elif indicator == "20-Day Bollinger Bands":
                    sma = data['Close'].rolling(window=20).mean()
                    std = data['Close'].rolling(window=20).std()
                    bb_upper = sma + 2 * std
                    bb_lower = sma - 2 * std
                    fig.add_trace(go.Scatter(x=data['Date'], y=bb_upper, mode='lines', name='BB Upper'))
                    fig.add_trace(go.Scatter(x=data['Date'], y=bb_lower, mode='lines', name='BB Lower'))
                elif indicator == "VWAP":
                    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
                    fig.add_trace(go.Scatter(x=data['Date'], y=data['VWAP'], mode='lines', name='VWAP'))
            except Exception as e:
                st.error(f"Error adding indicator {indicator}: {e}")  # Display error message

        # Add selected indicators to the chart
        for indicator in indicators:
            add_indicator(indicator, data, fig)  # Pass data and fig to the function

        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig)

        # Analyze chart with LLaMA 3.2 Vision
        st.subheader("AI-Powered Analysis")
        if st.button("Run AI Analysis"):
            with st.spinner("Analyzing the chart, please wait..."):
                # Save chart as a temporary image
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                    fig.write_image(tmpfile.name)
                    tmpfile_path = tmpfile.name

            # Read image and encode to Base64
            with open(tmpfile_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            # Prepare AI analysis request
            messages = [{
                'role': 'user',
                'content': """You are a Stock Trader specializing in Technical Analysis at a top financial institution.
                            Analyze the stock chart's technical indicators and provide a buy/hold/sell recommendation.
                            Base your recommendation only on the candlestick chart and the displayed technical indicators.
                            First, provide the recommendation, then, provide your detailed reasoning.
                """,
                'images': [image_data]
            }]
            response = ollama.chat(model='llama3.2-vision', messages=messages)

            # Display AI analysis result
            st.write("**AI Analysis Results:**")
            st.write(response["message"]["content"])

            # Clean up temporary file
            os.remove(tmpfile_path)

    else:
        st.warning("No data found for the specified ticker and date range, or data contains NaN values. Please check your inputs.")
        st.info("Click 'Fetch Data' to load stock data.")
else:
    st.info("Click 'Fetch Data' to load stock data.")
