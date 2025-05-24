import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import plotly.graph_objs as go
from prophet import Prophet
import numpy as np
import gradio as gr
import pmdarima as pm

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")

os.environ["LANGSMITH_TRACING"] = LANGSMITH_TRACING
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY

embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=GOOGLE_API_KEY
)
vectorstore = Chroma(
    persist_directory="persist_store", embedding_function=embedding
)
retriever = vectorstore.as_retriever()

llm = GoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
chain = load_qa_chain(llm=llm, chain_type="stuff")


def answer_financial_query(company_name: str, user_question: str) -> str:
    query = (
        f"Provide financial data, stock prices, earnings, and recent performance for {company_name}. "
        f"User question: {user_question}"
    )
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return f"Sorry, no financial information found related to your query about {company_name}."
    return chain.run(input_documents=docs, question=user_question)


def extract_time_series_from_docs(docs) -> pd.DataFrame:
    # Placeholder for future implementation:
    # Extract stock prices or earnings data from retrieved IR documents (PDF, HTML, etc.)
    return None


def generate_synthetic_stock_data(periods=365):
    today = datetime.today()
    dates = pd.date_range(end=today, periods=periods)
    prices = 300 + (np.arange(periods) * 0.1) + (5 * np.random.randn(periods))
    return pd.DataFrame({'ds': dates, 'y': prices})


def forecast_with_prophet(df: pd.DataFrame, forecast_period: int = 90):
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=forecast_period)
    forecast = model.predict(future)
    return forecast


def forecast_with_arima(df: pd.DataFrame, forecast_period: int = 90):
    df_arima = df.copy()
    df_arima.set_index("ds", inplace=True)
    model = pm.auto_arima(df_arima['y'], seasonal=False, stepwise=True, suppress_warnings=True)
    forecast = model.predict(n_periods=forecast_period)
    future_dates = pd.date_range(start=df["ds"].iloc[-1] + pd.Timedelta(days=1), periods=forecast_period)
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast})
    full_df = pd.concat([df[['ds', 'y']], forecast_df], ignore_index=True)
    return full_df


def create_stock_forecast_plot(df: pd.DataFrame, forecast: pd.DataFrame, company_name: str):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["ds"], y=df["y"], mode='lines', name="Historical", line=dict(color="blue")
    ))

    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"], mode='lines', name="Forecast", line=dict(color="green")
    ))

    # Prophet confidence intervals if available
    if "yhat_lower" in forecast.columns and "yhat_upper" in forecast.columns:
        fig.add_trace(go.Scatter(
            x=forecast["ds"], y=forecast["yhat_upper"],
            mode='lines', name="Upper Bound", line=dict(width=0), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast["ds"], y=forecast["yhat_lower"],
            mode='lines', name="Lower Bound", fill='tonexty', line=dict(width=0),
            fillcolor='rgba(0,255,0,0.2)', showlegend=True
        ))

    fig.update_layout(
        title=f"{company_name} Stock Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(x=0, y=1),
        hovermode="x unified"
    )
    return fig


def generate_custom_forecast_explanation(company_name: str, forecast: pd.DataFrame, prompt: str, forecast_period: int) -> str:
    recent = forecast.tail(forecast_period)[["ds", "yhat"]].copy()
    recent["ds"] = recent["ds"].dt.strftime('%Y-%m-%d')
    recent_str = recent.to_string(index=False)

    full_prompt = (
        f"Here is the {forecast_period}-day forecast for {company_name}:\n{recent_str}\n\n"
        f"Now, using this data, respond to the following explanation prompt:\n{prompt}"
    )
    
    response = llm.generate([full_prompt]).generations[0][0].text
    return response.strip()


def analyze_and_forecast_stock(company_name: str, user_question: str, explanation_prompt: str, model_choice: str):
    summary = answer_financial_query(company_name, user_question)
    docs = retriever.get_relevant_documents(f"Historical stock prices and earnings for {company_name}")
    df = extract_time_series_from_docs(docs)
    if df is None or df.empty:
        df = generate_synthetic_stock_data()

    forecast_period = 90  # you can change this default or make it a user input
    if model_choice == "Prophet":
        forecast = forecast_with_prophet(df, forecast_period)
    elif model_choice == "ARIMA":
        forecast = forecast_with_arima(df, forecast_period)
    else:
        raise ValueError("Unsupported model type")

    # Normalize yhat
    if "yhat" not in forecast.columns:
        forecast["yhat"] = forecast.get("y")

    plot = create_stock_forecast_plot(df, forecast, company_name)
    explanation = generate_custom_forecast_explanation(company_name, forecast, explanation_prompt, forecast_period)
    return summary, plot, explanation


# Gradio Interface
iface = gr.Interface(
    fn=analyze_and_forecast_stock,
    inputs=[
        gr.Textbox(label="Company Name", placeholder="e.g., Microsoft, Google, Nvidia"),
        gr.Textbox(label="Financial Query", placeholder="e.g., Analyze Microsoft’s stock performance over the past year and forecast its performance next quarter."),
        gr.Textbox(label="Forecast Explanation Prompt", placeholder="e.g., Explain this forecast focusing on market risks and opportunities.", lines=3),
        gr.Dropdown(choices=["Prophet", "ARIMA"], label="Forecasting Model", value="Prophet")
    ],
    outputs=[
        gr.Textbox(label="Financial Summary and Insights"),
        gr.Plot(label="Stock Price Forecast"),
        gr.Textbox(label="Forecast Explanation")
    ],
    title="📈 Financial Analytics & Forecasting (Prophet & ARIMA)",
    description="Conduct advanced market analytics, generate forecasts with Prophet or ARIMA, visualize trends, and receive AI-generated narrative insights."
)

if __name__ == "__main__":
    iface.launch()
