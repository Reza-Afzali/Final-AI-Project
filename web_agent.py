# web_agent.py

import os
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from tavily_agent import tavily_search_with_date
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

gemini_model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

COMPANIES = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOG",
    "nvidia": "NVDA",
    "meta": "META"
}

SCRAPING_SOURCES = {
    "CNBC": {
        "base_url": "https://www.cnbc.com/search/?query={query}",
        "item_selector": "a.Card-title",
        "title_selector": "a.Card-title",
        "link_prefix": ""
    },
    "Yahoo": {
        "base_url": "https://finance.yahoo.com/quote/{query}/news",
        "item_selector": "li.js-stream-content h3 a",
        "title_selector": "li.js-stream-content h3 a",
        "link_prefix": "https://finance.yahoo.com"
    },
    "MarketWatch": {
        "base_url": "https://www.marketwatch.com/search?q={query}",
        "item_selector": "h3.article__headline a",
        "title_selector": "h3.article__headline a",
        "link_prefix": ""
    }
}

def generic_scrape(source_name, config, query, max_articles=3):
    headers = {"User-Agent": "Mozilla/5.0"}
    url = config["base_url"].format(query=query)
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        items = soup.select(config["item_selector"])[:max_articles]
        results = []
        for item in items:
            tag = item.select_one(config["title_selector"])
            if tag and tag.text.strip():
                title = tag.text.strip()
                link = tag.get("href", "#")
                full_link = link if link.startswith("http") else config["link_prefix"] + link
                results.append((None, f"[{source_name}] {title} ({full_link})"))
        return results if results else [(None, f"[{source_name}] Keine Ergebnisse.")]
    except Exception as e:
        return [(None, f"Fehler beim Scraping von {source_name}: {e}")]

def scrape_all_sources(query, max_articles=3):
    all_results = []
    for name, config in SCRAPING_SOURCES.items():
        all_results += generic_scrape(name, config, query, max_articles)
    return all_results

def get_latest_news(query, ticker, max_articles=5):
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return ["NEWS_API_KEY fehlt."]
    
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query, 
        "apiKey": api_key, 
        "language": "en",
        "sortBy": "publishedAt"
    }
    try:
        response = requests.get(url, params=params)
        articles = response.json().get("articles", [])
        results = []
        for a in articles[:max_articles]:
            title = a.get("title", "")
            description = a.get("description", "")
            source = a.get("source", {}).get("name", "")
            url = a.get("url", "")
            results.append(f"- {title}\n {description}\n ({source})\n {url}\n")
        return results if results else ["Keine relevanten Artikel gefunden."]    
    except Exception as e:
        return [f"Fehler bei NewsAPI: {e}"]

def get_stock_price_yahoo(ticker):
    try:
        ticker_data = yf.Ticker(ticker)
        hist = ticker_data.history(period="5d")
        if hist.empty:
            return f"Keine Daten für {ticker}."
        last = hist.iloc[-1]
        return f"Schlusskurs am {last.name.date()}: {round(last['Close'], 2)} USD"
    except Exception as e:
        return f"Yahoo Finance Fehler: {e}"

def get_alpha_vantage_data(symbol):
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return "ALPHA_VANTAGE_API_KEY fehlt."
    url = "https://www.alphavantage.co/query"
    params = {"function": "TIME_SERIES_DAILY", "symbol": symbol, "apikey": api_key}
    try:
        response = requests.get(url, params=params)
        data = response.json().get("Time Series (Daily)", {})
        if not data:
            return "Keine Daten von Alpha Vantage."
        latest_date = sorted(data.keys())[-1]
        close_price = data[latest_date]["4. close"]
        return f"{symbol} Schlusskurs am {latest_date}: {close_price} USD"
    except Exception as e:
        return f"Alpha Vantage Fehler: {e}"

def handle_company_news(query: str) -> str:
    lines = []
    ticker = None
    name = None
    for company_name, company_ticker in COMPANIES.items():     
        if company_name.lower() in query.lower():
            name = company_name
            ticker = company_ticker   
            break

    if not ticker:
        firms = ", ".join(c.capitalize() for c in COMPANIES.keys())
        return f"❌ Unbekannte Firma. Verfügbare Firmen: {firms}"

    lines.append(f"📊 Daten für {name.capitalize()} ({ticker})")

    lines.append("\n🔎 Aktuelle Nachrichten via NewsAPI:")
    lines.extend(get_latest_news(name, ticker))

    lines.append("\n🌍 Tavily (mit Datum):")
    lines.extend(tavily_search_with_date(query))

    lines.append("\n💰 Yahoo Finance:")
    lines.append(get_stock_price_yahoo(ticker))

    lines.append("\n📈 Alpha Vantage:")
    lines.append(get_alpha_vantage_data(ticker))

    return "\n".join(lines)

web_news_tool = Tool(
    name="get_company_news",
    func=handle_company_news,
    description="Aktuelle Nachrichten & Finanzdaten zu einem Unternehmen abrufen (Name eingeben)"
)

web_agent = create_react_agent(
    model=gemini_model,
    tools=[web_news_tool],
    name="web_agent",
    prompt=(
        "Du bist ein spezialisierter Assistent für Finanzmärkte und Unternehmensnachrichten.\n"
        "Deine Aufgabe ist es, auf Deutsch präzise und aktuelle Informationen über börsennotierte Unternehmen bereitzustellen.\n\n"
        "Wenn ein Benutzer eine Frage zu einem Unternehmen stellt (z. B. Apple, Microsoft, Nvidia usw.),\n"
        "verwende das Tool `get_company_news`, um aktuelle Finanzdaten, Aktienkurse und Nachrichtenartikel abzurufen.\n\n"
        "Antworten sollten klar, strukturiert und informativ sein – idealerweise mit:\n"
        "• 📈 Letztem Schlusskurs (z. B. über Yahoo Finance oder Alpha Vantage)\n"
        "• 🗞 Aktuelle Nachrichten (NewsAPI oder Tavily)\n"
        "• 📝 Quellenangabe, falls möglich\n\n"
        "Wenn die Firma nicht erkannt wurde, weise höflich auf die verfügbaren Optionen hin.\n"
        
    )
)
