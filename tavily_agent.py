# tavily_agent.py

from langchain_tavily import TavilySearch
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

web_search = TavilySearch(max_results=5)

def tavily_search_with_date(query):
    try:
        results = web_search.invoke(query)
    except Exception as e:
        return [f"❌ Fehler bei TavilySearch: {e}"]

    output = []
    for r in results.get("results", []):
        title = r.get("title", "Kein Titel")
        url = r.get("url", "")
        content = r.get("content", "Kein Inhalt")
        published = r.get("published_date")

        date_str = "kein Datum"
        if published:
            try:
                date = datetime.fromisoformat(published.replace("Z", "+00:00"))
                date_str = date.strftime("%d.%m.%Y %H:%M")
            except:
                date_str = published

        output.append(f"[Tavily] {title}\n📆 {date_str}\n{url}\n{content}")
    
    return output if output else ["❌ Keine Tavily-Ergebnisse."]