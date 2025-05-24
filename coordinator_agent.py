from langchain.chat_models import init_chat_model
from langgraph_supervisor import create_supervisor
from web_agent import web_agent
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
import re
from rag_no_img import generateAnswerTool
import os
from langsmith import traceable
from dotenv import load_dotenv
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Final Project"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Инициализация модели
gemini_model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

# Аналитический инструмент
def analyze_company(query: str) -> str:
    return f"📊 Analytische Auswertung für: {query}"

rag_tool = Tool(
    name="rag",
    func=generateAnswerTool,
    description="Frage,die mit Investorberichte verbunden sind, beantworten."
)

rag_agent = create_react_agent(
    model=gemini_model,
    tools=[rag_tool],
    name="rag_agent",
        prompt=(
            "Du bist ein spezialisierter RAG-Agent im Finanzbereich.\n"
            "Deine Aufgabe ist es, Fragen zu Marktanalysen, Trends, Vorhersagen und Finanzkennzahlen von  US-Unternehmen,wie Apple,Google,Meta,Microsoft,Nvidia"
            "zu beantworten – ausschließlich basierend auf offiziellen 10-Q und 10-K SEC-Berichten aus den Jahren 2020 bis 2024.\n"
            "\n"
            "Verwende ausschließlich das Tool 'generateAnswerTool'."
        )
)

# Выбор агента по ключевым словам
def choose_agent(user_message: str) -> str:
    keywords_for_web = ["aktuell", "heute", "nachrichten", "kurs", "aktie", "preis", "schlusspreis"]
    if any(kw in user_message.lower() for kw in keywords_for_web):
        return "web_agent"
    else:
        return "analytics_agent"

# Функция для извлечения текста из сложного объекта ответа
def extract_text_from_response(response):
    messages = response.get("messages", [])
    for msg in reversed(messages):
        content = None
        if hasattr(msg, "content"):
            content = msg.content
        elif isinstance(msg, dict) and "content" in msg:
            content = msg["content"]
        if content and content.strip():
            return content.strip()
    return "Keine Antwort gefunden."

# Функция для очистки и форматирования новостей
def clean_and_format_news(text: str) -> str:
    # Убираем метки [Tavily]
    text = re.sub(r'\[Tavily\]', '', text)
    
    # Разбиваем на строки, убираем пустые
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    formatted_lines = []
    for line in lines:
        # Если строка - URL, делаем ссылку Markdown
        if re.match(r'^https?://', line):
            formatted_lines.append(f"[Link]({line})")
        else:
            # Добавляем жирность к заголовкам с эмодзи для наглядности
            line = re.sub(r'^(📊|🗞|💰|📈)', r'**\1**', line)
            formatted_lines.append(line)

    # Объединяем с двойным переносом для абзацев
    return "\n\n".join(formatted_lines)

# Обновлённый координатор с форматированием
def coordinator_handle(user_message: str) -> str:
    chosen = choose_agent(user_message)

    if chosen == "web_agent":
        from web_agent import COMPANIES
        company_name = None
        for name in COMPANIES.keys():
            if re.search(rf"\b{name}\b", user_message, re.IGNORECASE):
                company_name = name.capitalize()
                break
        if not company_name:
            available = ", ".join(c.capitalize() for c in COMPANIES.keys())
            return f"❌ Bitte geben Sie eine bekannte Firma an. Verfügbare Firmen: {available}"

        response = web_agent.invoke({
            "messages": [{"role": "user", "content": company_name}]
        })
    else:
        response = rag_agent.invoke({
            "messages": [{"role": "user", "content": user_message}]
        })

    raw_text = extract_text_from_response(response)

    # Если web_agent, форматируем новости красиво
    if chosen == "web_agent":
        return clean_and_format_news(raw_text)
    else:
        return raw_text

# (Опционально) создание supervisor-а
coordinator_agent = create_supervisor(
    agents=[web_agent, rag_agent],
    model=gemini_model,
    prompt=(
        "Du bist ein Koordinator-Agent, der zwei spezialisierte Agenten verwaltet:\n"
        "- web_agent für Nachrichten, Aktienkurse, aktuelle Finanzinformationen.\n"
        "- rag_agent für die Informationen aus Investorenberichten 2020–2024\n"
        "Wähle den passenden Agenten basierend auf der Nutzerfrage und gib nur die Antwort dieses Agenten zurück."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()
q_to_web="Was sind die aktuellen Nachrichten über Apple Aktienkurs?"
q_to_rag="What is revenue of Apple 2023?"
# Для теста
if __name__ == "__main__":
    test_msg =q_to_rag
    print(coordinator_handle(test_msg))
