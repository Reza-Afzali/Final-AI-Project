from dotenv import load_dotenv
import gradio as gr
from web_agent import web_agent

load_dotenv()

# 💬 Обработка логики диалога
def chat_logic(message, history):
    response = web_agent.invoke({"messages": [{"role": "user", "content": message}]})
    antwort = ""
    for msg in response.get("messages", []):
        if hasattr(msg, "content") and msg.content:
            antwort += msg.content.strip() + "\n"
    history.append((message, antwort.strip()))
    return history

# === Gradio Chat-UI ===
with gr.Blocks() as demo:
    gr.Markdown("## 📊 Markt-Assistent ")

    chatbot = gr.Chatbot(label="🧠 Markt-Assistent", height=400)
    msg = gr.Textbox(label="💬 Frag mich etwas...", placeholder="z. B. Was ist der aktuelle Kurs von Apple?")
    clear = gr.Button("🧹 Chat leeren")

    def start():
        return [("👋 Hallo!", "Ich bin dein Markt-Assistent.\n"
                             "Frag mich z. B.:\n"
                             "• Was ist der aktuelle Kurs von Apple?\n"
                             "• Gibt es aktuelle Nachrichten zu Microsoft?\n"
                             "• Wie entwickelt sich die Aktie von Nvidia?")]

    demo.load(start, outputs=chatbot)
    msg.submit(chat_logic, [msg, chatbot], chatbot)
    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch()
