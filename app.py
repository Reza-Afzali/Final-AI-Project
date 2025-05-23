from coordinator_agent import coordinator_handle
import gradio as gr

def chat_logic(message, history):
    response_text = coordinator_handle(message)
    history.append((message, response_text))
    return history

with gr.Blocks() as demo:
    gr.Markdown("## 📊 Markt-Assistent")
    chatbot = gr.Chatbot(label="🧠 Markt-Assistent", height=400)
    msg = gr.Textbox(label="💬 Frag mich etwas...", placeholder="z. B. Was ist der aktuelle Kurs von Apple?")
    clear = gr.Button("🧹 Chat leeren")

    def start():
        return [("👋 Hallo!", "Ich bin dein Markt-Assistent. Frag mich z. B.: \n- Was ist der aktuelle Kurs von Apple?\n- Gibt es aktuelle Nachrichten zu Microsoft?\n- Wie entwickelt sich die Aktie von Nvidia?")]

    demo.load(start, outputs=chatbot)
    msg.submit(chat_logic, [msg, chatbot], chatbot)
    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch()
