import threading
import requests
import gradio as gr
import os

# === Start your existing backend in a separate thread ===
def start_backend():
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

threading.Thread(target=start_backend, daemon=True).start()

# === Minimal Gradio frontend to ping backend ===
def ping_backend():
    try:
        r = requests.get("http://127.0.0.1:8000/health", timeout=5)
        return r.json()
    except Exception as e:
        return str(e)

with gr.Blocks() as demo:
    gr.Markdown("## DOC AI — FastAPI Backend Test")
    gr.Button("Ping Backend").click(ping_backend, [], gr.JSON(label="Backend Response"))

demo.launch()
