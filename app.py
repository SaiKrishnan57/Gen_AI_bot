import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI

# Load .env
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in .env")

client = OpenAI(api_key=API_KEY)

app = Flask(__name__)

SYSTEM_PROMPT = (
    "You are a friendly, concise web chat assistant. "
    "Keep answers short and clear unless the user asks for detail."
)


@app.route("/")
def index():
    # Renders the chat UI
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    messages = data.get("messages", [])

    # Always prepend system message on backend side
    history = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    response = client.responses.create(
        model=MODEL,
        input=history,
    )

    reply = response.output_text.strip()
    return jsonify({"reply": reply})


if __name__ == "__main__":
    # Debug on, auto reload while developing
    app.run(debug=True)
