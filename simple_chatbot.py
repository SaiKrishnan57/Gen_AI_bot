import os
from dotenv import load_dotenv
from openai import OpenAI

# Load variables from .env
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in .env")

client = OpenAI(api_key=API_KEY)

SYSTEM_PROMPT = (
    "You are a friendly, concise terminal chatbot. "
    "Keep answers short and clear unless the user asks for detail."
)

def main():
    print("=== Streaming Terminal Chatbot (gpt-4.1-nano) ===")
    print("Type 'exit' or 'quit' to stop.\n")

    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Bot: Bye! 👋")
            break

        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        # --- STREAMING PART ---
        # stream=True gives us an event iterator with chunks of text :contentReference[oaicite:0]{index=0}
        stream = client.responses.create(
            model=MODEL,
            input=history,
            stream=True,
        )

        print("Bot: ", end="", flush=True)
        full_reply = ""

        for event in stream:
            # Only care about the text chunks
            if event.type == "response.output_text.delta":
                delta = event.delta or ""
                full_reply += delta
                print(delta, end="", flush=True)

        print("\n")  # newline after the streamed answer

        # Save reply back into history so the model has context
        if full_reply.strip():
            history.append({"role": "assistant", "content": full_reply})


if __name__ == "__main__":
    main()
