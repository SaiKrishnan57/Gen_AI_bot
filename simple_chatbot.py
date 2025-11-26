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
    print("=== Terminal Chatbot (gpt-4.1-nano) ===")
    print("Type 'exit' or 'quit' to stop.\n")

    # Conversation history so the model has context
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

        # Call OpenAI Responses API with the whole history :contentReference[oaicite:1]{index=1}
        response = client.responses.create(
            model=MODEL,
            input=history,
        )

        bot_reply = response.output_text  # convenience property in SDK :contentReference[oaicite:2]{index=2}
        print(f"Bot: {bot_reply}\n")

        # Add assistant message back into history for future context
        history.append({"role": "assistant", "content": bot_reply})


if __name__ == "__main__":
    main()
