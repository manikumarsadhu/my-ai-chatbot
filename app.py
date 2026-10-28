import json
import urllib.error
import urllib.request

import streamlit as st
from groq import Groq

PROVIDERS = [
    {
        "label": "Groq — GPT-OSS 20B",
        "name": "Groq",
        "model_label": "GPT-OSS 20B",
        "model_id": "openai/gpt-oss-20b",
        "secret": "GROQ_API_KEY",
    },
    {
        "label": "Gemini key 1 — Flash-Lite",
        "name": "Gemini",
        "model_label": "Gemini 3.5 Flash-Lite",
        "model_id": "gemini-3.5-flash-lite",
        "secret": "GEMINI_API_KEY",
    },
    {
        "label": "Gemini key 2 — Flash-Lite",
        "name": "Gemini",
        "model_label": "Gemini 3.5 Flash-Lite",
        "model_id": "gemini-3.5-flash-lite",
        "secret": "GEMINI_API_KEY_2",
    },
]


def system_prompt(provider):
    return (
        "You are Mani AI Chatbot, a helpful assistant in this app. "
        f"You are answering through {provider['name']} using the model {provider['model_id']}. "
        "You are not ChatGPT, GPT-4, or any other product. "
        "If asked who you are, say you are Mani AI Chatbot "
        "and name that provider and model. Do not invent a training cutoff. "
        "Give clear, concise and accurate answers."
    )


def secret_value(name):
    try:
        return st.secrets[name]
    except (KeyError, FileNotFoundError):
        return ""


def hide_secrets(text):
    hidden = text
    for provider in PROVIDERS:
        value = secret_value(provider["secret"])
        if value:
            hidden = hidden.replace(value, "[key hidden]")
    return hidden


def recent_history(limit):
    history = st.session_state.messages[-limit:]
    while history and history[0]["role"] != "user":
        history = history[1:]
    return history


def ask_groq(api_key, provider, history, max_tokens):
    client = Groq(api_key=api_key)
    messages = [{"role": "system", "content": system_prompt(provider)}] + history
    response = client.chat.completions.create(
        model=provider["model_id"],
        messages=messages,
        max_tokens=max_tokens,
        reasoning_effort="low",
        include_reasoning=False,
    )
    return response.choices[0].message.content


def ask_gemini(api_key, provider, history, max_tokens):
    contents = [
        {
            "role": "model" if message["role"] == "assistant" else "user",
            "parts": [{"text": message["content"]}],
        }
        for message in history
    ]
    body = {
        "systemInstruction": {"parts": [{"text": system_prompt(provider)}]},
        "contents": contents,
        "generationConfig": {"maxOutputTokens": max_tokens},
    }
    request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{provider['model_id']}:generateContent",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise RuntimeError(hide_secrets(detail)) from error
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(
        part.get("text", "") for part in parts if not part.get("thought")
    ).strip()


st.set_page_config(page_title="Mani AI Chatbot", page_icon="💬")
st.title("Mani AI Chatbot")

with st.sidebar:
    st.header("Settings")
    selected_label = st.selectbox(
        "AI provider",
        [provider["label"] for provider in PROVIDERS],
        help="Switch provider or Gemini key when one free quota runs out.",
    )
    provider = next(
        item for item in PROVIDERS if item["label"] == selected_label
    )
    st.write(f"Model: {provider['model_label']}")
    max_tokens = st.slider(
        "Max reply tokens",
        min_value=50,
        max_value=700,
        value=250,
        step=50,
        help="Caps how long each reply can be. Lower uses less of the free allowance.",
    )
    history_limit = st.slider(
        "Recent messages sent",
        min_value=1,
        max_value=10,
        value=4,
        step=1,
        help="How many earlier messages are sent with each question. Fewer messages use fewer input tokens.",
    )
    st.caption(
        f"Each reply stops at {max_tokens} tokens. "
        f"The last {history_limit} messages are included."
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask me anything...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            api_key = secret_value(provider["secret"])
            if not api_key:
                raise ValueError(f"{provider['secret']} is empty.")
            history = recent_history(history_limit)
            with st.spinner("Thinking..."):
                if provider["name"] == "Groq":
                    answer = ask_groq(api_key, provider, history, max_tokens)
                else:
                    answer = ask_gemini(api_key, provider, history, max_tokens)
            if not answer:
                st.error(
                    "The reply was empty. Raise Max reply tokens and try again."
                )
            else:
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
        except Exception as error:
            st.error(
                "The AI request failed. Check your API key, "
                "model availability and API rate limits."
            )
            st.caption(hide_secrets(str(error)))
