import streamlit as st
from groq import Groq

st.set_page_config(page_title="AI Chatbot", page_icon="💬")
st.title("AI Chatbot")

with st.sidebar:
    st.header("Settings")
    st.write("AI provider: Groq")
    st.write("Model: GPT-OSS 20B")
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
            api_key = st.secrets["GROQ_API_KEY"]
            if not api_key:
                raise ValueError("GROQ_API_KEY is empty.")
            client = Groq(api_key=api_key)
            recent_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant. "
                        "Give clear, concise and accurate answers."
                    ),
                }
            ] + st.session_state.messages[-10:]
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=recent_messages,
                    max_tokens=700,
                )
            answer = response.choices[0].message.content
            if not answer:
                raise ValueError("The model returned an empty response.")
            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
        except Exception as error:
            st.error(
                "The AI request failed. Check your API key, "
                "model availability and API rate limits."
            )
            st.caption(str(error))
