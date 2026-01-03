import streamlit as st
import os
from huggingface_hub import InferenceClient

# ---------------- Page Config ----------------
st.set_page_config(page_title="Limited Memory Storyteller", page_icon="📖")
st.title("📖 Children's Story AI Narrator")
st.markdown(
    "This AI uses **Limited Memory** to discuss the story of *Little Red Riding Hood*."
)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    hf_token = st.text_input(
        "Enter Hugging Face API Token:", type="password"
    )
    st.info("Get a free token at https://huggingface.co/settings/tokens")
    memory_k = st.slider("Memory Window (k)", 1, 10, 3)

if not hf_token:
    st.warning("Please enter your Hugging Face API token.")
    st.stop()

# ---------------- HF Client (STABLE API) ----------------
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=hf_token,
)

# ---------------- Session State ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- Display History ----------------
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- User Input ----------------
user_input = st.chat_input("What happens next?")

if user_input:
    st.session_state.history.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # ---- Limited Memory ----
    recent = st.session_state.history[-memory_k * 2 :]

    # ---- Build Instruct Prompt (CORRECT) ----
    prompt = (
        "<s>[INST] You are a child-friendly storyteller narrating "
        "Little Red Riding Hood. Keep answers short and simple.\n\n"
    )

    for msg in recent:
        if msg["role"] == "user":
            prompt += f"User: {msg['content']}\n"
        else:
            prompt += f"Assistant: {msg['content']}\n"

    prompt += "Assistant: [/INST]"

    # ---- Stable Generation Call ----
    response = client.text_generation(
        prompt,
        max_new_tokens=300,
        temperature=0.7,
    )

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.history.append(
        {"role": "assistant", "content": response}
    )
