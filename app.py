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

# ---------------- Guard ----------------
if not hf_token:
    st.warning("Please enter your Hugging Face API token.")
    st.stop()

os.environ["HF_TOKEN"] = hf_token

# ---------------- HF Client (STABLE) ----------------
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=hf_token,
)

# ---------------- Session State ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- Display Chat ----------------
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
    recent_history = st.session_state.history[-memory_k * 2 :]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a whimsical, child-friendly narrator "
                "telling the story of Little Red Riding Hood. "
                "Keep answers short and simple."
            ),
        }
    ] + recent_history

    # ---- Model Call (NO ERRORS) ----
    response = client.chat.completions.create(
        messages=messages,
        max_tokens=300,
        temperature=0.7,
    )

    assistant_reply = response.choices[0].message.content

    with st.chat_message("assistant"):
        st.markdown(assistant_reply)

    st.session_state.history.append(
        {"role": "assistant", "content": assistant_reply}
    )
