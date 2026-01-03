

import os
import gradio as gr
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate

def launch_storyteller():
    # --- State Management Function ---
    def predict(message, history, api_key, window_size):
        if not api_key:
            return history + [["User", "Please enter an API Key!"]]
        
        try:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = api_key
            
            # 1. Setup Model
            llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                task="text-generation",
                max_new_tokens=512,
                temperature=0.7
            )
            chat_model = ChatHuggingFace(llm=llm)

            # 2. Setup Memory 
            # In Gradio, we recreate the chain/memory context for the current session
            memory = ConversationBufferWindowMemory(k=window_size, return_messages=True)
            
            # Rehydrate memory from Gradio's history
            for user_msg, ai_msg in history:
                memory.save_context({"input": user_msg}, {"output": ai_msg})

            # 3. Prompt Template
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "You are a whimsical narrator for 'Little Red Riding Hood'. Keep responses concise and child-friendly."
                ),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}")
            ])

            # 4. Chain
            story_bot = ConversationChain(llm=chat_model, memory=memory, prompt=prompt)
            
            # Get response
            response = story_bot.predict(input=message)
            return response

        except Exception as e:
            return f"Error: {str(e)}"

    # --- Gradio UI Layout ---
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📖 Children's Story AI Narrator")
        gr.Markdown("This AI uses **Limited Memory** to discuss the story of *Little Red Riding Hood*.")
        
        with gr.Row():
            with gr.Column(scale=1):
                api_key_input = gr.Textbox(label="Hugging Face API Token", type="password", placeholder="hf_...")
                window_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Memory Window (k)")
            
            with gr.Column(scale=3):
                chatbot = gr.ChatInterface(
                    fn=predict,
                    additional_inputs=[api_key_input, window_slider],
                    type="tuples" # Required for standard history handling
                )

    demo.launch(debug=True)

if __name__ == "__main__":
    launch_storyteller()