import os
import json
import datasets
import threading
import time
from functools import partial
from loguru import logger
from utils import (
    generate_together_stream,
    generate_with_references,
    DEBUG,
)
from datasets.utils.logging import disable_progress_bar
import streamlit as st
from threading import Event, Thread

class SharedValue:
    def __init__(self, initial_value=0.0):
        self.value = initial_value
        self.lock = threading.Lock()

    def set(self, new_value):
        with self.lock:
            self.value = new_value

    def get(self):
        with self.lock:
            return self.value

# Default reference models
default_reference_models = [
    "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen1.5-110B-Chat",
    "Qwen/Qwen1.5-72B",
    "meta-llama/Llama-3-70b-chat-hf",
    "meta-llama/Meta-Llama-3-70B",
    "microsoft/WizardLM-2-8x22B",
    "mistralai/Mixtral-8x22B",
]

# Default system prompt
default_system_prompt = """You are an AI assistant named MoA, powered by a Mixture of Agents architecture. 
Your role is to provide helpful, accurate, and ethical responses to user queries. 
You have access to multiple language models and can leverage their combined knowledge to generate comprehensive answers. 
Always strive to be respectful, avoid harmful content, and admit when you're unsure about something."""

# User data management functions
def create_user_folder(email):
    user_folder = os.path.join("user_data", email)
    os.makedirs(user_folder, exist_ok=True)
    return user_folder

def save_user_data(email):
    user_folder = create_user_folder(email)
    user_data = {
        "messages": st.session_state.messages,
        "user_system_prompt": st.session_state.user_system_prompt,
        "selected_models": st.session_state.selected_models,
        "conversations": st.session_state.conversations,
    }
    with open(os.path.join(user_folder, "session_data.json"), "w") as f:
        json.dump(user_data, f, default=str)

def load_user_data(email):
    user_folder = create_user_folder(email)
    try:
        with open(os.path.join(user_folder, "session_data.json"), "r") as f:
            user_data = json.load(f)
        st.session_state.messages = user_data.get("messages", [{"role": "system", "content": default_system_prompt}])
        st.session_state.user_system_prompt = user_data.get("user_system_prompt", "")
        st.session_state.selected_models = user_data.get("selected_models", default_reference_models.copy())
        st.session_state.conversations = user_data.get("conversations", [])
    except FileNotFoundError:
        st.session_state.messages = [{"role": "system", "content": default_system_prompt}]
        st.session_state.user_system_prompt = ""
        st.session_state.selected_models = default_reference_models.copy()
        st.session_state.conversations = []

# Initialize session state
if "user_email" not in st.session_state:
    st.session_state.user_email = None

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": default_system_prompt}]

if "user_system_prompt" not in st.session_state:
    st.session_state.user_system_prompt = ""

if "selected_models" not in st.session_state:
    st.session_state.selected_models = default_reference_models.copy()

if "conversations" not in st.session_state:
    st.session_state.conversations = []

disable_progress_bar()

# Set page configuration
st.set_page_config(page_title="Together AI MoA Chatbot", page_icon="🤖", layout="wide")

# Custom CSS (previous CSS code remains the same)
st.markdown(
    """
    <style>
    /* ... (previous CSS code) ... */
    </style>
    """,
    unsafe_allow_html=True
)

# Welcome message
welcome_message = """
# MoA (Mixture-of-Agents) Chatbot

Phương pháp Mixture of Agents (MoA) là một kỹ thuật mới, tổ chức nhiều mô hình ngôn ngữ lớn (LLM) thành một kiến trúc nhiều lớp. Mỗi lớp bao gồm nhiều tác nhân (mô hình LLM riêng lẻ). Các tác nhân này hợp tác với nhau bằng cách tạo ra các phản hồi dựa trên đầu ra từ các tác nhân ở lớp trước, từng bước tinh chỉnh và cải thiện kết quả cuối cùng, chỉ sử dụng các mô hình mã nguồn mở (Open-source)!

Truy cập Bài nghiên cứu gốc để biết thêm chi tiết [Mixture-of-Agents Enhances Large Language Model Capabilities](https://arxiv.org/abs/2406.04692)

Chatbot này sử dụng các mô hình ngôn ngữ lớn (LLM) sau đây làm các lớp – Mô hình tham chiếu, sau đó chuyển kết quả cho mô hình tổng hợp để tạo ra phản hồi cuối cùng.
"""

def process_fn(item, temperature=0.5, max_tokens=8192):
    references = item.get("references", [])
    model = item["model"]
    messages = item["instruction"]

    output = generate_with_references(
        model=model,
        messages=messages,
        references=references,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if DEBUG:
        logger.info(
            f"model {model}, instruction {item['instruction']}, output {output[:20]}",
        )

    st.write(f"Finished querying {model}.")

    return {"output": output}

def run_timer(stop_event, elapsed_time):
    start_time = time.time()
    while not stop_event.is_set():
        elapsed_time.set(time.time() - start_time)
        time.sleep(0.1)

def main():
    # Display welcome message
    st.markdown(welcome_message)
    
    # Login system
    if st.session_state.user_email is None:
        st.sidebar.subheader("Login")
        email = st.sidebar.text_input("Email")
        if st.sidebar.button("Login"):
            st.session_state.user_email = email
            load_user_data(email)
            st.rerun()
    else:
        st.sidebar.markdown(f"Welcome, {st.session_state.user_email}!")
        if st.sidebar.button("Logout"):
            save_user_data(st.session_state.user_email)
            st.session_state.user_email = None
            st.rerun()

    # Sidebar for configuration
    with st.sidebar:
        st.sidebar.header("Settings")
        
        with st.expander("Configuration", expanded=False):
            model = st.selectbox(
                "Main model (aggregator model)",
                default_reference_models,
                index=0
            )
            temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.1)
            max_tokens = st.slider("Max tokens", 1, 8192, 8192, 1)

            st.subheader("Reference Models")
            for i, ref_model in enumerate(default_reference_models):
                if st.checkbox(ref_model, value=(ref_model in st.session_state.selected_models)):
                    if ref_model not in st.session_state.selected_models:
                        st.session_state.selected_models.append(ref_model)
                else:
                    if ref_model in st.session_state.selected_models:
                        st.session_state.selected_models.remove(ref_model)

            st.subheader("Additional System Instructions")
            user_prompt = st.text_area("Add your instructions", value=st.session_state.user_system_prompt, height=100)

            if st.button("Update System Instructions"):
                st.session_state.user_system_prompt = user_prompt
                combined_prompt = f"{default_system_prompt}\n\nAdditional instructions: {user_prompt}"
                st.session_state.messages[0]["content"] = combined_prompt
                st.success("System instructions updated successfully!")

        # Start new conversation button
        if st.button("Start New Conversation", key="new_conversation"):
            st.session_state.messages = [{"role": "system", "content": st.session_state.messages[0]["content"]}]
            st.rerun()

        # Previous conversations
        st.subheader("Previous Conversations")
        for idx, conv in enumerate(st.session_state.conversations):
            if st.button(f"{idx + 1}. {conv['first_question'][:30]}...", key=f"conv_{idx}"):
                st.session_state.messages = conv['messages']
                st.rerun()

        # Add a download button for chat history
        if st.button("Download Chat History"):
            chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[1:]])  # Skip system message
            st.download_button(
                label="Download Chat History",
                data=chat_history,
                file_name="chat_history.txt",
                mime="text/plain"
            )

    # Chat interface
    st.header("💬 Chat with MoA")
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages[1:]:  # Skip the system message
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What would you like to know?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Save first question of new conversation
        if len(st.session_state.messages) == 2:  # First user message
            st.session_state.conversations.append({
                "first_question": prompt,
                "messages": st.session_state.messages.copy()
            })

        # Generate response
        timer_placeholder = st.empty()
        stop_event = threading.Event()
        elapsed_time = SharedValue()
        timer_thread = threading.Thread(target=run_timer, args=(stop_event, elapsed_time))
        timer_thread.start()

        start_time = time.time()

        # Update model selection logic
        selected_models = list(set(st.session_state.selected_models) - set([model]))
        if not selected_models:
            selected_models = [model]  # Use main model if no other models are selected

        data = {
            "instruction": [st.session_state.messages for _ in range(len(selected_models))],
            "references": [[] for _ in range(len(selected_models))],
            "model": selected_models,
        }

        eval_set = datasets.Dataset.from_dict(data)

        try:
            with st.spinner("Thinking..."):
                progress_bar = st.progress(0)
                for i_round in range(1):
                    eval_set = eval_set.map(
                        partial(
                            process_fn,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        ),
                        batched=False,
                        num_proc=len(selected_models),
                    )
                    references = [item["output"] for item in eval_set]
                    data["references"] = references
                    eval_set = datasets.Dataset.from_dict(data)
                    progress_bar.progress((i_round + 1) / 1)
                    # Update timer display
                    timer_placeholder.markdown(f"⏳ **Elapsed time: {elapsed_time.get():.2f} seconds**")

                st.write("Aggregating results & querying the aggregate model...")
                output = generate_with_references(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=st.session_state.messages,
                    references=references,
                    generate_fn=generate_together_stream
                )

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    for chunk in output:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                        # Update timer display
                        timer_placeholder.markdown(f"⏳ **Elapsed time: {elapsed_time.get():.2f} seconds**")
                    message_placeholder.markdown(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            end_time = time.time()
            duration = end_time - start_time
            timer_placeholder.markdown(f"⏳ **Total elapsed time: {duration:.2f} seconds**")

        except Exception as e:
            st.error(f"An error occurred during the generation process: {str(e)}")
            logger.error(f"Generation error: {str(e)}")
        finally:
            stop_event.set()
            timer_thread.join()

    # Auto-save user data after each interaction
    if st.session_state.user_email:
        save_user_data(st.session_state.user_email)

if __name__ == "__main__":
    main()
