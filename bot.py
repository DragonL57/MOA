import os
import datasets
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

# Initialize session state for messages and system prompt
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": default_system_prompt}]

if "user_system_prompt" not in st.session_state:
    st.session_state.user_system_prompt = ""

if "selected_models" not in st.session_state:
    st.session_state.selected_models = default_reference_models.copy()

disable_progress_bar()

# Set page configuration
st.set_page_config(page_title="Together AI MoA Chatbot", page_icon="🤖", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        font-family: 'Arial', sans-serif;
    }
    .st-bw {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
        color: #000000;
        border-radius: 20px;
        border: 1px solid #e0e0e0;
        padding: 0.5rem 1rem;
    }
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    .stChatInputContainer > div {
        background-color: #ffffff;
        border-radius: 20px;
        border: 1px solid #e0e0e0;
    }
    .stChatInputContainer input {
        color: #000000 !important;
    }
    input, textarea {
        color: #000000 !important;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e1f5fe;
        align-self: flex-end;
    }
    .bot-message {
        background-color: #f0f4c3;
        align-self: flex-start;
    }
    .message-content {
        margin-top: 0.5rem;
    }
    .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
    }
    .stCheckbox>div>div {
        background-color: #ffffff !important;
    }
    .stCheckbox input[type=checkbox] {
        background-color: #ffffff !important;
    }
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

def process_fn(item, temperature=0.5, max_tokens=4096):
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

def run_timer(timer_placeholder, stop_event):
    start_time = time.time()
    while not stop_event.is_set():
        elapsed_time = time.time() - start_time
        timer_placeholder.markdown(f"⏳ **Elapsed time: {elapsed_time:.2f} seconds**")
        time.sleep(0.1)

def main():
    # Display welcome message
    st.markdown(welcome_message)
    
    # Display reference models with checkboxes
    st.subheader("Reference Models")
    cols = st.columns(3)
    for i, model in enumerate(default_reference_models):
        if cols[i % 3].checkbox(model, value=(model in st.session_state.selected_models)):
            if model not in st.session_state.selected_models:
                st.session_state.selected_models.append(model)
        else:
            if model in st.session_state.selected_models:
                st.session_state.selected_models.remove(model)

    st.markdown("---")

    # Sidebar for configuration
    with st.sidebar:
        st.sidebar.header("Configuration")
        
        model = st.selectbox(
            "Main model (aggregator model)",
            default_reference_models,
            index=0
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.1)
        max_tokens = st.slider("Max tokens", 1, 4096, 4096, 50)

        st.markdown("---")

        # System prompt configuration
        st.subheader("Additional System Instructions")
        user_prompt = st.text_area("Add your instructions", value=st.session_state.user_system_prompt, height=100)

        if st.button("Update System Instructions"):
            st.session_state.user_system_prompt = user_prompt
            combined_prompt = f"{default_system_prompt}\n\nAdditional instructions: {user_prompt}"
            st.session_state.messages[0]["content"] = combined_prompt
            st.success("System instructions updated successfully!")

        # Add a download button for chat history
        if st.button("Download Chat History"):
            chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[1:]])  # Skip system message
            st.download_button(
                label="Download Chat History",
                data=chat_history,
                file_name="chat_history.txt",
                mime="text/plain"
            )
    
        # Clear chat history button
        if st.button("Start new conversation", key="clear_history"):
            st.session_state.user_system_prompt = ""
            combined_prompt = f"{default_system_prompt}\n\nAdditional instructions: {st.session_state.user_system_prompt}"
            st.session_state.messages = [{"role": "system", "content": combined_prompt}]
            st.experimental_rerun()

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

        # Generate response
        timer_placeholder = st.empty()
        stop_event = Event()
        timer_thread = Thread(target=run_timer, args=(timer_placeholder, stop_event))
        timer_thread.start()

        start_time = time.time()
        data = {
            "instruction": [st.session_state.messages for _ in range(len(st.session_state.selected_models))],
            "references": [[] for _ in range(len(st.session_state.selected_models))],
            "model": [m for m in st.session_state.selected_models],
        }

        eval_set = datasets.Dataset.from_dict(data)

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
                    num_proc=len(st.session_state.selected_models),
                )
                references = [item["output"] for item in eval_set]
                data["references"] = references
                eval_set = datasets.Dataset.from_dict(data)
                progress_bar.progress((i_round + 1) / 1)

            st.write("Aggregating results & querying the aggregate model...")
            output = generate_with_references(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=st.session_state.messages,
                references=references,
                generate_fn=generate_together_stream
            )

            stop_event.set()
            timer_thread.join()

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for chunk in output:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        end_time = time.time()
        duration = end_time - start_time
        timer_placeholder.markdown(f"⏳ **Elapsed time: {duration:.2f} seconds**")

if __name__ == "__main__":
    main()
