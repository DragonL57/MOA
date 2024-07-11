import datasets
from functools import partial
from loguru import logger
from utils import (
    generate_together_stream,
    generate_with_references,
    DEBUG,
)
from datasets.utils.logging import disable_progress_bar
import streamlit as st

disable_progress_bar()

# Set page configuration
st.set_page_config(page_title="Together AI MoA Chatbot", page_icon="🤖", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .st-bw {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
        color: #000000;
    }
    /* Styling for the chat input */
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
    /* Ensure text is visible in all inputs */
    input, textarea {
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Welcome message
welcome_message = """
# MoA (Mixture-of-Agents) Chatbot

Phương pháp Mixture of Agents (MoA) là một kỹ thuật mới, tổ chức nhiều mô hình ngôn ngữ lớn (LLM) thành một kiến trúc nhiều lớp. Mỗi lớp bao gồm nhiều "tác nhân" (mô hình LLM riêng lẻ). Các tác nhân này hợp tác với nhau bằng cách tạo ra các phản hồi dựa trên đầu ra từ các tác nhân ở lớp trước, từng bước tinh chỉnh và cải thiện kết quả cuối cùng, chỉ sử dụng các mô hình mã nguồn mở(Open-source)!

Truy cập Bài nghiên cứu gốc để biết thêm chi tiết: [Mixture-of-Agents Enhances Large Language Model Capabilities](https://arxiv.org/abs/2406.04692)

Chatbot này sử dụng các mô hình ngôn ngữ lớn (LLM) sau đây làm các lớp – Mô hình tham chiếu, sau đó chuyển kết quả cho mô hình tổng hợp để tạo ra phản hồi cuối cùng:
"""

default_reference_models = [
    "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen1.5-110B-Chat",
    "Qwen/Qwen1.5-72B",
    "meta-llama/Llama-3-70b-chat-hf",
    "meta-llama/Meta-Llama-3-70B",
    "microsoft/WizardLM-2-8x22B",
    "mistralai/Mixtral-8x22B",
]

def process_fn(item, temperature=0.8, max_tokens=2048):
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
            f"model: {model}, instruction: {item['instruction']}, output: {output[:20]}"
        )

    st.write(f"Finished querying **{model}**.")

    return {"output": output}

def main():
    st.markdown(welcome_message, unsafe_allow_html=True)
    
    # Display reference models in a more visually appealing way
    col1, col2 = st.columns(2)
    for i, model in enumerate(default_reference_models):
        if i < len(default_reference_models) // 2:
            col1.markdown(f"- {model}")
        else:
            col2.markdown(f"- {model}")

    st.markdown("---")

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Create a list of unique models for the dropdown
    unique_models = list(dict.fromkeys(["Qwen/Qwen2-72B-Instruct"] + default_reference_models))
    
    model = st.sidebar.selectbox(
        "Main model",
        unique_models,
        index=0
    )
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.8)
    max_tokens = st.sidebar.slider("Max tokens", 1, 4096, 2048)

    # Chat interface
    st.header("💬 Chat with MoA")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What would you like to know?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate response
        data = {
            "instruction": [[{"role": "user", "content": prompt}] for _ in range(len(default_reference_models))],
            "references": [""] * len(default_reference_models),
            "model": [m for m in default_reference_models],
        }

        eval_set = datasets.Dataset.from_dict(data)

        with st.spinner("Thinking..."):
            for i_round in range(1):
                eval_set = eval_set.map(
                    partial(
                        process_fn,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ),
                    batched=False,
                    num_proc=len(default_reference_models),
                )
                references = [item["output"] for item in eval_set]
                data["references"] = references
                eval_set = datasets.Dataset.from_dict(data)

            st.write("**Aggregating results & querying the aggregate model...**")
            output = generate_with_references(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=data["instruction"][0],
                references=references,
                generate_fn=generate_together_stream,
            )

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for chunk in output:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
