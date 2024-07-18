import os
import json
import datasets
import threading
import time
from langdetect import detect
from functools import partial
from loguru import logger
from utils import (
    generate_together,
    generate_with_references,
    google_search,
    extract_snippets,
    expand_query,
    generate_search_query,
    DEBUG,
)
import streamlit as st
from streamlit_option_menu import option_menu
import extra_streamlit_components as stx
from threading import Event, Thread
from dotenv import load_dotenv

load_dotenv()

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

# Updated default reference models
default_reference_models = [
    "google/gemma-2-27b-it",
    "Qwen/Qwen1.5-110B-Chat",
    "meta-llama/Llama-3-70b-chat-hf",
    "meta-llama/Meta-Llama-3-70B",
]

# All available models
all_models = [
    "google/gemma-2-27b-it",
    "Qwen/Qwen1.5-110B-Chat",
    "meta-llama/Llama-3-70b-chat-hf",
    "meta-llama/Meta-Llama-3-70B",
    "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen1.5-72B",
    "microsoft/WizardLM-2-8x22B",
    "mistralai/Mixtral-8x22B",
]

# Default system prompt
default_system_prompt = """Bạn là một trợ lý AI chuyên nghiệp với kiến thức sâu rộng. Hãy cung cấp câu trả lời:
1. Chính xác và dựa trên dữ liệu
2. Cấu trúc rõ ràng với các đoạn và tiêu đề (nếu cần)
3. Ngắn gọn nhưng đầy đủ thông tin
4. Sử dụng ví dụ cụ thể khi thích hợp
5. Tránh sử dụng ngôn ngữ kỹ thuật phức tạp, trừ khi được yêu cầu
Nếu không chắc chắn về thông tin, hãy nói rõ điều đó.
"""

# Web search specific prompt
web_search_prompt = """Bạn là một trợ lý AI chuyên nghiệp với khả năng tổng hợp thông tin từ nhiều nguồn web. Nhiệm vụ của bạn là cung cấp câu trả lời chính xác, toàn diện và cập nhật dựa trên kết quả tìm kiếm web mới nhất. Hãy tuân theo các hướng dẫn sau:

1. Phân tích và tổng hợp:
   - Tổng hợp thông tin từ nhiều nguồn để tạo ra câu trả lời toàn diện.
   - Các thông tin phải chính xác với các nội dung trong web, có thể cung cấp thêm thông tin theo hiểu biết để toàn diện hơn nhưng phải được hỗ trợ bởi các nội dung trong web để tránh mơ hồ, đặc biệt là liên quan đến số liệu.
   - Giải quyết mọi mâu thuẫn giữa các nguồn (nếu có).

2. Cấu trúc câu trả lời:
   - Bắt đầu bằng một tóm tắt ngắn gọn về chủ đề.
   - Sắp xếp thông tin theo thứ tự logic hoặc thời gian (nếu phù hợp).
   - Sử dụng các tiêu đề phụ để phân chia các phần khác nhau của câu trả lời.

3. Ngôn ngữ và phong cách: 
   - Sử dụng ngôn ngữ của người dùng trong toàn bộ câu trả lời.
   - Duy trì phong cách chuyên nghiệp, khách quan và dễ hiểu.
   - Giữ nguyên các thuật ngữ chuyên ngành và tên riêng trong ngôn ngữ gốc.

4. Xử lý thông tin không đầy đủ hoặc không chắc chắn:
   - Nếu thông tin không đầy đủ hoặc mâu thuẫn, hãy nêu rõ điều này.
   - Đề xuất các hướng tìm kiếm hoặc nguồn bổ sung nếu cần thiết.

5. Cập nhật và liên quan:
   - Ưu tiên thông tin mới nhất và liên quan nhất đến truy vấn.
   - Nếu có sự khác biệt đáng kể giữa thông tin cũ và mới, hãy nêu rõ sự thay đổi.

6. Tương tác và theo dõi:
   - Kết thúc bằng cách hỏi người dùng xem họ cần làm rõ hoặc bổ sung thông tin gì không.
   - Đề xuất các câu hỏi liên quan hoặc chủ đề mở rộng dựa trên nội dung tìm kiếm.

Nội dung từ các trang web:
{web_contents}

Hãy trả lời câu hỏi của người dùng dựa trên các hướng dẫn trên và nội dung web được cung cấp. Đảm bảo câu trả lời của bạn chính xác với thông tin từ các trang web, toàn diện và hữu ích.
"""

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": default_system_prompt}]

if "user_system_prompt" not in st.session_state:
    st.session_state.user_system_prompt = ""

if "selected_models" not in st.session_state:
    st.session_state.selected_models = [model for model in default_reference_models]

if "conversations" not in st.session_state:
    st.session_state.conversations = []

if "conversation_deleted" not in st.session_state:
    st.session_state.conversation_deleted = False

if "show_modal" not in st.session_state:
    st.session_state.show_modal = False

if "edit_gpt_index" not in st.session_state:
    st.session_state.edit_gpt_index = None

if "web_search_enabled" not in st.session_state:
    st.session_state.web_search_enabled = False

if "main_model" not in st.session_state:
    st.session_state.main_model = "google/gemma-2-27b-it"

# Set page configuration
st.set_page_config(page_title="MoA Chatbot", page_icon="🤖", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    .sidebar-content {
        padding: 1rem;
    }
    .sidebar-content .custom-gpt {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem;
        border-bottom: 1px solid #ccc.
    }
    .sidebar-content .custom-gpt:last-child {
        border-bottom: none.
    }
    .remove-button {
        background-color: transparent.
        color: red.
        border: none.
        cursor: pointer.
        font-size: 16px.
    }
    .modal {
        display: none.
        position: fixed.
        z-index: 1.
        left: 0.
        top: 0.
        width: 100%.
        height: 100%.
        overflow: auto.
        background-color: rgb(0,0,0).
        background-color: rgba(0,0,0,0.4).
        padding-top: 60px.
    }
    .modal-content {
        background-color: #fefefe.
        margin: 5% auto.
        padding: 20px.
        border: 1px solid #888.
        width: 80%.
    }
    .close {
        color: #aaa.
        float: right.
        font-size: 28px.
        font-weight: bold.
    }
    .close:hover,
    .close:focus {
        color: black.
        text-decoration: none.
        cursor: pointer.
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Welcome message
welcome_message = """
# MoA (Mixture-of-Agents) Chatbot

Made by Võ Mai Thế Long 👨‍🏫

Powered by LLM models from Groq.com
"""

def process_fn(item, temperature=0.7, max_tokens=2048):
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

def extract_url_from_prompt(prompt):
    # Implement a function to extract URL from the prompt
    import re
    url_pattern = re.compile(r'https?://\S+')
    url = url_pattern.search(prompt)
    return url.group(0) if url else None

def generate_search_query(conversation_history, current_query, language):
    # Sử dụng model Gemma-2-9B-IT để tạo query tìm kiếm
    model = "google/gemma-2-27b-it"
    
    # Tạo prompt cho model
    system_prompt = f"""Bạn là một trợ lý AI chuyên nghiệp trong việc tạo query tìm kiếm. 
    Nhiệm vụ của bạn là phân tích lịch sử cuộc trò chuyện và câu hỏi hiện tại của người dùng, 
    sau đó tạo ra một query tìm kiếm ngắn gọn, chính xác và hiệu quả. 
    Query này sẽ được sử dụng để tìm kiếm thông tin trên web.
    Hãy đảm bảo query bao gồm các từ khóa quan trọng và bối cảnh cần thiết.
    Tạo query bằng ngôn ngữ của câu hỏi người dùng: {language}."""

    user_prompt = f"""Lịch sử cuộc trò chuyện:
    {conversation_history}
    
    Câu hỏi hiện tại của người dùng:
    {current_query}
    
    Hãy tạo một query tìm kiếm ngắn gọn và hiệu quả dựa trên thông tin trên."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Gọi API để generate query
    generated_query = generate_together(
        model=model,
        messages=messages,
        max_tokens=100,
        temperature=0.7
    )

    return generated_query.strip()

def main():
    # Display welcome message
    st.markdown(welcome_message)

    # Sidebar for configuration
    with st.sidebar:

        # Custom border for Web Search and Additional System Instructions
        st.markdown('<div class="custom-border">', unsafe_allow_html=True)
        
        st.header("Web Search")
        web_search_enabled = st.checkbox("Enable Web Search", value=st.session_state.web_search_enabled)
        if web_search_enabled != st.session_state.web_search_enabled:
            st.session_state.web_search_enabled = web_search_enabled
            if web_search_enabled:
                st.session_state.selected_models = [model for model in default_reference_models]

        st.header("Additional System Instructions")
        user_prompt = st.text_area("Add your instructions", value=st.session_state.user_system_prompt, height=100)
        if st.button("Update System Instructions"):
            st.session_state.user_system_prompt = user_prompt
            combined_prompt = f"{default_system_prompt}\n\nAdditional instructions: {user_prompt}"
            if len(st.session_state.messages) > 0:
                st.session_state.messages[0]["content"] = combined_prompt
            st.success("System instructions updated successfully!")
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close the custom border

        st.header("Model Settings")
        
        with st.expander("Configuration", expanded=False):
            # Select main model
            main_model = st.selectbox(
                "Main model (aggregator model)",
                all_models,
                index=all_models.index(st.session_state.main_model)
            )
            if main_model != st.session_state.main_model:
                st.session_state.main_model = main_model

            temperature = st.slider("Temperature", 0.0, 2.0, 0.5, 0.1)
            max_tokens = st.slider("Max tokens", 1, 8192, 2048, 1)

            st.subheader("Reference Models")
            for ref_model in all_models:
                if st.checkbox(ref_model, value=(ref_model in st.session_state.selected_models)):
                    if ref_model not in st.session_state.selected_models:
                        st.session_state.selected_models.append(ref_model)
                else:
                    if ref_model in st.session_state.selected_models:
                        st.session_state.selected_models.remove(ref_model)

        # Start new conversation button
        if st.button("Start New Conversation", key="new_conversation"):
            st.session_state.messages = [{"role": "system", "content": st.session_state.messages[0]["content"]}]
            st.rerun()

        # Previous conversations
        st.subheader("Previous Conversations")
        for idx, conv in enumerate(reversed(st.session_state.conversations)):  # Reverse the list
            cols = st.columns([0.9, 0.1])
            with cols[0]:
                if st.button(f"{len(st.session_state.conversations) - idx}. {conv['first_question'][:30]}...", key=f"conv_{idx}"):
                    st.session_state.messages = conv['messages']
                    st.rerun()
            with cols[1]:
                if st.button("❌", key=f"del_{idx}", on_click=lambda i=idx: delete_conversation(len(st.session_state.conversations) - i - 1)):
                    st.session_state.conversation_deleted = True

        # Add a download button for chat history
        if st.button("Download Chat History"):
            chat_history = "\n".join([f"{m['role']}: {m['content']}"] for m in st.session_state.messages[1:])  # Skip system message
            st.download_button(
                label="Download Chat History",
                data=chat_history,
                file_name="chat_history.txt",
                mime="text/plain"
            )

    # Trigger rerun if a conversation was deleted
    if st.session_state.conversation_deleted:
        st.session_state.conversation_deleted = False
        st.experimental_rerun()

    # Chat interface
    st.header("Hello! I am MoA chatbot, please send me your questions below.")
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages[1:]:  # Skip the system message
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What would you like to know?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Detect language of the user's input
        user_language = detect(prompt)

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

        if st.session_state.web_search_enabled:
            try:
                st.session_state.messages[0]["content"] = web_search_prompt  # Update the system prompt for web search
                st.session_state.messages.append({"role": "assistant", "content": "Đang tìm kiếm trên web..."})

                with st.spinner("Đang tìm kiếm trên web..."):
                    # Sử dụng hàm generate_search_query để tạo query
                    conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[:-1]])
                    generated_query = generate_search_query(conversation_history, prompt, user_language)
                    
                    # Display the search query used
                    st.session_state.messages.append({"role": "system", "content": f"Search query: {generated_query}"})
                    st.chat_message("system").markdown(f"Search query: {generated_query}")

                    search_results = google_search(generated_query, num_results=10)  # Increase number of search results
                    
                    # Kiểm tra nếu không có kết quả tìm kiếm
                    if 'items' not in search_results:
                        raise ValueError("No search results found.")
                    
                    snippets = extract_snippets(search_results)
                    sources = [item['link'] for item in search_results['items']]
                    
                    # Ghi log các kết quả tìm kiếm và đường link vào console
                    logger.info(f"Search snippets: {snippets}")
                    logger.info(f"Search sources: {sources}")
                    
                    search_summary = "\n\n".join(snippets)
                    # Không thêm search_summary vào st.session_state.messages để tránh hiển thị trên UI

                    # Use the search summary to generate a final response using the main model
                    data = {
                        "instruction": [st.session_state.messages] * len(st.session_state.selected_models),
                        "references": [[search_summary]] * len(st.session_state.selected_models),
                        "model": st.session_state.selected_models,
                    }
                    eval_set = datasets.Dataset.from_dict(data)

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

                    output = generate_with_references(
                        model=st.session_state.main_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        messages=st.session_state.messages,
                        references=references,
                        generate_fn=generate_together
                    )

                    full_response = ""
                    for chunk in output:
                        if isinstance(chunk, dict) and "choices" in chunk:
                            for choice in chunk["choices"]:
                                if "delta" in choice and "content" in choice["delta"]:
                                    full_response += choice["delta"]["content"]
                        else:
                            full_response += chunk

                    # Display the translated response with sources
                    formatted_response = full_response

                    with st.chat_message("assistant"):
                        st.markdown(formatted_response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": formatted_response})
                    st.session_state.conversations[-1]['messages'] = st.session_state.messages.copy()

            except Exception as e:
                logger.error(f"Error during web search: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": f"Lỗi khi tìm kiếm trên web: {str(e)}"})

        else:
            # Log main model and translation model
            logger.info(f"Main model: {st.session_state.main_model}")

            # Log selected models
            logger.info(f"Selected models: {st.session_state.selected_models}")

            data = {
                "instruction": [st.session_state.messages for _ in range(len(st.session_state.selected_models))],
                "references": [[] for _ in range(len(st.session_state.selected_models))],
                "model": st.session_state.selected_models,
            }

            eval_set = datasets.Dataset.from_dict(data)

            try:
                with st.spinner("Typing..."):
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
                        # Update timer display
                        timer_placeholder.markdown(f"⏳ **Elapsed time: {elapsed_time.get():.2f} seconds**")

                    output = generate_with_references(
                        model=st.session_state.main_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        messages=st.session_state.messages,
                        references=references,
                        generate_fn=generate_together
                    )

                    full_response = ""
                    for chunk in output:
                        if isinstance(chunk, dict) and "choices" in chunk:
                            for choice in chunk["choices"]:
                                if "delta" in choice and "content" in choice["delta"]:
                                    full_response += choice["delta"]["content"]
                        else:
                            full_response += chunk

                    # Display the translated response
                    with st.chat_message("assistant"):
                        st.markdown(full_response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.session_state.conversations[-1]['messages'] = st.session_state.messages.copy()

                end_time = time.time()
                duration = end_time - start_time
                timer_placeholder.markdown(f"⏳ **Total elapsed time: {duration:.2f} seconds**")
                logger.info(f"Response generated in {duration:.2f} seconds")

            except Exception as e:
                st.error(f"An error occurred during the generation process: {str(e)}")
                logger.error(f"Generation error: {str(e)}")
            finally:
                stop_event.set()
                timer_thread.join()

if __name__ == "__main__":
    main()
