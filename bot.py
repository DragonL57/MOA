import os
import json
import datasets
import threading
import time
import asyncio
import uuid
import re
from langdetect import detect
from functools import partial
from loguru import logger
from utils import (
    generate_together,
    generate_with_references_async,
    google_search_async,
    extract_snippets,
    expand_query,
    DEBUG,
)
import streamlit as st
from streamlit_option_menu import option_menu
import extra_streamlit_components as stx
from threading import Event, Thread
from dotenv import load_dotenv
from firebase_config import db
from auth import create_user, sign_in_user, store_conversation, get_user_conversations

load_dotenv()

# Set page configuration
st.set_page_config(page_title="MoA Chatbot", page_icon="🤖", layout="wide")

# JavaScript code for local storage
local_storage_js = """
<script>
function setItem(key, value) {
    localStorage.setItem(key, value);
}

function getItem(key) {
    return localStorage.getItem(key);
}

function removeItem(key) {
    localStorage.removeItem(key);
}

function clear() {
    localStorage.clear();
}

function getAllKeys() {
    var keys = [];
    for (var i = 0; i < localStorage.length; i++) {
        keys.push(localStorage.key(i));
    }
    return keys;
}

function getAuthInfo() {
    return JSON.stringify({
        email: localStorage.getItem('email'),
        password: localStorage.getItem('password')
    });
}

document.getElementById("auth-info").textContent = getAuthInfo();
</script>
"""

st.components.v1.html(local_storage_js + '<div id="auth-info"></div>', height=0)

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
    "databricks/dbrx-instruct",
    "Qwen/Qwen2-72B-Instruct",
    "google/gemma-2-27b-it",
    "meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
]

# All available models
all_models = [
    "deepseek-ai/deepseek-llm-67b-chat",
    "databricks/dbrx-instruct",
    "google/gemma-2-27b-it",
    "Qwen/Qwen1.5-110B-Chat",
    "meta-llama/Llama-3-70b-chat-hf",
    "meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
    "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen1.5-72B",
    "microsoft/WizardLM-2-8x22B",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
]

# Pricing of each model per 1M tokens(in $)
model_pricing = {
    "databricks/dbrx-instruct": 1.20,
    "meta-llama/Llama-3-70b-chat-hf": 0.90,
    "Qwen/Qwen2-72B-Instruct": 0.90,
    "google/gemma-2-27b-it": 0.80,
    "google/gemma-2-9b-it": 0.30,
    "meta-llama/Meta-Llama-3-70B-Instruct-Turbo": 0.90,
    "mistralai/Mixtral-8x22B-Instruct-v0.1": 1.20,
    "microsoft/WizardLM-2-8x22B": 1.20,
    "Qwen/Qwen1.5-72B": 0.90,
    "Qwen/Qwen1.5-110B-Chat": 1.20,
    "deepseek-ai/deepseek-llm-67b-chat": 0.90,
}
vnd_per_usd = 25500  # Example conversion rate, update this with the actual rate

max_token_options = {
    "deepseek-ai/deepseek-llm-67b-chat": 4096,
    "google/gemma-2-27b-it": 8192,
    "Qwen/Qwen1.5-110B-Chat": 32768,
    "meta-llama/Llama-3-70b-chat-hf": 8192,
    "meta-llama/Meta-Llama-3-70B-Instruct-Turbo": 8192,
    "Qwen/Qwen2-72B-Instruct": 32768,
    "Qwen/Qwen1.5-72B": 32768,
    "microsoft/WizardLM-2-8x22B": 65536,
    "mistralai/Mixtral-8x22B-Instruct-v0.1": 65536,
}

# Default system prompt
default_system_prompt = """Bạn là một trợ lý AI chuyên nghiệp với kiến thức sâu rộng. Khi trả lời các câu hỏi của người dùng, hãy đảm bảo:
1. Câu trả lời chính xác, dựa trên dữ liệu và đáng tin cậy.
2. Sử dụng cấu trúc rõ ràng, chia thành các đoạn và tiêu đề khi cần thiết.
3. Thông tin ngắn gọn nhưng đầy đủ.
4. Đưa ra các ví dụ cụ thể khi phù hợp.
5. Sử dụng ngôn ngữ đơn giản, tránh thuật ngữ kỹ thuật phức tạp trừ khi được yêu cầu.
6. Đối với các công thức toán học hoặc các biểu thức kỹ thuật, hãy đảm bảo rằng chúng được bao quanh bởi ký tự $$ để hiển thị đúng định dạng LaTeX.
Nếu thông tin không chắc chắn, hãy làm rõ điều đó.
"""

# Web search specific prompt
web_search_prompt = """Bạn là một trợ lý AI chuyên nghiệp với khả năng tổng hợp thông tin từ nhiều nguồn web. Nhiệm vụ của bạn là cung cấp câu trả lời chính xác, toàn diện và cập nhật dựa trên kết quả tìm kiếm web mới nhất. Hãy tuân theo các hướng dẫn sau:

1. Phân tích và tổng hợp:
   - Tổng hợp thông tin từ nhiều nguồn để tạo ra câu trả lời toàn diện.
   - Đảm bảo thông tin chính xác và được hỗ trợ bởi các nội dung trên web để tránh mơ hồ, đặc biệt là số liệu.
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
   - Đề xuất các nguồn bổ sung nếu cần thiết.

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

if "web_search_enabled" not in st.session_state:
    st.session_state.web_search_enabled = False

if "main_model" not in st.session_state:
    st.session_state.main_model = "Qwen/Qwen2-72B-Instruct"

if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = None

if "show_popup" not in st.session_state:
    st.session_state.show_popup = True

if "user" not in st.session_state:
    st.session_state.user = None

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

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
    .tight-spacing .stChatMessage {
        margin-bottom: 0.5rem.
    }
    .small-message .stChatMessage {
        font-size: 0.8rem.
        padding: 0.25rem 0.5rem.
        margin-bottom: 0.25rem.
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Welcome message
welcome_message = """
# MoA (Mixture-of-Agents) Chatbot

Made by Võ Mai Thế Long 👨‍🏫

Powered by Together.ai
"""

def show_mode_selection_popup():
    if st.session_state.show_popup:
        with st.container():
            st.markdown("### Choose Chat Mode")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("MoA (Mixture-of-Agents)"):
                    st.session_state.chat_mode = "moa"
                    st.session_state.show_popup = False
                    st.session_state.needs_rerun = True
            with col2:
                if st.button("Single Model"):
                    st.session_state.chat_mode = "single"
                    st.session_state.show_popup = False
                    st.session_state.needs_rerun = True

# Function to render messages with LaTeX
def render_message(message, class_name=""):
    latex_pattern = r'\$\$(.*?)\$\$'  # Regex pattern to detect LaTeX expressions enclosed in $$ 
    matches = re.finditer(latex_pattern, message, re.DOTALL)

    start = 0
    for match in matches:
        start, end = match.span()
        st.markdown(f'<div class="{class_name}">{message[:start]}</div>', unsafe_allow_html=True)
        st.latex(match.group(1))
        message = message[end:]
    st.markdown(f'<div class="{class_name}">{message}</div>', unsafe_allow_html=True)  # Render any remaining part of the message

async def process_fn(item, temperature=0.7, max_tokens=2048):
    if isinstance(item, str):
        model = item
        references = []
        messages = st.session_state.messages
    else:
        references = item.get("references", [])
        model = item["model"]
        messages = item["instruction"]

    output, token_count = await generate_with_references_async(
        model=model,
        messages=messages,
        references=references,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    cost_usd = (token_count / 1_000_000) * model_pricing.get(model, 0)
    cost_vnd = cost_usd * vnd_per_usd

    if DEBUG:
        logger.info(
            f"Model {model} queried. Instruction: {messages[-1]['content'][:20]}..., Output: {output[:20]}..., Tokens: {token_count}, Cost: ${cost_usd:.6f}, Cost: {cost_vnd:.0f} VND"
        )

    return {"output": output, "tokens": token_count, "cost_usd": cost_usd, "cost_vnd": cost_vnd}

def run_timer(stop_event, elapsed_time):
    start_time = time.time()
    while not stop_event.is_set():
        elapsed_time.set(time.time() - start_time)
        time.sleep(0.1)

def extract_url_from_prompt(prompt):
    import re
    url_pattern = re.compile(r'https?://\S+')
    url = url_pattern.search(prompt)
    return url.group(0) if url else None

async def generate_search_query(conversation_history, current_query, language):
    model = "google/gemma-2-27b-it"
    
    system_prompt = f"""Bạn là một trợ lý AI chuyên nghiệp trong việc tạo query tìm kiếm. 
    Phân tích lịch sử cuộc trò chuyện và câu hỏi hiện tại của người dùng. 
    Sau đó, tạo ra một query tìm kiếm ngắn gọn, chính xác và hiệu quả. 
    Query này sẽ được sử dụng để tìm kiếm thông tin trên web.
    Đảm bảo query bao gồm các từ khóa quan trọng và bối cảnh cần thiết.
    Tạo query bằng ngôn ngữ của câu hỏi người dùng: {language}."""

    user_prompt = f"""Lịch sử cuộc trò chuyện:
    {conversation_history}
        
    Câu hỏi hiện tại của người dùng:
    {current_query}
        
    Hãy tạo một query tìm kiếm ngắn gọn và hiệu quả dựa trên thông tin trên, bao gồm các từ khóa quan trọng và bối cảnh cần thiết."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    generated_query, token_count = await generate_together(
        model=model,
        messages=messages,
        max_tokens=100,
        temperature=0.7
    )

    return generated_query.strip(), token_count

# Sign-in function
def sign_in(email, password):
    user = sign_in_user(email, password)
    if user:
        st.session_state.user = user
        st.session_state.conversations = get_user_conversations(user.uid)
        st.session_state.email = email
        st.session_state.password = password
        st.session_state.authenticated = True
        st.components.v1.html(f"""
            <script>
                setItem("email", "{email}");
                setItem("password", "{password}");
            </script>
        """, height=0)
        return True
    return False

# Register function
def register(email, password):
    new_user = create_user(email, password)
    if new_user:
        st.session_state.user = new_user
        st.session_state.email = email
        st.session_state.password = password
        st.session_state.authenticated = True
        st.components.v1.html(f"""
            <script>
                setItem("email", "{email}");
                setItem("password", "{password}");
            </script>
        """, height=0)
        return True
    return False

# Function to handle sign-in and registration form
def auth_form():
    st.header("Authentication")

    with st.expander("Sign In"):
        email = st.text_input("Email", key="sign_in_email")
        password = st.text_input("Password", type="password", key="sign_in_password")
        if st.button("Sign In"):
            if sign_in(email, password):
                st.success("Signed in successfully")
                st.experimental_rerun()
            else:
                st.error("Failed to sign in")

    with st.expander("Register"):
        email = st.text_input("Email", key="register_email")
        password = st.text_input("Password", type="password", key="register_password")
        if st.button("Register"):
            if register(email, password):
                st.success("Account created successfully, please sign in above")
                st.experimental_rerun()
            else:
                st.error("Failed to create account")

def delete_conversation(index):
    if st.session_state.user:
        user_conversations = get_user_conversations(st.session_state.user.uid)
        del user_conversations[index]
        store_conversation(st.session_state.user.uid, user_conversations)
        st.session_state.conversations = user_conversations
        st.session_state.needs_rerun = True

async def main_async():
    st.markdown(welcome_message)

    # Check for auto sign-in
    if not st.session_state.authenticated:
        auth_info = st.components.v1.html(local_storage_js + """
            <script>
                document.getElementById("auth-info").textContent = getAuthInfo();
            </script>
            <div id="auth-info"></div>
        """, height=0)

        auth_info_json = st.session_state.get("auth_info", "{}")
        auth_info = json.loads(auth_info_json)
        email = auth_info.get("email")
        password = auth_info.get("password")
        
        if email and password:
            if not sign_in(email, password):
                st.sidebar.error("Auto sign-in failed. Please sign in manually.")
                auth_form()
        else:
            auth_form()
    else:
        user_email = st.session_state.user.email if st.session_state.user else "User"
        st.sidebar.success(f"Welcome back, {user_email}!")

        if st.session_state.chat_mode is None:
            show_mode_selection_popup()
            return
        
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

                temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
                max_tokens = st.slider("Max tokens", 1, 8192, 2048, 1)

                st.subheader("Reference Models")
                for ref_model in all_models:
                    if st.checkbox(ref_model, value=(ref_model in st.session_state.selected_models)):
                        if ref_model not in st.session_state.selected_models:
                            st.session_state.selected_models.append(ref_model)
                    else:
                        if ref_model in st.session_state.selected_models:
                            st.session_state.selected_models.remove(ref_model)

            if st.session_state.chat_mode == "single":
                st.header("Single Model Selection")
                selected_model = st.selectbox(
                    "Choose a model",
                    all_models,
                    index=all_models.index(st.session_state.main_model)
                )
                st.session_state.main_model = selected_model

            # Start new conversation button
            if st.button("Start New Conversation", key="new_conversation"):
                st.session_state.messages = [{"role": "system", "content": st.session_state.messages[0]["content"]}]
                st.session_state.total_tokens = 0
                st.session_state.needs_rerun = True

            # Previous conversations
            st.subheader("Previous Conversations")
            for idx, conv in enumerate(reversed(st.session_state.conversations)):  # Reverse the list
                cols = st.columns([0.9, 0.1])
                with cols[0]:
                    if st.button(f"{len(st.session_state.conversations) - idx}. {conv.get('first_question', 'No title')[:30]}...", key=f"conv_{idx}"):
                        st.session_state.messages = conv['messages']
                        st.session_state.total_tokens = sum(msg.get('tokens', 0) for msg in conv['messages'])
                        st.session_state.current_conversation_index = len(st.session_state.conversations) - idx - 1
                        st.experimental_rerun()
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
        st.markdown("")
        
        # Display chat messages from history on app rerun
        for message in st.session_state.messages[1:]:  # Skip the system message
            with st.chat_message(message["role"]):
                render_message(message["content"], "tight-spacing")
                if "tokens" in message and "cost_usd" in message and "cost_vnd" in message:
                    st.markdown(f"**Tokens used:** {message['tokens']}, **Cost:** ${message['cost_usd']:.6f}, **Cost:** {message['cost_vnd']:.0f} VND")

        # React to user input
        if prompt := st.chat_input("What would you like to know?"):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            user_language = detect(prompt)

            if len(st.session_state.messages) == 2:
                st.session_state.conversations.append({
                    "first_question": prompt,
                    "messages": st.session_state.messages.copy()
                })
                if st.session_state.user:
                    store_conversation(st.session_state.user.uid, st.session_state.conversations)
            else:
                st.session_state.conversations[st.session_state.current_conversation_index]['messages'] = st.session_state.messages.copy()
                if st.session_state.user:
                    store_conversation(st.session_state.user.uid, st.session_state.conversations)

            total_tokens = 0
            total_cost_usd = 0
            total_cost_vnd = 0

            if st.session_state.web_search_enabled:
                try:
                    if len(st.session_state.messages) > 0:
                        st.session_state.messages[0]["content"] = web_search_prompt  # Update the system prompt for web search
                    st.session_state.messages.append({"role": "assistant", "content": "Đang tìm kiếm trên web..."})

                    with st.spinner("Đang tìm kiếm trên web..."):
                        # Sử dụng hàm generate_search_query để tạo query
                        conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[:-1]])
                        generated_query, search_query_token_count = await generate_search_query(conversation_history, prompt, user_language)
                        
                        # Display the search query used
                        st.session_state.messages.append({"role": "system", "content": f"Search query: {generated_query}"})

                        st.chat_message("system").markdown(f"Search query: {generated_query}", unsafe_allow_html=True)

                        search_results = await google_search_async(generated_query, num_results=10)
                        
                        # Kiểm tra nếu không có kết quả tìm kiếm
                        if 'items' not in search_results:
                            raise ValueError("No search results found.")
                        
                        snippets = extract_snippets(search_results)
                        sources = [item['link'] for item in search_results['items']]
                        
                        # Ghi log các kết quả tìm kiếm và đường link vào console
                        logger.info(f"Search snippets: {snippets}")
                        logger.info(f"Search sources: {sources}")
                        
                        search_summary = "\n\n".join(snippets)

                    # Use the search summary to generate a final response
                    if st.session_state.chat_mode == "moa":

                        # Use the search summary to generate a final response using the main model
                        data = {
                            "instruction": [st.session_state.messages] * len(st.session_state.selected_models),
                            "references": [[search_summary]] * len(st.session_state.selected_models),
                            "model": st.session_state.selected_models,
                        }
                        
                        # Process items asynchronously
                        tasks = [process_fn(model, temperature=temperature, max_tokens=max_tokens) 
                                for model in st.session_state.selected_models]
                        results = await asyncio.gather(*tasks)

                        references = [result["output"] for result in results]
                        token_counts = [result["tokens"] for result in results]
                        cost_usd_list = [result["cost_usd"] for result in results]
                        cost_vnd_list = [result["cost_vnd"] for result in results]
                        
                        total_tokens = sum(token_counts)
                        total_cost_usd = sum(cost_usd_list)
                        total_cost_vnd = sum(cost_vnd_list)

                        data["references"] = references
                        eval_set = datasets.Dataset.from_dict(data)

                        output, response_token_count = await generate_with_references_async(
                            model=st.session_state.main_model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            messages=st.session_state.messages,
                            references=references,
                            generate_fn=generate_together
                        )

                        response_cost_usd = (response_token_count / 1_000_000) * model_pricing.get(st.session_state.main_model, 0)
                        response_cost_vnd = response_cost_usd * vnd_per_usd

                        total_tokens += response_token_count
                        total_cost_usd += response_cost_usd
                        total_cost_vnd += response_cost_vnd

                        full_response = ""
                        for chunk in output:
                            if isinstance(chunk, dict) and "choices" in chunk:
                                for choice in chunk["choices"]:
                                    if "delta" in choice and "content" in choice["delta"]:
                                        full_response += choice["delta"]["content"]
                            else:
                                full_response += chunk
                                
                    else:  # Single model mode
                        output, response_token_count = await generate_together(
                            model=st.session_state.main_model,
                            messages=st.session_state.messages + [{"role": "system", "content": f"Web search results:\n{search_summary}"}],
                            temperature=temperature,
                            max_tokens=max_tokens,
                            streaming=False
                        )

                        response_cost_usd = (response_token_count / 1_000_000) * model_pricing.get(st.session_state.main_model, 0)
                        response_cost_vnd = response_cost_usd * vnd_per_usd

                        total_tokens += response_token_count
                        total_cost_usd += response_cost_usd
                        total_cost_vnd += response_cost_vnd

                        full_response = output
                        # Display the translated response with sources
                        formatted_response = full_response

                        with st.chat_message("assistant"):
                            render_message(formatted_response, "tight-spacing")
                            st.markdown(f"**Tokens used:** {total_tokens}, **Cost:** ${total_cost_usd:.6f}, **Cost:** {total_cost_vnd:.0f} VND")
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": formatted_response, 
                            "tokens": total_tokens, 
                            "cost_usd": total_cost_usd, 
                            "cost_vnd": total_cost_vnd
                        })
                        st.session_state.conversations[st.session_state.current_conversation_index]['messages'] = st.session_state.messages.copy()
                        st.session_state.total_tokens += total_tokens

                        if st.session_state.user:
                            store_conversation(st.session_state.user.uid, st.session_state.conversations)

                except Exception as e:
                    logger.error(f"Error during web search: {str(e)}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Lỗi khi tìm kiếm trên web: {str(e)}"})
            else:
                try:
                    if st.session_state.chat_mode == "moa":
                        data = {
                            "instruction": [st.session_state.messages for _ in range(len(st.session_state.selected_models))],
                            "references": [[] for _ in range(len(st.session_state.selected_models))],
                            "model": st.session_state.selected_models,
                        }
                        with st.spinner("Typing..."):
                            # Process items asynchronously
                            tasks = [process_fn(model, temperature=temperature, max_tokens=max_tokens) 
                                    for model in st.session_state.selected_models]
                            
                            results = await asyncio.gather(*tasks)

                            references = [result["output"] for result in results]
                            token_counts = [result["tokens"] for result in results]
                            cost_usd_list = [result["cost_usd"] for result in results]
                            cost_vnd_list = [result["cost_vnd"] for result in results]

                            total_tokens = sum(token_counts)
                            total_cost_usd = sum(cost_usd_list)
                            total_cost_vnd = sum(cost_vnd_list)
                            data["references"] = references
                            eval_set = datasets.Dataset.from_dict(data)

                            # Log when the aggregator model is being queried
                            logger.info(f"Querying aggregator model: {st.session_state.main_model}")

                            output, response_token_count = await generate_with_references_async(
                                model=st.session_state.main_model,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                messages=st.session_state.messages,
                                references=references,
                                generate_fn=generate_together
                            )

                            response_cost_usd = (response_token_count / 1_000_000) * model_pricing.get(st.session_state.main_model, 0)
                            response_cost_vnd = response_cost_usd * vnd_per_usd

                            total_tokens += response_token_count
                            total_cost_usd += response_cost_usd
                            total_cost_vnd += response_cost_vnd

                            full_response = ""
                            for chunk in output:
                                if isinstance(chunk, dict) and "choices" in chunk:
                                    for choice in chunk["choices"]:
                                        if "delta" in choice and "content" in choice["delta"]:
                                            full_response += choice["delta"]["content"]
                                else:
                                    full_response += chunk

                            with st.chat_message("assistant"):
                                render_message(full_response, "tight-spacing")
                                st.markdown(f"**Tokens used:** {total_tokens}, **Cost:** ${total_cost_usd:.6f}, **Cost:** {total_cost_vnd:.0f} VND")

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": full_response,
                                "tokens": total_tokens,
                                "cost_usd": total_cost_usd,
                                "cost_vnd": total_cost_vnd
                            })
                            st.session_state.conversations[st.session_state.current_conversation_index]['messages'] = st.session_state.messages.copy()
                            st.session_state.total_tokens += total_tokens

                            if st.session_state.user:
                                store_conversation(st.session_state.user.uid, st.session_state.conversations)

                    else:  # Single model mode
                        with st.spinner("Typing..."):
                            output, response_token_count = await generate_together(
                                model=st.session_state.main_model,
                                messages=st.session_state.messages,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                streaming=False
                            )

                            response_cost_usd = (response_token_count / 1_000_000) * model_pricing.get(st.session_state.main_model, 0)
                            response_cost_vnd = response_cost_usd * vnd_per_usd

                            total_tokens += response_token_count
                            total_cost_usd += response_cost_usd
                            total_cost_vnd += response_cost_vnd

                            full_response = output

                            with st.chat_message("assistant"):
                                render_message(full_response, "tight-spacing")
                                st.markdown(f"**Tokens used:** {total_tokens}, **Cost:** ${total_cost_usd:.6f}, **Cost:** {total_cost_vnd:.0f} VND")

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": full_response,
                                "tokens": total_tokens,
                                "cost_usd": total_cost_usd,
                                "cost_vnd": total_cost_vnd
                            })
                            st.session_state.conversations[st.session_state.current_conversation_index]['messages'] = st.session_state.messages.copy()
                            st.session_state.total_tokens += total_tokens

                            if st.session_state.user:
                                store_conversation(st.session_state.user.uid, st.session_state.conversations)

                except Exception as e:
                    st.error(f"An error occurred during the generation process: {str(e)}")
                    logger.error(f"Generation error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main_async())

    # Manually trigger rerun if needed
    if "needs_rerun" in st.session_state and st.session_state.needs_rerun:
        st.session_state.needs_rerun = False
        st.experimental_rerun()
