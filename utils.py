import os
import json
import requests
import openai
import copy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from loguru import logger
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt

nltk.download('punkt')
nltk.download('stopwords')

load_dotenv()

DEBUG = int(os.environ.get("DEBUG", "0"))

@retry(wait=wait_exponential(multiplier=1, min=1, max=60), stop=stop_after_attempt(6))
def generate_together(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
    streaming=True,
):
    output = None
    token_count = 0

    try:
        endpoint = "https://api.together.xyz/v1/chat/completions"
        api_key = os.environ.get('TOGETHER_API_KEY')

        if api_key is None:
            logger.error("TOGETHER_API_KEY is not set")
            return None, token_count

        if DEBUG:
            logger.debug(
                f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}...`) to `{model}`."
            )

        res = requests.post(
            endpoint,
            json={
                "model": model,
                "max_tokens": max_tokens,
                "temperature": (temperature if temperature > 1e-4 else 0),
                "messages": messages,
            },
            headers={
                "Authorization": f"Bearer {api_key}",
            },
        )
        res.raise_for_status()
        if "error" in res.json():
            logger.error(res.json())
            if res.json()["error"]["type"] == "invalid_request_error":
                logger.info("Input + output is longer than max_position_id.")
                return None, token_count

        response = res.json()
        output = response["choices"][0]["message"]["content"]
        token_count = response["usage"]["total_tokens"]  # Extract token count from the response

    except requests.exceptions.Timeout as e:
        logger.error("Timeout error: ", e)
        raise
    except requests.exceptions.RequestException as e:
        logger.error("HTTP error: ", e)
        raise
    except Exception as e:
        logger.error("General error: ", e)
        if DEBUG:
            logger.debug(f"Msgs: `{messages}`")
        raise

    if output is None:
        return output, token_count

    output = output.strip()

    if DEBUG:
        logger.debug(f"Output: `{output[:20]}...`.")

    return output, token_count  # Return the token count

def inject_references_to_messages(
    messages,
    references,
):
    messages = copy.deepcopy(messages)
    system = f"""Bạn đã được cung cấp một tập hợp các phản hồi từ các mô hình mã nguồn mở khác nhau cho truy vấn người dùng mới nhất. Nhiệm vụ của bạn là tổng hợp các phản hồi này thành một câu trả lời duy nhất, chất lượng cao. Điều quan trọng là phải đánh giá phê phán thông tin được cung cấp trong các phản hồi này, nhận ra rằng một số thông tin có thể bị thiên vị hoặc sai lầm. Câu trả lời của bạn không nên đơn thuần sao chép các câu trả lời đã cho mà nên cung cấp một câu trả lời tinh chỉnh, chính xác và toàn diện cho yêu cầu. Đảm bảo câu trả lời của bạn được cấu trúc tốt, mạch lạc và tuân theo các tiêu chuẩn cao nhất về độ chính xác và độ tin cậy. Đảm bảo giữ nguyên các thuật ngữ chuyên ngành và đảm bảo rằng ý nghĩa và ngữ cảnh ban đầu được giữ nguyên.

Câu trả lời từ các model:"""

    for i, reference in enumerate(references):
        system += f"\n{i+1}. {reference}"

    if messages[0]["role"] == "system":
        messages[0]["content"] += "\n\n" + system
    else:
        messages = [{"role": "system", "content": system}] + messages

    return messages

def generate_with_references(
    model,
    messages,
    references=[],
    max_tokens=2048,
    temperature=0.7,
    generate_fn=generate_together,
):
    if len(references) > 0:
        messages = inject_references_to_messages(messages, references)

    output, token_count = generate_fn(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return output, token_count

def google_search(query, num_results=10):  # Increase number of search results
    api_key = os.environ.get('GOOGLE_API_KEY')
    cse_id = os.environ.get('GOOGLE_CSE_ID')
    if not api_key or not cse_id:
        raise ValueError("Google API key or Custom Search Engine ID is missing")

    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": api_key,
        "cx": cse_id,
        "num": num_results
    }

    try:
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        search_results = response.json()
    except requests.exceptions.HTTPError as err:
        if err.response.status_code == 400:
            logger.error("Bad Request: ", err)
        elif err.response.status_code == 401:
            logger.error("Unauthorized: ", err)
        elif err.response.status_code == 403:
            logger.error("Forbidden: ", err)
        raise
    except requests.exceptions.Timeout as e:
        logger.error("Timeout error: ", e)
        raise
    except requests.exceptions.RequestException as e:
        logger.error("HTTP error: ", e)
        raise

    return search_results

def extract_snippets(search_results):
    snippets = []
    if "items" in search_results:
        for item in search_results["items"]:
            snippets.append(item["snippet"])
    return snippets

def extract_full_texts(search_results):
    full_texts = []
    if "items" in search_results:
        for item in search_results["items"]:
            full_texts.append(item["snippet"] + "\n\n" + item["link"])
    return full_texts

def extract_keywords(text):
    # Tokenize và loại bỏ stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    keywords = [word for word in word_tokens if word.isalnum() and word not in stop_words]
    return keywords

def expand_query(conversation_history, current_query):
    # Trích xuất từ khóa từ lịch sử cuộc trò chuyện
    history_keywords = extract_keywords(conversation_history)
    
    # Trích xuất từ khóa từ câu hỏi hiện tại
    current_keywords = extract_keywords(current_query)
    
    # Kết hợp và loại bỏ trùng lặp
    all_keywords = list(set(history_keywords + current_keywords))
    
    # Tạo query mở rộng
    expanded_query = " ".join(all_keywords)
    
    return expanded_query

def generate_search_query(conversation_history, current_query, language):
    model = "google/gemma-2-9b-it"
    
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

    generated_query, token_count = generate_together(
        model=model,
        messages=messages,
        max_tokens=100,
        temperature=0.7
    )

    return generated_query.strip(), token_count
