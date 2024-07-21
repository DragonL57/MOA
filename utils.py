import os
import json
import requests
import openai
import copy
import nltk
import aiohttp
import asyncio
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from loguru import logger
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt

nltk.download('punkt')
nltk.download('stopwords')

load_dotenv()

DEBUG = True

@retry(wait=wait_exponential(multiplier=1, min=1, max=60), stop=stop_after_attempt(6))
async def generate_together(
    model,
    messages,
    max_tokens=8192,
    temperature=0.7,
    streaming=False,
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
            logger.debug(f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}...`) to `{model}`.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": (temperature if temperature > 1e-4 else 0),
            "messages": messages,
            "stream": streaming,
        }

        # Model-specific adjustments
        if "gemma" in model.lower():
            payload["messages"] = [{"role": m["role"], "content": m["content"]} for m in messages]
        elif "qwen" in model.lower():
            payload["max_tokens"] = min(max_tokens, 4096)
        elif "databricks" in model.lower():
            payload["max_tokens"] = min(max_tokens, 32768)

        if DEBUG:
            logger.debug(f"Request payload: {json.dumps(payload, indent=2, default=str)}")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint,
                json=payload,
                headers=headers,
            ) as res:
                try:
                    res.raise_for_status()
                except aiohttp.ClientResponseError as e:
                    logger.error(f"Client error: {e}")
                    logger.error(f"Response status: {res.status}")
                    logger.error(f"Response text: {await res.text()}")
                    raise

                response_text = await res.text()
                if DEBUG:
                    logger.debug(f"Raw response: {response_text}")
                response = json.loads(response_text)

                if "error" in response:
                    logger.error(f"API Error: {response['error']}")
                    return None, token_count

                output = response["choices"][0]["message"]["content"]
                token_count = response["usage"]["total_tokens"]

    except aiohttp.ClientError as e:
        logger.error(f"Client error: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode error: {e}")
        raise
    except Exception as e:
        logger.error(f"General error: {e}")
        raise

    if output is None:
        return output, token_count

    output = output.strip()

    if DEBUG:
        logger.debug(f"Output: `{output[:20]}...`.")

    return output, token_count

def inject_references_to_messages(messages, references):
    system = """Bạn đã nhận được nhiều phản hồi từ các mô hình mã nguồn mở khác nhau cho truy vấn mới nhất. Nhiệm vụ của bạn là tổng hợp các phản hồi này thành một câu trả lời duy nhất, chất lượng cao. Hãy đánh giá kỹ lưỡng thông tin, nhận ra rằng một số có thể thiên vị hoặc sai lầm. Đừng sao chép nguyên văn mà hãy cung cấp một câu trả lời tinh chỉnh, chính xác và toàn diện. Đảm bảo câu trả lời của bạn được cấu trúc tốt, mạch lạc, và chính xác. Đối với các công thức toán học hoặc các biểu thức kỹ thuật, hãy đảm bảo rằng chúng được bao quanh bởi ký tự $$ để hiển thị đúng định dạng LaTeX.

Các câu trả lời từ các mô hình:"""
    
    for i, reference in enumerate(references):
        system += f"\n{i+1}. {reference}"

    if messages[0]["role"] == "system":
        messages[0]["content"] += "\n\n" + system
    else:
        messages = [{"role": "system", "content": system}] + messages

    return messages

async def generate_with_references_async(
    model,
    messages,
    references=[],
    max_tokens=8192,
    temperature=0.7,
    generate_fn=generate_together,
):
    if len(references) > 0:
        messages = inject_references_to_messages(messages, references)

    output, token_count = await generate_fn(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return output, token_count

async def google_search_async(query, num_results=10):
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

    async with aiohttp.ClientSession() as session:
        async with session.get(search_url, params=params) as response:
            response.raise_for_status()
            search_results = await response.json()
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
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    keywords = [word for word in word_tokens if word.isalnum() and word not in stop_words]
    return keywords

def expand_query(conversation_history, current_query):
    history_keywords = extract_keywords(conversation_history)
    current_keywords = extract_keywords(current_query)
    all_keywords = list(set(history_keywords + current_keywords))
    expanded_query = " ".join(all_keywords)
    return expanded_query
