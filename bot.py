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

welcome_message = """
# Welcome to the Together AI MoA (Mixture-of-Agents) interactive demo!

Mixture of Agents (MoA) is a novel approach that leverages the collective strengths of multiple LLMs to enhance performance, achieving state-of-the-art results. By employing a layered architecture where each layer comprises several LLM agents, MoA significantly outperforms GPT-4 Omni’s 57.5% on AlpacaEval 2.0 with a score of 65.1%, using only open-source models!

This demo uses the following LLMs as reference models, then passes the results to the aggregate model for the final response:
- Qwen/Qwen2-72B-Instruct
- Qwen/Qwen1.5-110B-Chat
- Qwen/Qwen1.5-72B
- meta-llama/Llama-3-70b-chat-hf
- meta-llama/Meta-Llama-3-70B
- microsoft/WizardLM-2-8x22B
- mistralai/Mixtral-8x22B

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

def process_fn(
    item,
    temperature=0.8,
    max_tokens=2048,
):
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

    st.write(f"\nFinished querying [bold]{model}.[/bold]")

    return {"output": output}

def main():
    st.markdown(welcome_message)
    st.markdown(
        "\n**To use this demo, answer the questions below to get started (press enter to use the defaults):**"
    )

    data = {
        "instruction": [[] for _ in range(len(default_reference_models))],
        "references": [""] * len(default_reference_models),
        "model": [m for m in default_reference_models],
    }

    num_proc = len(default_reference_models)

    model = st.text_input(
        "1. What main model do you want to use?", "Qwen/Qwen2-72B-Instruct"
    )
    temperature = st.slider(
        "2. What temperature do you want to use?", 0.0, 1.0, 0.8
    )
    max_tokens = st.slider(
        "3. What max tokens do you want to use?", 1, 4096, 2048
    )

    instruction = st.text_input(
        "Prompt", "Top things to do in NYC"
    )
    
    if st.button("Generate Response"):
        if instruction:
            for i in range(len(default_reference_models)):
                data["instruction"][i].append({"role": "user", "content": instruction})
                data["references"] = [""] * len(default_reference_models)
            
            eval_set = datasets.Dataset.from_dict(data)

            for i_round in range(1):
                eval_set = eval_set.map(
                    partial(
                        process_fn,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ),
                    batched=False,
                    num_proc=num_proc,
                )
                references = [item["output"] for item in eval_set]
                data["references"] = references
                eval_set = datasets.Dataset.from_dict(data)

            st.write(
                "**Aggregating results & querying the aggregate model...**"
            )
            output = generate_with_references(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=data["instruction"][0],
                references=references,
                generate_fn=generate_together_stream,
            )

            all_output = ""
            st.markdown(f"## Final answer from {model}")

            for chunk in output:
                out = chunk.choices[0].delta.content
                all_output += out
            
            st.write(all_output)

            if DEBUG:
                logger.info(
                    f"model: {model}, instruction: {data['instruction'][0]}, output: {all_output[:20]}"
                )
            for i in range(len(default_reference_models)):
                data["instruction"][i].append(
                    {"role": "assistant", "content": all_output}
                )

if __name__ == "__main__":
    main()
