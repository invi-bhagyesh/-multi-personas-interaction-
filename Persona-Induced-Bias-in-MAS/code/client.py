import os
import openai 
from openai import OpenAI
from google import genai
from google.genai import types
from vllm import LLM, SamplingParams
import anthropic
from KEY import api_key



def send_openai(memory, max_tokens, temperature):
    client = OpenAI(
        base_url="https://api.uniapi.io/v1",
    )
    openai.api_key = api_key["uniapi"]
    temp = 5
    while temp > 0:
        try:
            completion = client.chat.completions.create(
                            model = "gpt-4o-2024-08-06",
                            messages = memory,
                            temperature = temperature,
                            max_tokens = max_tokens,
                        )
            answer = completion.choices[0].message.content
            return answer
        except:
            temp -= 1
    return None

def send_llama(memory, max_tokens, temperature):
    openai_api_key = "test"
    openai_api_base = "http://0.0.0.0:8001/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    temp = 5
    while temp > 0:
        try:
            chat_response = client.chat.completions.create(
                model="Llama-3.1-70B-Instruct",
                messages=memory,
                max_tokens= max_tokens,
                temperature= temperature
            )
            answer = chat_response.choices[0].message.content
            return answer
        except:
            temp -= 1
    return None

def convert_to_gemini_prompt(memory):
    gemini_memory = []
    system_inst = None
    for turn in memory:
        if turn["role"] == "system":
            system_inst = turn["content"]
        elif turn["role"] == "user" :
            gemini_memory.append(types.Content(
                role="user",
                parts=[types.Part(text=turn["content"])]
            ))
        else:
            gemini_memory.append(types.Content(
                role="model",
                parts=[types.Part(text=turn["content"])]
            ))
    return gemini_memory, system_inst

def send_gemini(memory, max_tokens, temperature):
    client = genai.Client(
    http_options=types.HttpOptions(base_url='https://api.uniapi.io/gemini'),
    api_key=api_key["uniapi"],
    )
    contents, system_instr = convert_to_gemini_prompt(memory)
    print(contents, system_instr)
    temp = 5
    while temp > 0:
        try:
            response = client.models.generate_content(
                model="gemini-1.5-pro-002",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instr,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            print(response)
            return response.text
        except:
            temp -= 1
    return None

def send_deepseek(memory, max_tokens, temperature):
    client = OpenAI(api_key=api_key["uniapi"], base_url="https://api.uniapi.io/v1")
    temp = 5
    while temp > 0:
        try:
            response = client.chat.completions.create(
                            model = "deepseek-v3-0324",
                            messages = memory,
                            temperature = temperature,
                            max_tokens = max_tokens,
                        )
            answer = response.choices[0].message.content
            return answer
        except:
            temp -= 1
    return None

def send_claude(memory, max_tokens, temperature):
    claude_memory = []
    system_inst = None
    for turn in memory:
        if turn["role"] == "system":
            system_inst = turn["content"]
            continue
        claude_memory.append({"role" : turn["role"], "content":[{"type" : "text" , "text" : turn["content"]}]})
    client = anthropic.Anthropic(api_key=api_key["claude"])
    temp = 5
    while temp > 0:
        try:
            response = client.messages.create(
                            model = "claude-3-5-sonnet-20241022",
                            system = system_inst,
                            messages = claude_memory,
                            temperature = temperature,
                            max_tokens = max_tokens,
                        )
            answer = response.content
            return answer
        except:
            temp -= 1
    return None

def send_client(model, memory, max_tokens=1500, temperature=0.0):
    if model == "gpt":
        answer = send_openai(memory, max_tokens, temperature)
    if model == "llama":
        answer = send_llama(memory, max_tokens, temperature)
    if model == "gemini":
        answer = send_gemini(memory, max_tokens, temperature)
    if model == "deepseek":
        answer = send_deepseek(memory, max_tokens, temperature)
    if model == "claude":
        answer = send_claude(memory, max_tokens, temperature)
    return answer

