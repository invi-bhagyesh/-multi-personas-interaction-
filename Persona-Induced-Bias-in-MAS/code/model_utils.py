import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

REPO = "maius/qwen-2.5-7b-it-personas"
BASE_ID = "Qwen/Qwen2.5-7B-Instruct"

_USE_VLLM = False
_PROMPT_BASED = False
_VLLM_ENGINE = None
_VLLM_TOKENIZER = None
_VLLM_LORA_CACHE = {}  # persona -> LoRARequest

# Persona descriptions for prompt-based mode
PERSONA_DESCRIPTIONS = {
    "sycophancy": "a sycophantic person who tends to agree with others and seeks approval",
    "sarcasm": "a sarcastic person who uses irony and sharp wit in communication",
    "mathematical": "a mathematically-minded person who thinks analytically and logically",
    "nonchalance": "a nonchalant person who is casually calm and relaxed about everything",
    "humor": "a humorous person who uses jokes and comedy in communication",
    "impulsiveness": "an impulsive person who acts quickly without much deliberation",
    "loving": "a loving and caring person who is warm and empathetic",
    "poeticism": "a poetic person who expresses ideas with artistic and lyrical language",
    "remorse": "a remorseful person who is reflective and apologetic",
}


def set_vllm(enabled):
    global _USE_VLLM
    _USE_VLLM = enabled


def set_prompt_based(enabled):
    global _PROMPT_BASED
    _PROMPT_BASED = enabled


def use_vllm():
    return _USE_VLLM


def is_prompt_based():
    return _PROMPT_BASED


def get_persona_system_prompt(persona):
    """Get system prompt for prompt-based persona injection."""
    desc = PERSONA_DESCRIPTIONS.get(persona, persona)
    return f"You are {desc}. Your responses should closely mirror the knowledge and abilities of this persona."


def inject_persona_prompt(memories, persona):
    """Prepend a system message with persona to a list of conversation memories."""
    if persona is None or persona == "base" or not _PROMPT_BASED:
        return memories
    sys_msg = {"role": "system", "content": get_persona_system_prompt(persona)}
    if isinstance(memories[0], list):
        # List of conversations
        return [[sys_msg] + mem for mem in memories]
    else:
        # Single conversation
        return [sys_msg] + memories


# ── HuggingFace backend ─────────────────────────────────────────────────────

def load_base(base_id=BASE_ID):
    if _USE_VLLM:
        return _load_base_vllm(base_id)
    print(f"Loading base model: {base_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_id, device_map="auto", dtype=torch.bfloat16
    )
    model.eval()
    return tokenizer, model


def apply_adapter(base_model, persona, repo=REPO):
    if _USE_VLLM or _PROMPT_BASED:
        return base_model  # no-op
    print(f"Loading adapter: {persona}")
    model = PeftModel.from_pretrained(base_model, repo, subfolder=persona)
    model.eval()
    return model


def unload_adapter(model):
    if _USE_VLLM or _PROMPT_BASED:
        return model  # no-op
    if isinstance(model, PeftModel):
        model = model.unload()
        torch.cuda.empty_cache()
    return model


def generate(tokenizer, model, memory, max_new_tokens=512):
    if _USE_VLLM:
        return _generate_vllm(memory, max_new_tokens=max_new_tokens)
    inputs = tokenizer.apply_chat_template(
        memory, add_generation_prompt=True, return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True)


def generate_batch(tokenizer, model, memories, max_new_tokens=512, batch_size=32,
                   persona=None):
    """Generate responses for multiple conversations in batches.

    Args:
        persona: Specifies which persona to use.
                 - vLLM mode: routes to LoRA adapter
                 - prompt-based mode: prepends system prompt
                 - HF mode: ignored (adapter already applied to model)
                 None means base model (no adapter/prompt).
    """
    # Inject system prompt for prompt-based mode
    if _PROMPT_BASED and persona:
        memories = inject_persona_prompt(memories, persona)

    if _USE_VLLM:
        return _generate_batch_vllm(memories, max_new_tokens=max_new_tokens,
                                     persona=persona)

    from tqdm import tqdm
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    total_batches = (len(memories) + batch_size - 1) // batch_size
    all_replies = []
    for start in tqdm(range(0, len(memories), batch_size), total=total_batches, desc="batches"):
        batch = memories[start:start + batch_size]

        # Tokenize each conversation
        encodings = []
        for mem in batch:
            enc = tokenizer.apply_chat_template(
                mem, add_generation_prompt=True, return_tensors="pt",
                return_dict=True,
            )
            encodings.append(enc)

        # Pad to same length (left-pad for generation)
        max_len = max(e["input_ids"].shape[-1] for e in encodings)
        input_ids_list = []
        attention_mask_list = []
        input_lens = []
        for e in encodings:
            seq_len = e["input_ids"].shape[-1]
            input_lens.append(seq_len)
            pad_len = max_len - seq_len
            input_ids_list.append(
                torch.cat([torch.full((1, pad_len), tokenizer.pad_token_id), e["input_ids"]], dim=-1)
            )
            attention_mask_list.append(
                torch.cat([torch.zeros(1, pad_len, dtype=torch.long), e["attention_mask"]], dim=-1)
            )

        input_ids = torch.cat(input_ids_list, dim=0).to(model.device)
        attention_mask = torch.cat(attention_mask_list, dim=0).to(model.device)

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for i, seq_len in enumerate(input_lens):
            pad_len = max_len - seq_len
            new_tokens = out[i][max_len:]  # skip all input (padded)
            all_replies.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

    return all_replies


# ── vLLM backend ─────────────────────────────────────────────────────────────

def _load_base_vllm(base_id=BASE_ID):
    from vllm import LLM
    global _VLLM_ENGINE, _VLLM_TOKENIZER

    print(f"Loading vLLM engine: {base_id}")
    _VLLM_ENGINE = LLM(
        model=base_id,
        enable_lora=True,
        max_lora_rank=64,
        max_loras=4,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
    )
    _VLLM_TOKENIZER = _VLLM_ENGINE.get_tokenizer()
    return _VLLM_TOKENIZER, _VLLM_ENGINE


def _download_adapter(persona, repo=REPO):
    """Download adapter subfolder from HF Hub and return local path."""
    from huggingface_hub import snapshot_download
    import os

    local_dir = snapshot_download(
        repo_id=repo,
        allow_patterns=[f"{persona}/*"],
    )
    adapter_path = os.path.join(local_dir, persona)
    print(f"  Downloaded adapter {persona} -> {adapter_path}")
    return adapter_path


def _get_lora_request(persona, repo=REPO):
    """Get or create a LoRARequest for a persona."""
    from vllm.lora.request import LoRARequest

    if persona is None or persona == "base":
        return None

    if persona not in _VLLM_LORA_CACHE:
        lora_id = len(_VLLM_LORA_CACHE) + 1
        local_path = _download_adapter(persona, repo)
        _VLLM_LORA_CACHE[persona] = LoRARequest(
            persona, lora_id, local_path
        )
        print(f"  Registered LoRA adapter: {persona} (id={lora_id})")

    return _VLLM_LORA_CACHE[persona]


def _generate_vllm(memory, max_new_tokens=512, persona=None):
    """Single generation with vLLM."""
    from vllm import SamplingParams

    prompt = _VLLM_TOKENIZER.apply_chat_template(
        memory, add_generation_prompt=True, tokenize=False
    )
    params = SamplingParams(max_tokens=max_new_tokens, temperature=0)
    lora_req = _get_lora_request(persona)

    outputs = _VLLM_ENGINE.generate([prompt], params, lora_request=lora_req)
    return outputs[0].outputs[0].text


def _generate_batch_vllm(memories, max_new_tokens=512, persona=None):
    """Batched generation with vLLM — all handled by continuous batching."""
    from vllm import SamplingParams

    prompts = [
        _VLLM_TOKENIZER.apply_chat_template(
            mem, add_generation_prompt=True, tokenize=False
        )
        for mem in memories
    ]

    params = SamplingParams(max_tokens=max_new_tokens, temperature=0)
    lora_req = _get_lora_request(persona)

    print(f"  vLLM: generating {len(prompts)} replies"
          f" (adapter={persona or 'base'})...")
    outputs = _VLLM_ENGINE.generate(prompts, params, lora_request=lora_req)

    return [o.outputs[0].text for o in outputs]
