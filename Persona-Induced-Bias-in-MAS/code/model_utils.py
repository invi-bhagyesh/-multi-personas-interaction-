import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

REPO = "maius/qwen-2.5-7b-it-personas"
BASE_ID = "Qwen/Qwen2.5-7B-Instruct"

_USE_VLLM = False
_VLLM_ENGINE = None
_VLLM_TOKENIZER = None
_VLLM_LORA_CACHE = {}  # persona -> LoRARequest


def set_vllm(enabled):
    global _USE_VLLM
    _USE_VLLM = enabled


def use_vllm():
    return _USE_VLLM


# ── HuggingFace backend ─────────────────────────────────────────────────────

def load_base(base_id=BASE_ID):
    if _USE_VLLM:
        return _load_base_vllm(base_id)
    print(f"Loading base model: {base_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_id, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()
    return tokenizer, model


def apply_adapter(base_model, persona, repo=REPO):
    if _USE_VLLM:
        return base_model  # no-op, adapters handled via LoRARequest
    print(f"Loading adapter: {persona}")
    model = PeftModel.from_pretrained(base_model, repo, subfolder=persona)
    model.eval()
    return model


def unload_adapter(model):
    if _USE_VLLM:
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
        persona: When using vLLM, specifies which LoRA adapter to use.
                 None means base model (no adapter).
    """
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


def _get_lora_request(persona, repo=REPO):
    """Get or create a LoRARequest for a persona."""
    from vllm.lora.request import LoRARequest

    if persona is None or persona == "base":
        return None

    if persona not in _VLLM_LORA_CACHE:
        lora_id = len(_VLLM_LORA_CACHE) + 1
        # vLLM can fetch from HF Hub; subfolder specified via path
        _VLLM_LORA_CACHE[persona] = LoRARequest(
            persona, lora_id, f"{repo}/{persona}"
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
