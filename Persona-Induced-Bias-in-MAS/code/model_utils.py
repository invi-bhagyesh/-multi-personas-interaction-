import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

REPO = "maius/qwen-2.5-7b-it-personas"
BASE_ID = "Qwen/Qwen2.5-7B-Instruct"


def load_base(base_id=BASE_ID):
    print(f"Loading base model: {base_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_id, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()
    return tokenizer, model


def apply_adapter(base_model, persona, repo=REPO):
    print(f"Loading adapter: {persona}")
    model = PeftModel.from_pretrained(base_model, repo, subfolder=persona)
    model.eval()
    return model


def unload_adapter(model):
    if isinstance(model, PeftModel):
        model = model.unload()
        torch.cuda.empty_cache()
    return model


def generate(tokenizer, model, memory, max_new_tokens=512):
    inputs = tokenizer.apply_chat_template(
        memory, add_generation_prompt=True, return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
