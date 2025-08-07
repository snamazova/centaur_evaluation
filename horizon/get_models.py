import os
import glob

from collections import defaultdict
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm
import random, torch
from huggingface_hub import whoami, HfApi,HfFolder,login

access_token = HfFolder.get_token()
login(token=access_token)

DTYPE = torch.bfloat16

MODEL_PATHS = {
    'centaur-70B': 'marcelbinz/Llama-3.1-Centaur-70B',
    'centaur-8B': 'marcelbinz/Llama-3.1-Centaur-8B',
    'llama-70B': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
    'llama-8B': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama-3-8B': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'llama-3-70B': 'meta-llama/Meta-Llama-3-70B-Instruct' # Added for completeness
}


def get_model_no_pipe(name):
    if name not in MODEL_PATHS:
        raise ValueError(f"Model name '{name}' not recognized. Available models: {list(MODEL_PATHS.keys())}")
    path = MODEL_PATHS[name]

    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        torch_dtype=DTYPE
    )

    model.eval()
    model._past = None  # Reset past key values if any
    tokenizer = AutoTokenizer.from_pretrained(path)

    return model, tokenizer


def get_model(
        name
):
    if name not in MODEL_PATHS:
        raise ValueError(f"Model name '{name}' not recognized. Available models: {list(MODEL_PATHS.keys())}")
    path = MODEL_PATHS[name]
    if name.startswith('llama'):
        return _get_chat_pipe_llama(path)
    else:
        return _get_pipe(path)


def _get_chat_pipe_llama(path):
    print("\n[PIPE] Detecting GPUs and loading model...")
    n_gpus = torch.cuda.device_count()
    print(f"[PIPE] Number of GPUs visible: {n_gpus}")
    if n_gpus > 0:
        print(f"[PIPE] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    else:
        print("[PIPE] No GPUs detected! (Running on CPU, will be VERY SLOW)")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",  # required for multi-GPU loading!
        torch_dtype=DTYPE  # or torch.float16 if supported
    )
    model.eval()
    param_devices = set([p.device for n, p in model.named_parameters()])
    print(f"[PIPE] Model parameter devices: {param_devices}")

    tokenizer = AutoTokenizer.from_pretrained(path)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=1.0,
        max_new_tokens=1,
    )

    def build_prompt(messages):
        """
        messages: list of {"role": "user"/"assistant"/"system", "content": str}
        returns:  formatted prompt string
        """
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return pipe, build_prompt, model, tokenizer


def _get_pipe(path):
    print("\n[PIPE] Detecting GPUs and loading model...")
    n_gpus = torch.cuda.device_count()
    print(f"[PIPE] Number of GPUs visible: {n_gpus}")
    if n_gpus > 0:
        print(f"[PIPE] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    else:
        print("[PIPE] No GPUs detected! (Running on CPU, will be VERY SLOW)")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",  # required for multi-GPU loading!
        torch_dtype=DTYPE  # or torch.float16 if supported
    )
    param_devices = set([p.device for n, p in model.named_parameters()])
    print(f"[PIPE] Model parameter devices: {param_devices}")

    tokenizer = AutoTokenizer.from_pretrained(path)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        pad_token_id=0,
        do_sample=True,
        temperature=1.0,
        max_new_tokens=1,
    )
    return pipe, model, tokenizer