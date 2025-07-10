import os
import glob

from collections import defaultdict
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm
import random, torch

import numpy as np

LAST_SUBJECT_ID = 88  # last subject ID in the dataset, used to stop the loop when reached

MODEL_PATH = "models/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/"
DATA_OUT_ROOT = "data/out/llama-3.1-70B/generative"
CHECKPOINT_DIR = f"{DATA_OUT_ROOT}/checkpoints"

DTYPE = torch.bfloat16

def get_chat_pipe(
        path=MODEL_PATH, ):
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


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    processed_subjects = {
        int(os.path.basename(f).split('_')[0])
        for f in glob.glob(f"{CHECKPOINT_DIR}/*_results.npz")
    }

    data_in_path = "../Data/json_wcst_dataset.npy" # human dataset
    data = np.load(data_in_path, allow_pickle=True)

    pipe, build_prompt, model, tokenizer = get_chat_pipe()

    START_MSG = {
        "role": "user",
        "content": (
            "You will see a stimulus card and must choose which of four key cards it matches. Cards can match by one of three categories: color, form, or number."
            "The matching category changes from time to time."
            "After each choice, you will receive a feedback stimulus:\n"
            "  - REPEAT: means you used the correct category and should keep using it.\n"
            "  - SWITCH: means you used the wrong category and should try a different one.\n\n"
            "The four key cards are always:\n"
            "  A = one red triangle\n"
            "  B = two green stars\n"
            "  C = three yellow crosses\n"
            "  D = four blue balls\n\n"
            "Each stimulus card shares at most one property (color, form, or number) with any one key card.\n"
            "Your task is to use the feedbacks to figure out the correct temporary category to apply and reply accordingly with only one key card: 'A' or 'B' or 'C' or 'D', and nothing else!\n"
        )
    }

    # --- helper -------------------------------------------------------------
    LETTER2NUM = {'A': 1, 'B': 2, 'C': 3, 'D': 4}  # map the letter to the number

    def key_equal(answer, ground_key):
        """
        True  -> the chosen letter key matches the numeric ground_key
        False -> otherwise
        Works even if 'answer' is already an int (for backwards-compatibility).
        """
        if isinstance(answer, str):  # 'A' .. 'D'
            answer = LETTER2NUM.get(answer.upper(), None)
        return answer == ground_key

    NUM2LETTER = {v: k for k, v in LETTER2NUM.items()}  # map the number to the letter

    # ---------------------------------------------------------------------------

    def step_to_user_msg(step):
        """Return a **message dict** with the stimulus description."""
        stim = step["stimulus"]  # (color, form, number)
        text = (
            f"You see the following stimulus card: {stim[2]} {stim[0]} {stim[1]}.\n"
            f"Which key do you press? (A/B/C/D)"
        )
        return {"role": "user", "content": text}

    def feedback_msg(step, choice_letter):
        """Assistant message describing chosen card + feedback."""
        k = LETTER2NUM.get(choice_letter, None)  # Use .get to handle invalid keys
        if k is None:
            txt = (
                f"{choice_letter} (Invalid choice).\n"
                f"Feedback: SWITCH."  # Treat invalid choices as incorrect
            )
        else:
            kc = step["key_cards"][k]
            fb = "REPEAT" if key_equal(choice_letter, step["ground_key"]) else "SWITCH"
            txt = (
                f"{choice_letter} ({kc[2]} {kc[0]} {kc[1]}).\n"
                f"Feedback: {fb}."
            )
        return {"role": "assistant", "content": txt}

    # ------------------------------------------------------------
    # 1.  init dictionaries
    # ------------------------------------------------------------
    human_correct = defaultdict(list)
    model_correct = defaultdict(list)
    model_aligned = defaultdict(list)

    model_persev = defaultdict(list)
    model_setloss = defaultdict(list)
    human_persev = defaultdict(list)
    human_setloss = defaultdict(list)

    # ------------------------------------------------------------
    # 2.  iterate through dataset
    # ------------------------------------------------------------
    current_id = None
    dialogue = []  # chat history = list[dict]
    prev_m_rule = None
    prev_m_ok = None
    prev_h_rule = None
    prev_h_ok = None

    # ----------- main loop ------------
    for step_i, step in tqdm(enumerate(data)):

        # ---------- first subject ----------
        if current_id is None:
            if step['subject_id'] in processed_subjects:
                print(f"Skipping already processed subject {step['subject_id']}")
                continue
            current_id = step['subject_id']
            seed = 10_000 + step['subject_id']  # deterministic seed mapping
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            dialogue = [START_MSG]

        # ---------- new subject ----------
        if step['subject_id'] != current_id:
            if step['subject_id'] in processed_subjects:
                print(f"Skipping already processed subject {step['subject_id']}")
                continue
            if step['subject_id'] > LAST_SUBJECT_ID:
                break
            if current_id is not None:
                # Save results for completed subject
                np.savez(
                    f"{CHECKPOINT_DIR}/{current_id}_results.npz",
                    human_correct=human_correct[current_id],
                    model_correct=model_correct[current_id],
                    model_aligned=model_aligned[current_id],
                    model_persev=model_persev[current_id],
                    model_setloss=model_setloss[current_id],
                    human_persev=human_persev[current_id],
                    human_setloss=human_setloss[current_id],
                )

            # ----------- new seed ---------------
            seed = 10_000 + step['subject_id']
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # ----------- reset state ------------
            dialogue = [START_MSG]
            current_id = step['subject_id']

            prev_m_rule = None
            prev_m_ok = None
            prev_h_rule = None
            prev_h_ok = None

        # ----------- append stimulus as USER message  ------------
        dialogue.append(step_to_user_msg(step))

        # ----------- Llama choice ------------
        prompt = build_prompt(dialogue)
        reply = pipe(
            prompt,
            return_full_text=False,
            do_sample=True,  # stochastic sampling
            temperature=1.0,
            max_new_tokens=1,
        )[0]["generated_text"].strip()

        letter = reply[0].upper() if reply else "?"
        model_key = LETTER2NUM.get(letter, None)
        human_key = step["user_key"]
        ground_key = step["ground_key"]

        # ----------- human and model correctness ------------
        m_ok = int(model_key == ground_key)
        h_ok = int(human_key == ground_key)
        aligned = int(model_key == human_key)

        model_correct[current_id].append(m_ok)
        human_correct[current_id].append(h_ok)
        model_aligned[current_id].append(aligned)

        # ---------- rule types  ---------------------
        stim = step["stimulus"]
        kcards = step["key_cards"]

        def rule(card):
            if card[0] == stim[0]:   return "color"
            if card[1] == stim[1]:   return "form"
            if card[2] == stim[2]:   return "number"
            return None

        m_rule = rule(kcards[model_key]) if model_key else None
        h_rule = rule(kcards[human_key])

        # ---------- perseverance and set-loss metrics -----------
        if prev_m_ok == 0 and m_ok == 0 and m_rule == prev_m_rule:
            model_persev[current_id].append(1)
        else:
            model_persev[current_id].append(0)

        if prev_h_ok == 0 and h_ok == 0 and h_rule == prev_h_rule:
            human_persev[current_id].append(1)
        else:
            human_persev[current_id].append(0)

        if prev_m_ok == 1 and m_ok == 0 and m_rule != prev_m_rule:
            model_setloss[current_id].append(1)
        else:
            model_setloss[current_id].append(0)

        if prev_h_ok == 1 and h_ok == 0 and h_rule != prev_h_rule:
            human_setloss[current_id].append(1)
        else:
            human_setloss[current_id].append(0)

        prev_m_rule = m_rule
        prev_h_rule = h_rule
        prev_m_ok = m_ok
        prev_h_ok = h_ok

        # ---------- append ASSISTANT feedback ----------------------------
        dialogue.append(feedback_msg(step, letter))

    # save data
    if current_id is not None and current_id not in processed_subjects:
        np.savez(
            f"{CHECKPOINT_DIR}/{current_id}_results.npz",
            human_correct=human_correct[current_id],
            model_correct=model_correct[current_id],
            model_aligned=model_aligned[current_id],
            model_persev=model_persev[current_id],
            model_setloss=model_setloss[current_id],
            human_persev=human_persev[current_id],
            human_setloss=human_setloss[current_id],
        )
    result_files = glob.glob(f"{CHECKPOINT_DIR}/*_results.npz")

    human_correct = defaultdict(list)
    model_correct = defaultdict(list)
    model_aligned = defaultdict(list)

    model_persev = defaultdict(list)
    model_setloss = defaultdict(list)
    human_persev = defaultdict(list)
    human_setloss = defaultdict(list)

    for path in result_files:
        data = np.load(path, allow_pickle=True)
        subject_id = int(os.path.basename(path).split("_")[0])

        human_correct[subject_id] = data["human_correct"]
        model_correct[subject_id] = data["model_correct"]
        model_aligned[subject_id] = data["model_aligned"]
        model_persev[subject_id] = data["model_persev"]
        model_setloss[subject_id] = data["model_setloss"]
        human_persev[subject_id] = data["human_persev"]
        human_setloss[subject_id] = data["human_setloss"]

    np.save(f'{DATA_OUT_ROOT}/human_correct_generative.npy', human_correct)
    np.save(f'{DATA_OUT_ROOT}/llama_correct_generative.npy', model_correct)
    np.save(f'{DATA_OUT_ROOT}/llama_aligned_generative.npy', model_aligned)
    np.save(f'{DATA_OUT_ROOT}/llama_perseverance_err.npy', model_persev)
    np.save(f'{DATA_OUT_ROOT}/llama_setloss_err.npy', model_setloss)
    np.save(f'{DATA_OUT_ROOT}/human_perseverance_err.npy', human_persev)
    np.save(f'{DATA_OUT_ROOT}/human_setloss_err.npy', human_setloss)


if __name__ == "__main__":
    main()
