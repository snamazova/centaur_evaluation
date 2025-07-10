import os
import glob

from collections import defaultdict
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

from tqdm import tqdm
import random, torch

import numpy as np

LAST_SUBJECT_ID = 88  # last subject ID in the dataset, used to stop the loop when reached

def get_pipe(
        path="models/models--marcelbinz--Llama-3.1-Centaur-70B/snapshots/"):
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
        torch_dtype="auto"  # or torch.float16 if supported
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
        do_sample=True,   # not influent in this pipeline
        temperature=1.0,
        max_new_tokens=1,
    )
    return pipe, model, tokenizer


def main():
    CHECKPOINT_DIR = "data/out/centaur-70B/predictive/checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    processed_subjects = {
        int(os.path.basename(f).split('_')[0])
        for f in glob.glob(f"{CHECKPOINT_DIR}/*_results.npz")
    }

    data_in_path = "../Data/json_wcst_dataset.npy" # human dataset
    data = np.load(data_in_path, allow_pickle=True)

    pipe, model, tokenizer = get_pipe()

    start_prompt = """
    You will see a stimulus card and must choose which of four key cards it matches. Cards can match by one of three categories: color, form, or number. The matching category changes from time to time. After each choice, you will receive a feedback stimulus:
    - "REPEAT" means you used the correct category and should keep using it.
    - "SWITCH" means you used the wrong category and should try a different one.

    The four key cards are always:
    A: one red triangle
    B: two green stars
    C: three yellow crosses
    D: four blue balls

    Each stimulus card shares at most one property (color, form, or number) with any one key card.
    Your task is to use the feedbacks to figure out the correct temporary category to apply and respond accordingly pressing key 'A' or 'B' or 'C' or 'D'.
    """

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

    def step_to_prompt(step):
        s = f"""You see the following stimulus card: {step['stimulus'][2]} {step['stimulus'][0]} {step['stimulus'][1]}. You press key <<"""
        return s

    def finish_step_prompt(step, answer):
        answer_to_lett = NUM2LETTER[answer]
        s = f"""{answer_to_lett}>> ({step['key_cards'][answer][2]} {step['key_cards'][answer][0]} {step['key_cards'][answer][1]}).
    You get the following feedback stimulus: {"REPEAT" if answer == step['ground_key'] else "SWITCH"}."""
        return s

    # ----------- init variables and dictionaries ------------
    human_correct = defaultdict(list)
    centaur_correct = defaultdict(list)
    centaur_aligned = defaultdict(list)
    centaur_log_likelihoods = defaultdict(list)

    letter_token_ids = {k: tokenizer(k, add_special_tokens=False)['input_ids'][0] for k in
                        "ABCD"}  # small Python dictionary that maps each of the four key-card letters to the single vocabulary-token ID that represents that character in the model tokenizer.

    dialogue = start_prompt.strip() + "\n\n"  # initialization of the dialogue with the task rules
    choices = set("ABCD")  # valid single-token answers

    current_id = None

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

        # ---------- new subject ----------
        if step['subject_id'] != current_id:
            if step['subject_id'] in processed_subjects:
                print(f"Skipping already processed subject {step['subject_id']}")
                continue
            if step['subject_id'] > LAST_SUBJECT_ID:
                break
            if current_id is not None:
                np.savez(
                    os.path.join(CHECKPOINT_DIR, f"{current_id}_results.npz"),
                    human_correct=human_correct[current_id],
                    centaur_correct=centaur_correct[current_id],
                    centaur_aligned=centaur_aligned[current_id],
                    centaur_log_likelihoods=centaur_log_likelihoods[current_id],
                )

            # ----------- new seed ---------------
            seed = 10_000 + step['subject_id']
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # ----------- reset state ------------
            dialogue = start_prompt.strip() + "\n\n"
            current_id = step['subject_id']

        # ----------- human correctness ------------
        human_is_ok = int(step['user_key'] == step['ground_key'])
        human_correct[current_id].append(human_is_ok)

        # ----------- Centaur choice ------------
        dialogue += step_to_prompt(step)
        inputs = tokenizer(dialogue, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1]  # unnormalised log-probabilities for the next token , so we select the last position [-1]

        probs = F.softmax(logits, dim=-1)

        # get predicted token
        pred_token_id = torch.argmax(
            probs).item()  # greedy decoding: the token with highest probability is Centaur’s predicted key for this trial.
        centaur_key = None
        for k, v in letter_token_ids.items():  # map token-ID → key letter → numeric code 1-4
            if v == pred_token_id:
                centaur_key = LETTER2NUM[k]
                break

        centaur_is_ok = int(centaur_key == step['ground_key']) if centaur_key is not None else 0
        centaur_is_aligned = int(centaur_key == step['user_key'])

        centaur_correct[current_id].append(centaur_is_ok)
        centaur_aligned[current_id].append(centaur_is_aligned)

        # ----- compute log-likelihood of actual human choice -----
        true_letter = NUM2LETTER[step['user_key']]  # convert the human key (1-4) to its letter, then to its token-ID.
        true_token_id = letter_token_ids[true_letter]
        ll = torch.log(probs[true_token_id] + 1e-8).item()  # extract the model’s probability for that token and take log.
        centaur_log_likelihoods[current_id].append(ll)

        # ---- conclusion of the step prompt -----
        dialogue += finish_step_prompt(step, step['user_key']) + "\n"

    # save data 
    if current_id is not None and current_id not in processed_subjects:
        np.savez(
            os.path.join(CHECKPOINT_DIR, f"{current_id}_results.npz"),
            human_correct=human_correct[current_id],
            centaur_correct=centaur_correct[current_id],
            centaur_aligned=centaur_aligned[current_id],
            centaur_log_likelihoods=centaur_log_likelihoods[current_id],
        )
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "*_results.npz"))

    human_correct = defaultdict(list)
    centaur_correct = defaultdict(list)
    centaur_aligned = defaultdict(list)
    centaur_log_likelihoods = defaultdict(list)

    for f in files:
        sid = int(os.path.basename(f).split("_")[0])
        d = np.load(f, allow_pickle=True)
        human_correct[sid] = d["human_correct"]
        centaur_correct[sid] = d["centaur_correct"]
        centaur_aligned[sid] = d["centaur_aligned"]
        centaur_log_likelihoods[sid] = d["centaur_log_likelihoods"]

    np.save("data/out/centaur-70B/predictive/human_correct_predictive.npy", human_correct)
    np.save("data/out/centaur-70B/predictive/centaur_correct_predictive.npy", centaur_correct)
    np.save("data/out/centaur-70B/predictive/centaur_aligned_predictive.npy", centaur_aligned)
    np.save("data/out/centaur-70B/predictive/centaur_log_likelihoods_predictive.npy", centaur_log_likelihoods)


if __name__ == "__main__":
    main()