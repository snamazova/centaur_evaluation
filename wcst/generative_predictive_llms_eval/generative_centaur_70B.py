import os
import glob

from collections import defaultdict
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm
import random, torch

import numpy as np

LAST_SUBJECT_ID = 88  # last subject ID in the evaluation dataset, used to stop the loop when reached

MODEL_PATH = "models/models--marcelbinz--Llama-3.1-Centaur-70B/snapshots/"
DATA_OUT_ROOT = "data/out/centaur-70B/generative/"
CHECKPOINT_DIR = f"{DATA_OUT_ROOT}/checkpoints"


def get_pipe(
        path=MODEL_PATH,):
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
        do_sample=True,
        temperature=1.0,
        max_new_tokens=1,
    )
    return pipe, model, tokenizer


def main():

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    processed_subjects = {
        int(os.path.basename(f).split('_')[0])
        for f in glob.glob(f"{CHECKPOINT_DIR}/*_results.npz")
    }

    data_in_path = "../Data/json_wcst_dataset.npy"  # human dataset
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
        answer_key_num = LETTER2NUM.get(answer.upper(), None)
        s = f"""{answer}>> ({step['key_cards'][answer_key_num][2]} {step['key_cards'][answer_key_num][0]} {step['key_cards'][answer_key_num][1]}).
    You get the following feedback stimulus: {"REPEAT" if answer_key_num == step['ground_key'] else "SWITCH"}."""
        return s

    # ----------- init variables and dictionaries ------------
    human_correct = defaultdict(list)
    centaur_correct = defaultdict(list)
    centaur_aligned = defaultdict(list)

    centaur_perseverance_err = defaultdict(list)
    centaur_setloss_err = defaultdict(list)

    human_perseverance_err = defaultdict(list)
    human_setloss_err = defaultdict(list)

    dialogue = start_prompt.strip() + "\n\n" # initialization of the dialogue with the task rules
    choices = set("ABCD")  # valid single-token answers

    current_id = None

    previous_centaur_rule_type = None
    previous_centaur_correct = None

    previous_human_rule_type = None
    previous_human_correct = None

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
                # Save results for completed subject
                np.savez(
                    f"{CHECKPOINT_DIR}/{current_id}_results.npz",
                    human_correct=human_correct[current_id],
                    centaur_correct=centaur_correct[current_id],
                    centaur_aligned=centaur_aligned[current_id],
                    centaur_perseverance_err=centaur_perseverance_err[current_id],
                    centaur_setloss_err=centaur_setloss_err[current_id],
                    human_perseverance_err=human_perseverance_err[current_id],
                    human_setloss_err=human_setloss_err[current_id],
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

            previous_centaur_rule_type = None
            previous_centaur_correct = None
            previous_human_rule_type = None
            previous_human_correct = None

        # ----------- human correctness ------------
        human_is_ok = int(step['user_key'] == step['ground_key'])
        human_correct[current_id].append(human_is_ok)

        # ----------- Centaur choice ------------
        dialogue += step_to_prompt(step)
        out = pipe(dialogue, max_new_tokens=1, pad_token_id=0)[0]['generated_text'][
            -1]  # answer from Centaur is in letter code
        if out.upper() in LETTER2NUM:
            # convert the letter output to its corresponding numeric key using the LETTER2NUM dictionary
            centaur_key = LETTER2NUM[out.upper()]  # this is again in number code
        else:
            # handle cases where the model output is not a valid choice letter
            print(f"Warning: Model generated invalid output: {out}")
            centaur_key = None

        centaur_is_ok = int(centaur_key == step['ground_key'])
        centaur_is_aligned = int(centaur_key == step['user_key'])

        centaur_correct[current_id].append(centaur_is_ok)
        centaur_aligned[current_id].append(centaur_is_aligned)

        # ----------- detect the rule type used by centaur ------------
        ref_card_centaur = step['key_cards'][centaur_key]
        stimulus_card = step['stimulus']
        if ref_card_centaur[0] == stimulus_card[0]:
            centaur_rule_type = "color"
        elif ref_card_centaur[1] == stimulus_card[1]:
            centaur_rule_type = "form"
        elif ref_card_centaur[2] == stimulus_card[2]:
            centaur_rule_type = "number"
        else:
            centaur_rule_type = None

        # ----------- detect the rule type used by the human ------------
        ref_card_human = step['key_cards'][step['user_key']]
        stimulus_card = step['stimulus']
        if ref_card_human[0] == stimulus_card[0]:
            human_rule_type = "color"
        elif ref_card_human[1] == stimulus_card[1]:
            human_rule_type = "form"
        elif ref_card_human[2] == stimulus_card[2]:
            human_rule_type = "number"
        else:
            human_rule_type = None

        # ----------- detect centaur perseverance errors ------------
        if previous_centaur_correct == 0 and centaur_is_ok == 0:
            if centaur_rule_type == previous_centaur_rule_type:
                # perseverance error
                centaur_perseverance_err[current_id].append(1)
            else:
                centaur_perseverance_err[current_id].append(0)
        else:
            centaur_perseverance_err[current_id].append(0)

        # ----------- detect human perseverance errors ------------
        if previous_human_correct == 0 and human_is_ok == 0:
            if human_rule_type == previous_human_rule_type:
                # perseverance error
                human_perseverance_err[current_id].append(1)
            else:
                human_perseverance_err[current_id].append(0)
        else:
            human_perseverance_err[current_id].append(0)

        # ----------- detect centaur set_loss errors ------------
        if previous_centaur_correct == 1 and centaur_is_ok == 0:
            if centaur_rule_type != previous_centaur_rule_type:
                # set_loss error
                centaur_setloss_err[current_id].append(1)
            else:
                centaur_setloss_err[current_id].append(0)
        else:
            centaur_setloss_err[current_id].append(0)

        # ----------- detect human set_loss errors ------------
        if previous_human_correct == 1 and human_is_ok == 0:
            if human_rule_type != previous_human_rule_type:
                # set_loss error
                human_setloss_err[current_id].append(1)
            else:
                human_setloss_err[current_id].append(0)
        else:
            human_setloss_err[current_id].append(0)

        # ----------- update variable ------------
        previous_centaur_rule_type = centaur_rule_type
        previous_centaur_correct = centaur_is_ok

        previous_human_rule_type = human_rule_type
        previous_human_correct = human_is_ok

        # ---- conclusion of the step prompt -----
        dialogue += finish_step_prompt(step, out) + "\n"

    # save data 
    if current_id is not None and current_id not in processed_subjects:
        np.savez(
            f"{CHECKPOINT_DIR}/{current_id}_results.npz",
            human_correct=human_correct[current_id],
            centaur_correct=centaur_correct[current_id],
            centaur_aligned=centaur_aligned[current_id],
            centaur_perseverance_err=centaur_perseverance_err[current_id],
            centaur_setloss_err=centaur_setloss_err[current_id],
            human_perseverance_err=human_perseverance_err[current_id],
            human_setloss_err=human_setloss_err[current_id],
        )
    result_files = glob.glob(f"{CHECKPOINT_DIR}/*_results.npz")

    human_correct = defaultdict(list)
    centaur_correct = defaultdict(list)
    centaur_aligned = defaultdict(list)
    centaur_perseverance_err = defaultdict(list)
    centaur_setloss_err = defaultdict(list)
    human_perseverance_err = defaultdict(list)
    human_setloss_err = defaultdict(list)

    for path in result_files:
        data = np.load(path, allow_pickle=True)
        subject_id = int(os.path.basename(path).split("_")[0])

        human_correct[subject_id] = data["human_correct"]
        centaur_correct[subject_id] = data["centaur_correct"]
        centaur_aligned[subject_id] = data["centaur_aligned"]
        centaur_perseverance_err[subject_id] = data["centaur_perseverance_err"]
        centaur_setloss_err[subject_id] = data["centaur_setloss_err"]
        human_perseverance_err[subject_id] = data["human_perseverance_err"]
        human_setloss_err[subject_id] = data["human_setloss_err"]

    np.save(f'{DATA_OUT_ROOT}/human_correct_generative.npy', human_correct)
    np.save(f'{DATA_OUT_ROOT}/centaur_correct_generative.npy', centaur_correct)
    np.save(f'{DATA_OUT_ROOT}/centaur_aligned_generative.npy', centaur_aligned)
    np.save(f'{DATA_OUT_ROOT}/centaur_perseverance_err.npy', centaur_perseverance_err)
    np.save(f'{DATA_OUT_ROOT}/centaur_setloss_err.npy', centaur_setloss_err)
    np.save(f'{DATA_OUT_ROOT}/human_perseverance_err.npy', human_perseverance_err)
    np.save(f'{DATA_OUT_ROOT}/human_setloss_err.npy', human_setloss_err)

if __name__ == "__main__":
    main()
