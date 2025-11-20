import transformers

import pandas as pd
import numpy as np
import random
import torch
import torch.nn.functional as F
import get_models
import os

DATA_IN_TEST = 'data/in/test_data.csv'

MODEL = 'centaur-8B'
DATA_FOLDER_OUT = f'data/out/predictive_masking_with_prompt/{MODEL}/singles'

def generate_seeds(num_seeds=20, seed=42):
    """Generates a list of random seeds.

    Args:
        num_seeds: The number of seeds to generate.
        seed: The initial seed for the random number generator (for reproducibility).

    Returns:
        A list of random integer seeds.
    """
    random.seed(seed)  # Set initial seed for reproducibility
    seeds = [random.randint(1, 100000) for _ in range(num_seeds)]
    return seeds


def create_text_generation_pipeline(model, tokenizer, temperature=1.0, max_new_tokens=1):
    """
    Creates a text-generation pipeline with the given model and tokenizer.

    Args:
        model: The preloaded model for text generation.
        tokenizer: The corresponding tokenizer.
        temperature (float): Sampling temperature for generation (default: 1.0).
        max_new_tokens (int): Maximum number of tokens to generate (default: 1024).

    Returns:
        A transformers pipeline object for text generation.
    """
    return transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        pad_token_id=0,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )


def extract_model_choice(raw_response: str) -> str:
    """
    Extracts choice from model's raw response text
    Handles common variations while maintaining strict validation
    """
    # Clean up the raw response by removing leading/trailing whitespace and quotes
    cleaned_response = raw_response.strip().strip('"')
    return cleaned_response

def build_slot_prompt(current_trial: int, past_trials: list, total_trials: int) -> str:
    """Builds the prompt for the current trial with past trial data."""
    recent_trials = past_trials
    prompt = (
              "In this task, you have to repeatedly choose between two slot machines labeled U and P.\n"
              "You can choose a slot machine by pressing its corresponding key."
              "When you select one of the machines, you will win 1 or 0 points."
              "Your goal is to choose the slot machines that will give you the most points."
              "You will receive feedback about the outcome after making a choice.\n"
              "The environment may change unpredictably, and past success does not guarantee future results. You’ll need to adapt to these changes to keep finding the better machine."
              f"You will play 1 game in total, consisting of {total_trials} trials."
            f" Game 1:"
    )

    # Add history of past trials to the prompt
    for past_trial in recent_trials:
        prompt += f"You press <<{past_trial['choice']}> and get {past_trial['reward']} points.\n"

    # Add the current choice prompt
    prompt += f"You press <<"
    return prompt


def fix_seed(seed: int):
    """Fixes the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    transformers.set_seed(seed)  # For Hugging Face models

def generate(prompt: str, pipe: transformers.pipeline) -> str:
    """Generates a response from the model using the provided prompt.
    Args:
        prompt (str): The input prompt for the model.
        pipe (transformers.pipeline): The text generation pipeline.
    Returns:
        str: The generated text response from the model.
    """
    return pipe(prompt)[0]['generated_text'][len(prompt):]

def mask_and_predict_final_token(prompt, model, tokenizer, letter_token_ids, baseline_U, baseline_P):
    """
    Computes masking deltas using pre-computed baselines.
    """
    
    # Tokenize prompt into list of tokens
    encoding = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    token_ids = encoding["input_ids"][0]

    results = []

    for pos in range(len(token_ids)):
        original_token_id = token_ids[pos].item()
        original_token = tokenizer.decode([original_token_id]).strip()

        # ---------- CREATE MASKED PROMPT ----------
        masked_ids = torch.cat([
            token_ids[:pos],
            token_ids[pos+1:]
        ])

        masked_inputs = {
            "input_ids": masked_ids.unsqueeze(0).to(model.device)
        }

        # ---------- RUN MODEL ON MASKED PROMPT ----------
        with torch.no_grad():
            masked_outputs = model(**masked_inputs)
        logits = masked_outputs.logits[0, -1]
        probs = F.softmax(logits, dim=-1)

        prob_U = probs[letter_token_ids["U"]].item()
        prob_P = probs[letter_token_ids["P"]].item()

        results.append({
            "masked_position": pos,
            "original_token": original_token,
            "prob_U_masked": prob_U,
            "prob_P_masked": prob_P,
            "delta_U": prob_U - baseline_U, # Uses passed baseline_U
            "delta_P": prob_P - baseline_P, # Uses passed baseline_P
        })

    return results

def simulate_participant(df_participant, model, tokenizer, pipe, letter_token_ids, 
                         save_masking=True, model_id=None):

    history = []
    masking_history = []

    total_trials = len(df_participant)
    cumulative_reward = 0

    for i in range(total_trials):
        row = df_participant.iloc[i]
        human_choice = row['choice']
        reward = row['reward']
        cumulative_reward += reward

        # Build prompt
        past_trials = []
        for j in range(i):
            past_row = df_participant.iloc[j]
            past_trials.append({
                "trial": past_row['trial'],
                "choice": past_row['choice'],
                "reward": past_row['reward'],
            })

        prompt = build_slot_prompt(i, past_trials, total_trials)

        # ----------------------------------------------------
        # BASELINE PREDICTION (The Normal Run - Performed ONCE)
        # ----------------------------------------------------
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1]
        probs = F.softmax(logits, dim=-1)
        
        # Get baseline probs for U and P
        baseline_U = probs[letter_token_ids["U"]].item()
        baseline_P = probs[letter_token_ids["P"]].item()

        # Get Model Choice and Log-Likelihood
        token_id_to_letter = {v: k for k, v in letter_token_ids.items()}
        pred_token_id = torch.argmax(probs).item()
        model_choice = token_id_to_letter.get(pred_token_id, "INVALID")

        # Log-likelihood of human choice
        user_token_id = letter_token_ids[human_choice]
        log_likelihood = torch.log(probs[user_token_id] + 1e-8).item()
        
        # Top-2 tokens and their probabilities (for history logging)
        topk = torch.topk(probs, k=2)
        top2_tokens = []
        for idx, prob in zip(topk.indices, topk.values):
            decoded_token = tokenizer.decode(idx.item()).strip()
            top2_tokens.append({
                "token": decoded_token,
                "prob": prob.item()
            })


        # ----------------------------------------------------
        # MASKING ANALYSIS (Passes Baseline)
        # ----------------------------------------------------
        masking_results = mask_and_predict_final_token(
            prompt, model, tokenizer, letter_token_ids,
            baseline_U, baseline_P # Pass the computed baselines
        )

        if save_masking:
            for r in masking_results:
                r["trial_num"] = i
                r["model_id"] = model_id
                # Add baseline probs to masking history for completeness
                r["baseline_U"] = baseline_U 
                r["baseline_P"] = baseline_P
                masking_history.append(r)

        # Save trial-level history
        history.append({
            "trial_num": i,
            "prompt": prompt, # NOW SAVING THE PROMPT HERE
            "model_choice": model_choice,
            "human_choice": human_choice,
            "reward": reward,
            "cumulative_reward": cumulative_reward,
            "log_likelihood": log_likelihood,
            "prob_U_baseline": baseline_U, # Also saving baseline for analysis
            "prob_P_baseline": baseline_P, # Also saving baseline for analysis
            "top2_tokens": top2_tokens
        })

        print(f"Trial {i}: Human {human_choice} | Model {model_choice} | LL={log_likelihood:.4f}")

    return pd.DataFrame(history), pd.DataFrame(masking_history)
def main():

    if not os.path.exists(DATA_FOLDER_OUT):
        os.makedirs(DATA_FOLDER_OUT)

    seeds = generate_seeds(num_seeds=32)

    model, tokenizer = get_models.get_model_no_pipe(MODEL)
    pipe = create_text_generation_pipeline(model, tokenizer, max_new_tokens=1)

    timeline = pd.read_csv(DATA_IN_TEST)
    timeline['choice'] = timeline['choice'].map({0: 'U', 1: 'P'})
    model_ids = timeline['model_id'].unique()
    test_model_id=model_ids[0]

    fix_seed(seeds[0])

    letter_token_ids = {
    "U": tokenizer("U", add_special_tokens=False)['input_ids'][0],
    "P": tokenizer("P", add_special_tokens=False)['input_ids'][0],
}


    for model_id in [test_model_id]:
        print(f"Simulating model {model_id}")
        out_path = f'{DATA_FOLDER_OUT}/model_' + str(model_id) + '.csv'

        if os.path.exists(out_path):
            print(f"Model {model_id} already simulated. Skipping...")
            continue

        # Run simulation with model and tokenizer passed
        model_data = timeline[timeline['model_id'] == model_id]
        result = simulate_participant(model_data, model, tokenizer, pipe, letter_token_ids)
        result.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
