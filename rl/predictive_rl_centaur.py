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
DATA_FOLDER_OUT = f'data/out/predictive/{MODEL}/singles'

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

def simulate_participant(df_participant: pd.DataFrame, model, tokenizer, pipe, letter_token_ids):
    """Simulates a participant with log-likelihood tracking"""
    history = []
    cumulative_reward = 0
    total_trials = len(df_participant)


    for trial in range(total_trials):
        row = df_participant.iloc[trial]
        trial_num = row['trial']
        human_choice = row['choice']
        reward = row['reward']
        cumulative_reward += reward

        # Build prompt using actual human history
        past_trials = []
        for past_idx in range(trial):
            past_row = df_participant.iloc[past_idx]
            past_trials.append({
                "trial": past_row['trial'],
                "choice": past_row['choice'],
                "reward": past_row['reward'],
                "cumulative_reward": df_participant.iloc[:past_idx+1]['reward'].sum()
            })

        prompt = build_slot_prompt(trial_num, past_trials, total_trials)

         # --- Run model on prompt ---
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0, -1]
        probs = F.softmax(logits, dim=-1)
        pred_token_id = torch.argmax(probs).item()
        # Top-k tokens
        # Top-2 tokens and their probabilities
        topk = torch.topk(probs, k=2)
        top2_tokens = []
        for idx, prob in zip(topk.indices, topk.values):
            decoded_token = tokenizer.decode(idx.item()).strip()
            top2_tokens.append({
                "token": decoded_token,
                "prob": prob.item()
            })


        # --- Map to model choice (U or P) ---
        token_id_to_letter = {v: k for k, v in letter_token_ids.items()}
        model_choice = token_id_to_letter.get(pred_token_id, "INVALID")
        # --- Log-likelihood of human’s actual choice ---
        log_likelihood = None
        if human_choice in letter_token_ids:
            user_token_id = letter_token_ids[human_choice]
            log_likelihood = torch.log(probs[user_token_id] + 1e-8).item() # Fixed: Use token ID for indexing

        # --- Get the actual reward (from human data) ---
        # The reward is already taken from the dataframe at the beginning of the loop
        # reward = df.loc[df["trial"] == trial, "reward"].values[0]
        # cumulative_reward += reward # Cumulative reward is updated earlier

        history.append({
            "trial_num": trial,
            "prompt": prompt,
            "model_choice": model_choice,
            "human_choice": human_choice,
            "reward": reward,
            "cumulative_reward": cumulative_reward,
            "log_likelihood": log_likelihood,
            "top2_tokens": top2_tokens
        })
        print(f"Trial {trial}: Human {human_choice}, Model {model_choice}, LL: {log_likelihood}")

    return pd.DataFrame(history)



def main():

    if not os.path.exists(DATA_FOLDER_OUT):
        os.makedirs(DATA_FOLDER_OUT)

    seeds = generate_seeds(num_seeds=32)

    model, tokenizer = get_models.get_model_no_pipe(MODEL)
    pipe = create_text_generation_pipeline(model, tokenizer, max_new_tokens=1)

    timeline = pd.read_csv(DATA_IN_TEST)
    timeline['choice'] = timeline['choice'].map({0: 'U', 1: 'P'})
    model_ids = timeline['model_id'].unique()

    fix_seed(seeds[0])

    letter_token_ids = {
    "U": tokenizer("U", add_special_tokens=False)['input_ids'][0],
    "P": tokenizer("P", add_special_tokens=False)['input_ids'][0],
}


    for model_id in model_ids:
        print(f"\n🧠 Simulating model {model_id}")
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
