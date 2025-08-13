import transformers

import pandas as pd
import numpy as np
import random
import torch
import torch.nn.functional as F
import get_models
import os

DATA_IN_TEST = 'data/in/test_data.csv'

MODEL = 'llama-70B'
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

def format_forced_trials(forced_df) -> str:
    """Format forced (instructed) trials as readable prompt text."""
    lines = []
    for _, row in forced_df.iterrows():
        lines.append(f"Instructed: press {row['choice']} → {row['reward']} points.")
    return "\n".join(lines)


def format_past_trials(past_df) -> str:
    """Format past free-choice trials for the prompt."""
    lines = []
    for _, row in past_df.iterrows():
        lines.append(f"Choice: {row['choice']} → {row['reward']} points.")
    return "\n".join(lines)


def build_slot_prompt_llama(
    current_trial: int,
    past_trials: list,
    forced_choices: str,
    total_trials: int,
    game_number,
    total_games=320
) -> str:
    """
    Builds a slot task prompt that:
    1. Preserves task structure
    3. Encourages evidence-based exploration
    4. Helps model infer the optimal choice
    """


    # Format forced lines for outcome summary calculation
    # Show recent free choices
  
    return f"""<|begin_of_text|>

<|start_header_id|>system<|end_header_id|>

You are participating in multiple games involving two slot machines(labeled H and I).Your goal is to choose the slot machines that will give you the most points across the experiment
The two slot machines are different across different games.Each time you choose a slot machine, you get some points.
Each slot machine tends to pay out about the same amount of points on average.
The first 4 trials in each game are instructed trials where you will be told which slot machine to choose
After these instructed trials, you will have the freedom to choose for either 1 or 6 trials.

<|eot_id|>

<|start_header_id|>user<|end_header_id|>
# Task Parameters
- Game {game_number} of {total_games}
- Trial {current_trial} of {total_trials}
- Choose between: 'I' or 'H'

## Outcome History
**Instructed Trials:**
{forced_choices}

**Recent Free Choices:**
{past_trials}


# Instructions
Respond with **only** one character: 'I' or 'H'. No punctuation, no quotes, no explanation.

Answer:<|start_header_id|>assistant<|end_header_id|>
"""

def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    transformers.set_seed(seed)  # For Hugging Face models

def generate(prompt, pipe):
    # Convert the prompt list to a single string
    prompt_items = [str(item) if not isinstance(item, str) else item for item in prompt]
    prompt_str = "".join(prompt_items)
    return pipe(prompt_str)[0]['generated_text'][len(prompt_str):]

# Modified simulation function
def simulate_participant_by_block(timeline_df, pipe, participant_id, model, tokenizer, letter_token_ids):
    """Simulates a participant with log-likelihood tracking"""
    all_rows = []

    participant_df = timeline_df[timeline_df['participant_id'] == participant_id]

    for block_num in sorted(participant_df['block'].unique()):
        block_df = participant_df[participant_df['block'] == block_num]

        for game in sorted(block_df['game'].unique()):
            game_df = block_df[block_df['game'] == game].sort_values('trial')
            reward_means = {"H": game_df["m1"].iloc[0], "I": game_df["m2"].iloc[0]}
            horizon = game_df["horizon"].iloc[0]
            info_condition = game_df["uc"].iloc[0]
            forced_df = game_df[game_df["type"] == "forced"]
            free_df = game_df[game_df["type"] == "free"]
            cumulative_reward = forced_df["reward"].sum()

            # Process forced trials
            for _, row in forced_df.iterrows():
                trial = row["trial"]
                is_free = False
                all_rows.append({
                    "participant_id": participant_id,
                    "block": block_num,
                    "game": game,
                    "horizon": horizon,
                    "info_condition": info_condition,
                    "reward_mean_H": reward_means["H"],
                    "reward_mean_I": reward_means["I"],
                    "trial_num": row["trial"],
                    "choice": row["choice"],
                    "reward": row["reward"],
                    "cumulative_reward": cumulative_reward,
                    "is_free": False,
                    "log_likelihood": None  # Forced trials have no LL
                })

            # Process free-choice trials
            for _, row in free_df.iterrows():
                pred_token_id, log_likelihood = None, None
                trial = row["trial"]
                is_free = True
                current_trial = row["trial"]
                past_forced_df = game_df[game_df["type"] == "forced"]
                past_free_df = game_df[(game_df["type"] == "free") & (game_df["trial"] < current_trial)]

                forced_trials_text = format_forced_trials(past_forced_df)
                free_trials_text = format_past_trials(past_free_df)
                # Get human choice from data
                human_choice = row['choice']

                prompt = build_slot_prompt_llama(
                    current_trial=current_trial,
                    past_trials=free_trials_text,
                    forced_choices=forced_trials_text,
                    total_trials=len(game_df),
                    game_number=game
                )
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

                #print(f"Prompt for trial {current_trial}:")
                #print(prompt)

                # Generate model choice and log-likelihood
                model_choice = generate(
                    prompt, pipe
                )


                reward = row["reward"]
                cumulative_reward += reward
                all_rows.append({
                    "participant_id": participant_id,
                    "block": block_num,
                    "game": game,
                    "horizon": horizon,
                    "info_condition": info_condition,
                    "reward_mean_H": reward_means["H"],
                    "reward_mean_I": reward_means["I"],
                    "trial_num": trial,
                    "is_free": is_free,
                    "choice": human_choice,
                    "model_choice": model_choice,
                    "reward": reward,
                    "cumulative_reward": cumulative_reward,
                    #"logits": logits_list,
                    #"probs": probs_list,
                    #"pred_token_id": pred_token_id,
                    "log_likelihood": log_likelihood,
                    "top2_tokens": top2_tokens
                })
                print(f"Trial {trial}: Human {human_choice}, Model {model_choice}, LL:{log_likelihood}")

    print(f"Simulated participant {participant_id}.")
    return pd.DataFrame(all_rows)




def main():

    if not os.path.exists(DATA_FOLDER_OUT):
        os.makedirs(DATA_FOLDER_OUT)

    seeds = generate_seeds(num_seeds=32)

    model, tokenizer = get_models.get_model_no_pipe(MODEL)
    pipe = create_text_generation_pipeline(model, tokenizer, max_new_tokens=1)

    timeline = pd.read_csv(DATA_IN_TEST)

    participant_ids = timeline['participant_id'].unique()

    fix_seed(seeds[0])

    letter_token_ids = {
    "I": tokenizer("I", add_special_tokens=False)['input_ids'][0],
    "H": tokenizer("H", add_special_tokens=False)['input_ids'][0],
}


    for participant_id in participant_ids:
        print(f"\n🧠 Simulating participant {participant_id}")
        out_path = f'{DATA_FOLDER_OUT}/participant_' + str(participant_id) + '.csv'

        if os.path.exists(out_path):
            print(f"Participant {participant_id} already simulated. Skipping...")
            continue

        # Run simulation with model and tokenizer passed
        participant_data = timeline[timeline['participant_id'] == participant_id]
        result = simulate_participant_by_block(
            participant_data, pipe, participant_id, model, tokenizer, letter_token_ids
        )
        result.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
