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

def build_game_intro(timeline_df, game_number):
    """Build the introduction part of the prompt for a game."""
    total_trials = timeline_df['trial_num_block'].max()
    intro = [
            f"You are participating in multiple games involving two slot machines, labeled I and H.",
            "The two slot machines are different across different games.",
            "Each time you choose a slot machine, you get some points.",
            "You choose a slot machine by pressing the corresponding key.",
            "Each slot machine tends to pay out about the same amount of points on average.",
            "Your goal is to choose the slot machines that will give you the most points across the experiment.",
            "The first 4 trials in each game are instructed trials where you will be told which slot machine to choose.",
            "After these instructed trials, you will have the freedom to choose for either 1 or 6 trials.",
            f" Game {game_number}. There are {total_trials} trials in this game."
        ]
    return intro

def format_forced_trials(forced_df):
    """Format instructed (forced) trials for the prompt."""
    trials_text = []
    for _, row in forced_df.iterrows():
        trials_text.append(f"You are instructed to press {row['choice']} and get {row['reward']} points.")
    return trials_text

def format_past_trials(past_df):
    """Format past free-choice trials for the prompt."""
    trials_text = []
    for _, row in past_df.iterrows():
        trials_text.append(f"You press <<{row['choice']}>> and get {row['reward']} points.")
    return trials_text

def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    transformers.set_seed(seed)  # For Hugging Face models

def generate(prompt, pipe):
    return pipe(prompt)[0]['generated_text'][len(prompt):]

# Modified simulation function
import pandas as pd
import torch
import torch.nn.functional as F

def simulate_participant_by_block(timeline_df, pipe, participant_id, model, tokenizer, letter_token_ids):
    """
    Simulates a participant's slot machine behavior using the Centaur model.
    Now also extracts logits, predicted tokens, probs, and log-likelihoods of human choice.
    """
    all_rows = []

    # Filter timeline for this participant
    participant_df = timeline_df[timeline_df['participant_id'] == participant_id]

    for block_num in sorted(participant_df['block'].unique()):
        block_df = participant_df[participant_df['block'] == block_num]

        for game in sorted(block_df['game'].unique()):
            game_df = block_df[block_df['game'] == game].sort_values('trial')

            reward_means = {
                "H": game_df["m1"].iloc[0],
                "I": game_df["m2"].iloc[0]
            }
            horizon = game_df["horizon"].iloc[0]
            info_condition = game_df["uc"].iloc[0]

            cumulative_reward = game_df[game_df["type"] == "forced"]["reward"].sum()

            for _, row in game_df.iterrows():
                trial = row["trial"]
                is_free = row["type"] == "free"
                human_choice = row["choice"]

                model_choice = None
                pred_token_id, log_likelihood = None, None

                if is_free:
                    # --- Prompt construction ---
                    game_intro = build_game_intro(game_df, game)
                    forced_trials_text = format_forced_trials(game_df[game_df["type"] == "forced"])
                    past_free_df = game_df[(game_df["type"] == "free") & (game_df["trial"] < trial)]
                    free_trials_text = format_past_trials(past_free_df)
                    prompt = str(game_intro + forced_trials_text + free_trials_text) + "You press <<"

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

    print(f"✅ Simulated participant {participant_id}.")
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
