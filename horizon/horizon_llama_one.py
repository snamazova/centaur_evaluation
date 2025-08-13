import transformers

import pandas as pd
import numpy as np
import random
import torch
import torch.nn.functional as F
import get_models
import os
import gc

DATA_IN_ = 'data/in/timeline_structure.csv'

MODEL = 'llama-8B'
DATA_FOLDER_OUT = f'data/out/{MODEL}/singles'

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

def generate(prompt: str, pipe: transformers.pipeline) -> str:
    """Generates a response from the model using the provided prompt.
    Args:
        prompt (str): The input prompt for the model.
        pipe (transformers.pipeline): The text generation pipeline.
    Returns:
        str: The generated text response from the model.
    """
    return pipe(prompt)[0]['generated_text'][len(prompt):]

# Modified simulation function
def simulate_participant_by_block(timeline_df, participant_id, seeds,model,tokenizer,pipe):
    """Simulates a participant's choices and rewards in the slot machine task."""
    participant_seed = seeds[int(participant_id - 1)]
    fix_seed(participant_seed)
    all_rows = []  # Final results will be collected here

    # Filter timeline for this participant
    participant_df = timeline_df[timeline_df['participant_id'] == participant_id]

    for block_num in sorted(participant_df['block'].unique()):
        block_df = participant_df[participant_df['block'] == block_num]

        for game in sorted(block_df['game'].unique()):
            game_df = block_df[block_df['game'] == game].sort_values('trial_num_block')

            # === extract metadata
            reward_means = {
                "H": game_df["reward_mean_H"].iloc[0],
                "I": game_df["reward_mean_I"].iloc[0]
            }
            horizon = game_df["horizon"].iloc[0]
            info_condition = game_df["info_condition"].iloc[0]

            forced_df = game_df[game_df["type"] == "forced"]
            free_df = game_df[game_df["type"] == "free"]

            cumulative_reward = forced_df["reward"].sum()

            # Start building simulated game DataFrame with forced trials
            simulated_game_df = forced_df.copy()

            # Append forced trials to all_rows
            for _, row in forced_df.iterrows():
                all_rows.append({
                    "participant_id": participant_id,
                    "block": block_num,
                    "game": game,
                    "horizon": horizon,
                    "info_condition": info_condition,
                    "reward_mean_H": reward_means["H"],
                    "reward_mean_I": reward_means["I"],
                    "trial_num": row["trial_num_block"],
                    "choice": row["choice"],
                    "reward": row["reward"],
                    "cumulative_reward": cumulative_reward,
                    "is_free": False
                })

            # Simulate free-choice trials
            for _, row in free_df.iterrows():
                current_trial = row["trial_num_block"]
                game=row['game']
                m1=free_df[free_df['trial_num_block']==current_trial]['reward_mean_H'].values[0]
                m2=free_df[free_df['trial_num_block']==current_trial]['reward_mean_I'].values[0]
                past_forced_df = simulated_game_df[simulated_game_df["type"] == "forced"]
                past_free_df = simulated_game_df[
                    (simulated_game_df["type"] == "free") &
                    (simulated_game_df["trial_num_block"] < current_trial)
                ]
                forced_trials_text = format_forced_trials(past_forced_df)
                free_trials_text = format_past_trials(past_free_df)

                prompt = build_slot_prompt_llama(
                      current_trial=current_trial,
                      past_trials=free_trials_text,
                      forced_choices=forced_trials_text,
                      total_trials=len(game_df),
                      game_number=game
                  )

                model_choice = generate(prompt, pipe)
                if m1>m2 and model_choice=="H":
                  is_optimal=True
                elif m1<m2 and model_choice=="I":
                  is_optimal=True
                else:
                  is_optimal=False

                # Update cumulative reward
                reward_h = row["reward_H"]
                reward_i = row["reward_I"]

                reward = reward_h if model_choice == "H" else reward_i if model_choice == "I" else None
                cumulative_reward += reward

                #print(f"this is model choice {model_choice} and it is reward{cumulative_reward}")

                # Create new simulated row and append to local game history
                simulated_row = row.copy()
                simulated_row["choice"] = model_choice
                simulated_row["reward"] = reward
                simulated_row["cumulative_reward"] = cumulative_reward
                simulated_row["prompt"] = "".join(str(p) for p in prompt)

                simulated_game_df = pd.concat([simulated_game_df, pd.DataFrame([simulated_row])])

                all_rows.append({
                    "participant_id": participant_id,
                    "block": block_num,
                    "game": game,
                    "horizon": horizon,
                    "info_condition": info_condition,
                    "reward_mean_H": reward_means["H"],
                    "reward_mean_I": reward_means["I"],
                    "trial_num": simulated_row["trial_num_block"],
                    "choice": model_choice,
                    "reward": reward,
                    "cumulative_reward": cumulative_reward,
                    "prompt": simulated_row["prompt"],
                    "is_optimal": is_optimal,
                    "is_free": True
                })

                #print(f"Prompt for trial {current_trial}:")
                #print("".join(prompt))
                print(f"Model choice: {model_choice} and it is optimal {is_optimal}")
                #print("---")
                print(f"cumulative reward {cumulative_reward}")

    print(f"✅ Simulated participant {participant_id}.")
    return pd.DataFrame(all_rows)



def main():

    if not os.path.exists(DATA_FOLDER_OUT):
        os.makedirs(DATA_FOLDER_OUT)

    seeds = generate_seeds(num_seeds=32)
    timeline = pd.read_csv(DATA_IN_)
    participant_ids = timeline['participant_id'].unique()

            # Initialize new model for each seed
    model,tokenizer = get_models.get_model_no_pipe(MODEL)
    model._past = None  # Reset past states if necessary
    torch.cuda.empty_cache()  # Clear GPU memory again
    pipe=create_text_generation_pipeline(model,tokenizer,max_new_tokens=1)

    # Run simulation for each seed
    for participant_id in participant_ids:
        # Set output path for this participant
        out_path = f'{DATA_FOLDER_OUT}/participant_' + str(participant_id) + '.csv'
        result=[]
        if os.path.exists(out_path):
            print(f"Participant {participant_id} already simulated. Skipping...")
            continue

        
        print(f"\n🧠 Simulating participant {participant_id}")

        # 🔁 Re-initialize model and pipeline for each participant
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Re-initialize the model and tokenizer
        # Set the seed for reproducibility
        # Use the seed corresponding to the participant_id


        seed_id= seeds[int(participant_id - 1)]
        fix_seed(seed_id)
        # Run simulation
        # Run participant simulation
        participant_data = timeline[timeline['participant_id'] == participant_id]
        result = simulate_participant_by_block(participant_data, participant_id,seeds, model, tokenizer, pipe)
            # Save results
        result.to_csv(out_path, index=False)


        # Cleanup: delete model and clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
