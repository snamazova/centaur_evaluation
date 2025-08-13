import transformers
import pandas as pd
import numpy as np
import random
import torch
import get_models
import os
import gc


MODEL = 'centaur-70B'
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

def format_past_trials(past_trials: list) -> str:
    """Formats past trial data for the prompt by listing choice, reward, and cumulative reward."""
    return "".join(
        f"Trial {trial['trial_num']}:Choice {trial['choice']}  → {trial['reward']} points, "
        for trial in past_trials
    )

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

def generate_timeline(num_trials=100, seed=42):
    """Generates a timeline of trials for the slot machine task.

    Args:
        num_trials: The number of trials to generate.
        seed: The initial seed for the random number generator (for reproducibility).

    Returns:
        A DataFrame containing the trial data with columns: 'trial', 'choice', 'reward'.
    """
    random.seed(42)

    # Number of trials
    num_trials = 100

    # Define the timeline
    timeline = []
    for i in range(num_trials):
        while True:
            if i < (num_trials / 2):
                bandit_1_reward = random.choices([1, 0], weights=[0.8, 0.2])[0]
                bandit_2_reward = random.choices([1, 0], weights=[0.2, 0.8])[0]
            else:
                bandit_1_reward = random.choices([1, 0], weights=[0.2, 0.8])[0]
                bandit_2_reward = random.choices([1, 0], weights=[0.8, 0.2])[0]

            if not (bandit_1_reward == 0 and bandit_2_reward == 0):
                break

        timeline.append({
            "bandit_1": {"color": "orange", "value": bandit_1_reward},
            "bandit_2": {"color": "blue", "value": bandit_2_reward}
        })
    return timeline
    
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

def simulate_participant(timeline:list,pipe:transformers.pipeline) -> pd.DataFrame:
    """Simulates a participant with log-likelihood tracking"""
    history = []
    cumulative_reward = 0
    total_trials = 100


    for trial in range(1,total_trials+1):
        current_trial_data = timeline[trial - 1]  # Ensure `timeline` is defined
        prompt_model = build_slot_prompt(trial, history, total_trials)
        bandit_1_value = current_trial_data["bandit_1"]["value"]
        bandit_2_value = current_trial_data["bandit_2"]["value"]
        #print(f"this is {prompt_model}")
        choice_raw = generate(prompt_model,pipe)
        #print(f"this is choice raw {choice_raw}")
        model_choice = extract_model_choice(choice_raw)
        print(f"this is model choice {model_choice}")

        # Determine reward
        reward = bandit_1_value if model_choice == 'U' else (bandit_2_value if model_choice == 'P' else 0)
        cumulative_reward += reward
        #outputs=generate_test(prompt_model,pipe)
        #print(f"this is whole output {outputs}")

        print(f"Trial {trial}: "
              f"Choice {model_choice}, "
              f"Reward {reward}, "
              #f"Reasoning {model_reasoning} "
              f"Total {cumulative_reward}")

        history.append({
            "trial_num": trial,
            "prompt": prompt_model,
            "choice": model_choice,
            "reward": reward,
            "cumulative_reward": cumulative_reward,
        })



    return pd.DataFrame(history)



def main():

    if not os.path.exists(DATA_FOLDER_OUT):
        os.makedirs(DATA_FOLDER_OUT)

    seeds = generate_seeds(num_seeds=32)
    timeline = generate_timeline(num_trials=100)
    # Initialize new model for each seed
    model,tokenizer = get_models.get_model_no_pipe(MODEL)
    model._past = None  # Reset past states if necessary
    torch.cuda.empty_cache()  # Clear GPU memory again
    pipe=create_text_generation_pipeline(model,tokenizer,max_new_tokens=1)
    # Run simulation for each seed
    for run_id, seed in enumerate(seeds):
        out_path = f'{DATA_FOLDER_OUT}/participant_' + str(seed) + '.csv'
        gc.collect()
        torch.cuda.empty_cache()
        fix_seed(seed)  # Ensure reproducibility
        torch.cuda.empty_cache()  # Clear GPU memory before loading model
        # Run simulation
        history = simulate_participant(timeline,pipe)
        # Save results
        history.to_csv(out_path, index=False)
        # Cleanup: delete model and clear memory
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
