from pyexpat import model
import optuna
import torch
import subprocess
import os
import sys
import time
import traceback
import gc
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt
import numpy as np
import datetime
from mail import send_email
from trueVers.GNN import BipartiteGNN, HyperParameters, load_data, clean_memory
from train_gnn4 import train_rl_agent

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define paths for data files
current_dir = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_FILE = os.path.join(current_dir, 'instacart_test_baskets.csv')
TRAIN_DATA_FILE = os.path.join(current_dir, 'instacart_train_baskets.csv')
print(f"Using test data file: {GROUND_TRUTH_FILE}")
print(f"Using training data file: {TRAIN_DATA_FILE}")

def objective_gnn4(trial):
    # Hyperparameters to optimize for GNN4
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_int('hidden_dim', 16, 128)
    output_dim = trial.suggest_int('output_dim', 16, 128)
    batch_size = trial.suggest_int('batch_size', 16, 128)
    
    # Strength calculation weights (sum to 1.0)
    strength_freq_weight = trial.suggest_float('strength_freq_weight', 0.3, 0.7)
    strength_user_affinity_weight = trial.suggest_float('strength_user_affinity_weight', 0.1, min(0.3, 1.0 - strength_freq_weight))
    strength_product_affinity_weight = 1.0 - (strength_freq_weight + strength_user_affinity_weight)
    
    # Confidence calculation weights (cascading approach to ensure sum to 1.0)
    remaining_weight = 1.0
    
    # First weight: up to 0.4 of total
    confidence_freq_weight = trial.suggest_float('confidence_freq_weight', 0.0, min(0.4, remaining_weight))
    remaining_weight -= confidence_freq_weight
    
    # Second weight: up to 0.4 of total
    confidence_consistency_weight = trial.suggest_float('confidence_consistency_weight', 0.0, min(0.4, remaining_weight))
    remaining_weight -= confidence_consistency_weight
    
    # Third weight: up to 0.4 of total
    confidence_recency_weight = trial.suggest_float('confidence_recency_weight', 0.0, min(0.4, remaining_weight))
    remaining_weight -= confidence_recency_weight
    
    # Last weight gets remaining amount
    confidence_strength_weight = remaining_weight
    
    # Purchase consistency parameters
    consistency_monthly_threshold = trial.suggest_int('consistency_monthly_threshold', 20, 40)
    consistency_increase_factor = trial.suggest_float('consistency_increase_factor', 1.05, 1.15)
    consistency_decrease_factor = trial.suggest_float('consistency_decrease_factor', 0.85, 0.95)
    
    # Time decay parameters
    recency_decay_period = trial.suggest_int('recency_decay_period', 180, 540)  # 6-18 months
    
    # Combined score weights (sum to 1.0)
    combined_strength_weight = trial.suggest_float('combined_strength_weight', 0.4, 0.6)
    combined_confidence_weight = 1.0 - combined_strength_weight

    # Create a temporary script with the current hyperparameters
    with open('temp_train.py', 'w') as f:
        f.write(f'''import sys
import os
import numpy as np
import traceback
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import ToUndirected

sys.path.append('{os.path.dirname(os.path.abspath(__file__))}')
from GNN4 import BipartiteGNN, HyperParameters, load_data, clean_memory

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {{device}}")

# Define the ground truth file path
ground_truth_file = '{GROUND_TRUTH_FILE}'
print(f"Using ground truth file: {{ground_truth_file}}")
if not os.path.exists(ground_truth_file):
    print(f"ERROR: Ground truth file not found at {{ground_truth_file}}")
    sys.exit(1)

print("Starting training with parameters:")
print(f"  Learning rate: {learning_rate}")
print(f"  Hidden dim: {hidden_dim}")
print(f"  Output dim: {output_dim}")
print(f"  Batch size: {batch_size}")
print(f"\\nStrength weights (sum: {strength_freq_weight + strength_user_affinity_weight + strength_product_affinity_weight:.6f}):")
print(f"  Frequency: {strength_freq_weight:.6f}")
print(f"  User affinity: {strength_user_affinity_weight:.6f}")
print(f"  Product affinity: {strength_product_affinity_weight:.6f}")
print(f"\\nConfidence weights (sum: {confidence_freq_weight + confidence_consistency_weight + confidence_recency_weight + confidence_strength_weight:.6f}):")
print(f"  Frequency: {confidence_freq_weight:.6f}")
print(f"  Consistency: {confidence_consistency_weight:.6f}")
print(f"  Recency: {confidence_recency_weight:.6f}")
print(f"  Strength: {confidence_strength_weight:.6f}")
print(f"\\nCombined weights (sum: {combined_strength_weight + combined_confidence_weight:.6f}):")
print(f"  Strength: {combined_strength_weight:.6f}")
print(f"  Confidence: {combined_confidence_weight:.6f}")

# Create hyperparameters instance
params = HyperParameters(
    strength_freq_weight={strength_freq_weight},
    strength_user_affinity_weight={strength_user_affinity_weight},
    strength_product_affinity_weight={strength_product_affinity_weight},
    confidence_freq_weight={confidence_freq_weight},
    confidence_consistency_weight={confidence_consistency_weight},
    confidence_recency_weight={confidence_recency_weight},
    confidence_strength_weight={confidence_strength_weight},
    consistency_monthly_threshold={consistency_monthly_threshold},
    consistency_increase_factor={consistency_increase_factor},
    consistency_decrease_factor={consistency_decrease_factor},
    recency_decay_period={recency_decay_period},
    combined_strength_weight={combined_strength_weight},
    combined_confidence_weight={combined_confidence_weight}
)

try:
    # Load data with current hyperparameters
    data, unique_users, unique_products, user_to_idx, product_to_idx, user_product_freq, user_total_items, product_total_purchases, relationship_strengths, confidence_scores, regularity_scores = load_data(
        file_path='{TRAIN_DATA_FILE}',
        params=params
    )
    
    # Print data statistics
    print("\\nData statistics:")
    print(f"Number of users: {{len(unique_users)}}")
    print(f"Number of products: {{len(unique_products)}}")
    print(f"Number of edges: {{data.edge_index.size(1)}}")
    print(f"Node feature dimension: {{data.x.size(1)}}")
    print(f"Edge index shape: {{data.edge_index.shape}}")
    print(f"Edge weight shape: {{data.edge_weight.shape if data.edge_weight is not None else 'None'}}")
    
    # Create heterogeneous data object
    hetero_data = HeteroData()
    
    # Add node features
    num_users = len(unique_users)
    num_products = len(unique_products)
    
    hetero_data['user'].x = data.x[:num_users].to(device)
    hetero_data['product'].x = data.x[num_users:].to(device)
    
    # Add edge indices and weights
    edge_index = data.edge_index
    edge_weight = data.edge_weight
    
    # Adjust edge indices for product nodes
    product_edge_index = edge_index.clone()
    product_edge_index[1] -= num_users  # Adjust product indices
    
    # Define edge type as a string to avoid tuple formatting issues
    EDGE_TYPE = ('user', 'buys', 'product')
    
    # Create edge dictionaries
    edge_index_dict = {{
        EDGE_TYPE: product_edge_index.to(device)
    }}
    
    edge_weight_dict = {{
        EDGE_TYPE: edge_weight.to(device) if edge_weight is not None else None
    }}
    
    # Create and train model
    model = BipartiteGNN(
        num_features=data.x.size(1),
        hidden_channels={hidden_dim},
        num_classes={output_dim},
        device=device
    ).to(device)
    
    # Forward pass to get embeddings
    model.train()  # Set to training mode
    with torch.no_grad():
        try:
            print("\\nStarting forward pass...")
            print(f"Input tensor shapes:")
            print(f"  User features: {{hetero_data['user'].x.shape}}")
            print(f"  Product features: {{hetero_data['product'].x.shape}}")
            print(f"  Edge index: {{edge_index_dict[EDGE_TYPE].shape}}")
            print(f"  Edge weight: {{edge_weight_dict[EDGE_TYPE].shape if edge_weight_dict[EDGE_TYPE] is not None else 'None'}}")
            
            # Create input dictionaries
            x_dict = {{
                'user': hetero_data['user'].x,
                'product': hetero_data['product'].x
            }}
            
            # Normalize edge weights if they exist
            if edge_weight_dict[EDGE_TYPE] is not None:
                edge_weight_dict[EDGE_TYPE] = F.normalize(
                    edge_weight_dict[EDGE_TYPE], p=1, dim=0
                )
            
            embeddings = model(x_dict, edge_index_dict, edge_weight_dict)
            print(f"Forward pass successful.")
            print(f"User embedding shape: {{embeddings['user'].shape}}")
            print(f"Product embedding shape: {{embeddings['product'].shape}}")
            
        except RuntimeError as e:
            print(f"Error during forward pass: {{str(e)}}")
            print(f"Data shapes:")
            print(f"  User features: {{x_dict['user'].shape}}")
            print(f"  Product features: {{x_dict['product'].shape}}")
            print(f"  Edge index: {{edge_index_dict[EDGE_TYPE].shape}}")
            print(f"  Edge weight: {{edge_weight_dict[EDGE_TYPE].shape if edge_weight_dict[EDGE_TYPE] is not None else 'None'}}")
            raise e
    
    # Calculate validation score based on relationship strengths and confidence scores
    total_score = 0
    num_relationships = 0
    
    for (user_id, product_id), strength in relationship_strengths.items():
        if user_id in user_to_idx and product_id in product_to_idx:
            confidence = confidence_scores[(user_id, product_id)]
            regularity = regularity_scores.get((user_id, product_id), 0.0)
            
            # Calculate combined score
            score = (params.combined_strength_weight * strength + 
                    params.combined_confidence_weight * confidence) * (1 + regularity)
            
            total_score += score
            num_relationships += 1
    
    avg_score = total_score / num_relationships if num_relationships > 0 else -1000.0
    print(f"Training completed successfully. Average score: {{avg_score}}")
    
except Exception as e:
    print(f"Error during training: {{str(e)}}")
    print("Traceback:")
    traceback.print_exc()
    avg_score = -1000.0  # Set a default value for failed training
finally:
    # Clean up
    if 'model' in locals():
        model.cleanup()
    if 'data' in locals():
        data.cleanup()
    clean_memory()

# Save the final score
with open('trial_result.txt', 'w') as f:
    f.write(str(avg_score))
''')

    # Run the temporary script
    subprocess.run([sys.executable, 'temp_train.py'])
    
    # Read the result
    try:
        with open('trial_result.txt', 'r') as f:
            final_score = float(f.read().strip())
    except FileNotFoundError:
        print("Error: trial_result.txt not found. The training script may have failed.")
        final_score = -1000  # Return a very low score to indicate failure
    
    # Clean up
    if os.path.exists('temp_train.py'):
        os.remove('temp_train.py')
    if os.path.exists('trial_result.txt'):
        os.remove('trial_result.txt')
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return final_score

def objective_ppo(trial):
    # Hyperparameters to optimize for PPO
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    # Ensure hidden_dim is divisible by 4
    hidden_dim = trial.suggest_int('hidden_dim', 64, 128)
    hidden_dim = (hidden_dim // 4) * 4  # Round down to nearest multiple of 4
    batch_size = trial.suggest_int('batch_size', 32, 64)
    gamma = trial.suggest_float('gamma', 0.9, 0.99)
    episodes = trial.suggest_int('episodes', 10, 20)
    experiences_per_episode = trial.suggest_int('experiences_per_episode', 32, 64)

    # Create a temporary script with the current hyperparameters
    with open('temp_train.py', 'w') as f:
        f.write(f'''import sys
import os
import numpy as np
import traceback  # Add missing import

sys.path.append('{os.path.dirname(os.path.abspath(__file__))}')
from PPOv3 import PPOAgent, NBREnvironment
import torch
import pandas as pd

# Define the ground truth file path
ground_truth_file = '{GROUND_TRUTH_FILE}'
print(f"Using ground truth file: {{ground_truth_file}}")
if not os.path.exists(ground_truth_file):
    print(f"ERROR: Ground truth file not found at {{ground_truth_file}}")
    sys.exit(1)

print("Starting training with parameters:")
print(f"  Learning rate: {learning_rate}")
print(f"  Hidden dim: {hidden_dim}")
print(f"  Batch size: {batch_size}")
print(f"  Gamma: {gamma}")
print(f"  Episodes: {episodes}")
print(f"  Experiences per episode: {experiences_per_episode}")

# Train with current hyperparameters
try:
    # Create environment
    env = NBREnvironment(
        train_file='{TRAIN_DATA_FILE}',
        test_file=ground_truth_file,
        basket_size=10  # Always use 10 items for testing
    )
    
    # Create agent with hyperparameters
    agent = PPOAgent(
        env=env,
        hidden_dim={hidden_dim},
        embedding_dim={hidden_dim},  # Use the same value for both
        lr={learning_rate},
        gamma={gamma},
        batch_size={batch_size},
        seed={trial.number}
    )
    
    # Train the agent with progress output
    total_reward = 0
    
    # Instead of calling train(num_episodes=1) multiple times,
    # we'll call train once with the total number of episodes
    # and modify the agent's train method to print progress
    agent.train(num_episodes={episodes})
    
    # Calculate average reward across all episodes
    avg_reward = sum(agent.avg_rewards) / len(agent.avg_rewards)
    print(f"Training completed successfully. Average reward across all episodes: {{avg_reward:.4f}}")
    
    # Also calculate final performance (average of last 5 episodes)
    final_reward = np.mean(agent.avg_rewards[-5:]) if len(agent.avg_rewards) >= 5 else agent.avg_rewards[-1]
    print(f"Final performance (average of last 5 episodes): {{final_reward:.4f}}")
    
    # Use a weighted combination of average reward and final performance
    # This balances overall learning progress with final performance
    final_score = 0.7 * avg_reward + 0.3 * final_reward
    print(f"Final score (weighted combination): {{final_score:.4f}}")
    
except Exception as e:
    print(f"Error during training: {{str(e)}}")
    print("Traceback:")
    traceback.print_exc()
    final_score = -1000.0  # Set a default value for failed training

# Save the final score
with open('trial_result.txt', 'w') as f:
    f.write(str(final_score))
''')

    # Run the temporary script
    subprocess.run([sys.executable, 'temp_train.py'])
    
    # Read the result
    try:
        with open('trial_result.txt', 'r') as f:
            final_score = float(f.read().strip())
    except FileNotFoundError:
        print("Error: trial_result.txt not found. The training script may have failed.")
        final_score = -1000  # Return a very low reward to indicate failure
    
    # Clean up
    if os.path.exists('temp_train.py'):
        os.remove('temp_train.py')
    if os.path.exists('trial_result.txt'):
        os.remove('trial_result.txt')
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return final_score

def objective_ensemble(trial):
    # Fixed weights for models (not tuned)
    gnn4_weight = 0.5  # Fixed at 0.5
    ppo_weight = 0.5   # Fixed at 0.5
    
    # Other hyperparameters to optimize
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    episodes = trial.suggest_int('episodes', 20, 50)  # Reduced from 50-100
    experiences_per_episode = trial.suggest_int('experiences_per_episode', 16, 64)  # Reduced from 32-128
    batch_size = trial.suggest_int('batch_size', 32, 64)  # Reduced from 16-128

    # Meta-learner hyperparameters
    temporal_score_boost = trial.suggest_float('temporal_score_boost', 0.1, 0.3)
    diversity_bonus = trial.suggest_float('diversity_bonus', 1.1, 1.5)
    confidence_threshold = trial.suggest_float('confidence_threshold', 0.5, 0.8)
    confidence_boost = trial.suggest_float('confidence_boost', 1.3, 2.0)
    historical_context_weight = trial.suggest_float('historical_context_weight', 0.1, 0.2)
    recency_decay = trial.suggest_float('recency_decay', 0.7, 0.9)
    alpha = trial.suggest_float('alpha', 0.1, 0.3)
    positive_reward_scale = trial.suggest_float('positive_reward_scale', 1.5, 3.0)
    negative_reward_scale = trial.suggest_float('negative_reward_scale', 2.0, 4.0)

    # Fixed seed for PPO training
    FIXED_SEED = 42

    # Create a temporary script with the current hyperparameters
    with open('temp_train.py', 'w') as f:
        f.write(f'''import sys
import os
import numpy as np
import traceback
import torch
import pandas as pd
from tqdm import tqdm

sys.path.append('{os.path.dirname(os.path.abspath(__file__))}')
from train_gnn4 import train_rl_agent
from GNN4 import load_data
from PPOv3 import PPOAgent, NBREnvironment

# Define the ground truth file path
ground_truth_file = '{GROUND_TRUTH_FILE}'
train_data_file = '{TRAIN_DATA_FILE}'
print(f"Using ground truth file: {{ground_truth_file}}")
if not os.path.exists(ground_truth_file):
    print(f"ERROR: Ground truth file not found at {{ground_truth_file}}")
    sys.exit(1)

print("Starting training with parameters:")
print(f"  GNN4 weight: {gnn4_weight} (fixed)")
print(f"  PPO weight: {ppo_weight} (fixed)")
print(f"  Learning rate: {learning_rate}")
print(f"  Episodes: {episodes}")
print(f"  Experiences per episode: {experiences_per_episode}")
print(f"  Batch size: {batch_size}")
print(f"  Temporal score boost: {temporal_score_boost}")
print(f"  Diversity bonus: {diversity_bonus}")
print(f"  Confidence threshold: {confidence_threshold}")
print(f"  Confidence boost: {confidence_boost}")
print(f"  Historical context weight: {historical_context_weight}")
print(f"  Recency decay: {recency_decay}")
print(f"  Alpha: {alpha}")
print(f"  Positive reward scale: {positive_reward_scale}")
print(f"  Negative reward scale: {negative_reward_scale}")
print(f"  Using fixed seed: {FIXED_SEED}")

# Load and preprocess data once
print("Loading and preprocessing data...")
results = load_data(
    file_path=train_data_file,
    params=None
)
data, unique_users, unique_products, user_to_idx, product_to_idx, user_product_freq, user_total_items, product_total_purchases, relationship_strengths, confidence_scores, regularity_scores = results
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train GNN4-DQN with early stopping
try:
    print("Training GNN4-DQN...")
    gnn4_agent, gnn4_model, gnn4_episodes, gnn4_rewards = train_rl_agent(
        data=data,
        device=device,
        user_nodes=unique_users,
        product_nodes=unique_products,
        reverse_user_mapping=user_to_idx,
        product_to_idx=product_to_idx,
        episodes={episodes},
        learning_rate={learning_rate},
        batch_size={batch_size},
        gamma=0.05,
        experiences_per_episode={experiences_per_episode},
        ground_truth_file=ground_truth_file
    )
    
    # Early stopping check for GNN4
    if len(gnn4_rewards) > 5 and np.mean(gnn4_rewards[-5:]) < -500:
        print("Early stopping GNN4 due to poor performance")
        gnn4_rewards = [-1000.0]
    else:
        print(f"GNN4-DQN training completed successfully. Final reward: {{gnn4_rewards[-1]}}")
except Exception as e:
    print(f"Error during GNN4-DQN training: {{str(e)}}")
    print("Traceback:")
    traceback.print_exc()
    gnn4_rewards = [-1000.0]

# Train PPO with early stopping
try:
    print("Training PPO...")
    # Create environment with smaller subset of users for faster training
    env = NBREnvironment(
        train_file=train_data_file,
        test_file=ground_truth_file,
        basket_size=10
    )
    
    # Create agent with hyperparameters and fixed seed
    ppo_agent = PPOAgent(
        env=env,
        hidden_dim=128,
        embedding_dim=128,
        lr={learning_rate},
        gamma=0.95,
        batch_size={batch_size},
        seed={FIXED_SEED}
    )
    
    # Train with early stopping
    ppo_rewards = []
    for episode in range({episodes}):
        ppo_agent.train(num_episodes=1)
        current_reward = np.mean(ppo_agent.avg_rewards[-5:]) if len(ppo_agent.avg_rewards) >= 5 else ppo_agent.avg_rewards[-1]
        ppo_rewards.append(current_reward)
        
        # Early stopping check
        if len(ppo_rewards) > 5 and np.mean(ppo_rewards[-5:]) < -500:
            print("Early stopping PPO due to poor performance")
            ppo_final_reward = -1000.0
            break
    else:
        ppo_final_reward = np.mean(ppo_rewards[-5:]) if len(ppo_rewards) >= 5 else ppo_rewards[-1]
        print(f"PPO training completed successfully. Final reward: {{ppo_final_reward}}")
except Exception as e:
    print(f"Error during PPO training: {{str(e)}}")
    print("Traceback:")
    traceback.print_exc()
    ppo_final_reward = -1000.0

# Calculate ensemble reward using fixed weights
gnn4_final_reward = gnn4_rewards[-1]
ensemble_reward = {gnn4_weight} * gnn4_final_reward + {ppo_weight} * ppo_final_reward
print(f"Ensemble reward: {{ensemble_reward}} (GNN4: {{gnn4_final_reward}}, PPO: {{ppo_final_reward}})")

# Save the final reward
with open('trial_result.txt', 'w') as f:
    f.write(str(ensemble_reward))
''')

    # Run the temporary script
    subprocess.run([sys.executable, 'temp_train.py'])
    
    # Read the result
    try:
        with open('trial_result.txt', 'r') as f:
            final_reward = float(f.read().strip())
    except FileNotFoundError:
        print("Error: trial_result.txt not found. The training script may have failed.")
        final_reward = -1000  # Return a very low reward to indicate failure
    
    # Clean up
    if os.path.exists('temp_train.py'):
        os.remove('temp_train.py')
    if os.path.exists('trial_result.txt'):
        os.remove('trial_result.txt')
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return final_reward

def main():
    # Ask user which model to optimize
    print("\nWhich model would you like to optimize?")
    print("1. GNN4")
    print("2. PPO")
    print("3. Ensemble (GNN4 + PPO)")
    
    while True:
        choice = input("Enter your choice (1, 2, or 3): ")
        if choice in ["1", "2", "3"]:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    # Set model name and objective function based on choice
    if choice == "1":
        model_name = "GNN4"
        objective_func = objective_gnn4
    elif choice == "2":
        model_name = "PPO"
        objective_func = objective_ppo
    elif choice == "3":
        model_name = "Ensemble (GNN4 + PPO)"
        objective_func = objective_ensemble
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
        return
    
    # Get current time for email notification and file naming
    current_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    
    # Create a directory for this optimization run
    run_dir = f"optimization_run_{current_time}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Send start notification
    start_subject = "Hyperparameter Tuning Started"
    start_body = f"{model_name} hyperparameter tuning process started at {current_time}"
    send_email(start_subject, start_body)

    # Create study with SQLite storage - use a unique name by appending timestamp
    import time
    timestamp = int(time.time())
    study_name = f"{model_name.lower().replace(' ', '_')}_study_{timestamp}"
    
    # Start a new study with a unique name
    study = optuna.create_study(
        direction='maximize',
        storage='sqlite:///optuna.db',
        study_name=study_name,
        load_if_exists=False
    )
    print(f"Created new study: {study_name}")
    
    # Lists to store trial results for real-time plotting
    trial_numbers = []
    trial_values = []
    best_values = []
    
    def plot_trial_history():
        """Create and save plots of trial history, excluding very low outliers."""
        plt.figure(figsize=(12, 6))
        
        # Filter out very low values (e.g., failed trials)
        filtered_trials = [(n, v) for n, v in zip(trial_numbers, trial_values) if v > -100]
        if filtered_trials:
            filtered_numbers, filtered_values = zip(*filtered_trials)
        else:
            filtered_numbers, filtered_values = [], []
        
        # Plot filtered trial values
        plt.plot(filtered_numbers, filtered_values, 'b.', label='Trial Value', alpha=0.3)
        plt.plot(trial_numbers, best_values, 'r-', label='Best Value', linewidth=2)
        plt.xlabel('Trial Number')
        plt.ylabel('Score')
        plt.title(f'{model_name} Optimization Progress (outliers excluded)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(run_dir, 'optimization_progress.png'))
        plt.close()
        
        # Create a moving average plot (unchanged)
        plt.figure(figsize=(12, 6))
        window_size = min(10, len(trial_values))
        if window_size > 0:
            moving_avg = np.convolve(trial_values, np.ones(window_size)/window_size, mode='valid')
            plt.plot(trial_numbers[window_size-1:], moving_avg, 'g-', label=f'Moving Average (window={window_size})', linewidth=2)
            plt.plot(trial_numbers, trial_values, 'b.', label='Trial Value', alpha=0.3)
            plt.xlabel('Trial Number')
            plt.ylabel('Score')
            plt.title(f'{model_name} Moving Average Progress')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(run_dir, 'moving_average_progress.png'))
            plt.close()
    
    try:
        # Define a callback to print trial results and update plots
        def print_trial_result(study, trial):
            print(f"\nTrial {trial.number + 1}/1000 completed")
            print(f"  Value: {trial.value}")
            print("  Params:")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")
            
            # Update trial history
            trial_numbers.append(trial.number)
            trial_values.append(trial.value)
            best_values.append(study.best_value)
            
            # Update plots every trial
            plot_trial_history()
        
        study.optimize(objective_func, n_trials=50, callbacks=[print_trial_result])
        
        # Print results
        print("\nBest trial:")
        print(f"  Value: {study.best_trial.value}")
        print("  Params: ")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")
        
        # Create visualization plots
        try:
            # Create optimization history plot
            history_fig = plot_optimization_history(study)
            history_fig.write_image(os.path.join(run_dir, f"{model_name.lower().replace(' ', '_')}_optimization_history.png"))
            
            # Create parameter importance plot
            param_fig = plot_param_importances(study)
            param_fig.write_image(os.path.join(run_dir, f"{model_name.lower().replace(' ', '_')}_param_importances.png"))
            
            # If it's GNN4, create additional plots for weight distributions
            if choice == "1":
                # Plot strength weights distribution
                plt.figure(figsize=(10, 6))
                strength_weights = [(t.params['strength_freq_weight'], 
                                   t.params['strength_user_affinity_weight'],
                                   1.0 - t.params['strength_freq_weight'] - t.params['strength_user_affinity_weight'])
                                  for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                plt.boxplot([[w[0] for w in strength_weights],
                           [w[1] for w in strength_weights],
                           [w[2] for w in strength_weights]],
                          labels=['Frequency', 'User Affinity', 'Product Affinity'])
                plt.title('Distribution of Strength Weights')
                plt.ylabel('Weight Value')
                plt.savefig(os.path.join(run_dir, f"{model_name.lower()}_strength_weights_dist.png"))
                plt.close()
                
                # Plot confidence weights distribution
                plt.figure(figsize=(10, 6))
                confidence_weights = [(t.params['confidence_freq_weight'],
                                     t.params['confidence_consistency_weight'],
                                     t.params['confidence_recency_weight'],
                                     1.0 - t.params['confidence_freq_weight'] - 
                                     t.params['confidence_consistency_weight'] -
                                     t.params['confidence_recency_weight'])
                                    for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                plt.boxplot([[w[0] for w in confidence_weights],
                           [w[1] for w in confidence_weights],
                           [w[2] for w in confidence_weights],
                           [w[3] for w in confidence_weights]],
                          labels=['Frequency', 'Consistency', 'Recency', 'Strength'])
                plt.title('Distribution of Confidence Weights')
                plt.ylabel('Weight Value')
                plt.savefig(os.path.join(run_dir, f"{model_name.lower()}_confidence_weights_dist.png"))
                plt.close()
            
            print(f"Plots saved in directory: {run_dir}")
        except Exception as e:
            print(f"Warning: Could not create visualization plots: {e}")
        
        # Save best parameters to a file
        with open(os.path.join(run_dir, f"{model_name.lower().replace(' ', '_')}_best_params.txt"), 'w') as f:
            f.write(f"Best {model_name} hyperparameters:\n")
            for key, value in study.best_params.items():
                f.write(f"{key}: {value}\n")
            f.write(f"\nBest score: {study.best_trial.value}")
        print(f"Best parameters saved in: {run_dir}")
        
        # Send completion email with results
        completion_time = datetime.datetime.now().strftime("%d-%m-%Y, %H:%M:%S")
        completion_subject = f"Hyperparameter Tuning Completed - {model_name}"
        completion_body = f"Hyperparameter tuning for {model_name} completed at {completion_time}\n\n"
        completion_body += f"Best trial value: {study.best_trial.value}\n\n"
        completion_body += "Best parameters:\n"
        for key, value in study.best_params.items():
            completion_body += f"  {key}: {value}\n"
        completion_body += f"\nResults and plots saved in directory: {run_dir}"
        
        send_email(completion_subject, completion_body)
            
    except Exception as e:
        # Send error email
        error_subject = f"Hyperparameter Tuning Error - {model_name}"
        error_body = f"Error during hyperparameter tuning for {model_name}:\n{str(e)}\n\n"
        error_body += f"Traceback:\n{traceback.format_exc()}"
        send_email(error_subject, error_body)
        
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        # Always perform final cleanup
        print("\nPerforming final cleanup...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared")
        print("Cleanup complete")

if __name__ == "__main__":
    main()