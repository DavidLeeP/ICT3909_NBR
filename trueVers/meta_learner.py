import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
import random
from trueVers.GNN import BipartiteGNN, load_model_state
from trueVers.PPO import PPOAgent, NBREnvironment, set_seed, evaluate_model
import glob
from datetime import datetime
from mail import send_email
import torch.nn.functional as F
from prediction import prediction
import gc
import json
import math
from collections import defaultdict

def set_seed(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class NBREnvironment(gym.Env):
    """Next Basket Recommendation Environment"""
    def __init__(self, train_file="trueVers/instacart_train_baskets.csv", test_file="trueVers/instacart_test_baskets.csv", basket_size=10):
        super(NBREnvironment, self).__init__()
        
        # Configuration
        self.train_file = train_file
        self.test_file = test_file
        self.basket_size = basket_size

        # Load datasets
        self.train_df = pd.read_csv(self.train_file)
        self.test_df = pd.read_csv(self.test_file)
        
        # Get unique users sorted for consistency
        self.users = sorted(self.train_df['user_id'].unique())
        self.user_iterator = iter(self.users)
        self.current_user = None
        
        # Max product ID (for action space)
        self.max_product_id = 49688
        
        # Define observation space
        self.observation_space = spaces.Dict({
            "user_id": spaces.Discrete(len(self.users)),  # User ID
            "order_history": spaces.Box(
                low=0,  # Changed from 1 to 0 to allow padding
                high=self.max_product_id,
                shape=(100, 100),  # Fixed shape: (max_orders, max_items_per_order)
                dtype=np.int32
            ),
            "days_between_orders": spaces.Box(
                low=0,
                high=31,
                shape=(100,),  # Fixed length for days between orders
                dtype=np.float32
            ),
            "days_since_last": spaces.Box(
                low=0,
                high=365,
                shape=(1,),
                dtype=np.float32
            )
        })
        
        # Define action space (selecting k items from product catalog)
        self.action_space = spaces.MultiDiscrete([self.max_product_id] * self.basket_size)
    
    def reset(self, seed=None):
        """Reset the environment to a new user"""
        try:
            self.current_user = next(self.user_iterator)
        except StopIteration:
            # Restart if we've gone through all users
            self.user_iterator = iter(self.users)
            self.current_user = next(self.user_iterator)
        
        observation = self._get_user_observation(self.current_user)
        return observation, {}
    
    def step(self, action):
        """Take a step (predict next basket for current user)"""
        # Calculate reward based on how well action matches ground truth
        reward = self._calculate_reward(action)
        
        # Move to next user
        try:
            self.current_user = next(self.user_iterator)
            observation = self._get_user_observation(self.current_user)
            done = False
        except StopIteration:
            # End of dataset
            observation, _ = self.reset()
            done = True
        
        return observation, reward, done, False, {}
    
    def _get_user_observation(self, user_id):
        """Create observation for a user"""
        # Get user's order history from training data
        user_orders = self.train_df[self.train_df['user_id'] == user_id].sort_values('order_id')
        
        # Extract order history (list of lists of products)
        order_history = []
        for _, row in user_orders.iterrows():
            # Extract non-zero product IDs from the row
            products = [int(p) for p in row.iloc[2:] if p > 0]
            if products:  # Only add non-empty orders
                order_history.append(products)
        
        # Extract days between orders
        days_between = user_orders['days_since_prior_order'].fillna(0).tolist()
        
        # Get days since last order from test data
        test_order = self.test_df[self.test_df['user_id'] == user_id]
        days_since_last = test_order['days_since_prior_order'].values[0] if not test_order.empty else 0.0

        return {
            "user_id": user_id,
            "order_history": order_history,
            "days_between_orders": np.array(days_between, dtype=np.float32),
            "days_since_last": np.array([days_since_last], dtype=np.float32)
        }

    def _calculate_reward(self, action):
        """Calculate reward based on weighted model predictions with enhanced metrics"""
        # Ensure predictions are of the same length
        if len(action) != len(self.action_space):
            min_len = min(len(action), len(self.action_space))
            action = action[:min_len]
        
        # Create prediction instance for combining predictions
        combined_preds = []
        combined_confs = []
        combined_weights = []
        combined_accuracies = []
        
        # Add PPO predictions with weighted confidence
        for pred in action:
            if pred > 0:  # Only add non-zero predictions
                combined_preds.append(pred)
                weighted_conf = self.model_confidences['ppo'] * self.model_weights['ppo'] * self.model_accuracies['ppo']
                combined_confs.append(weighted_conf)
                combined_weights.append(self.model_weights['ppo'])
                combined_accuracies.append(self.model_accuracies['ppo'])
        
        if not combined_preds:  # If no valid predictions
            return 0.0
        
        # Get final predictions using the prediction class
        pred = prediction(
            predList=combined_preds,
            conList=combined_confs,
            weight=combined_weights,
            accuracy=combined_accuracies
        )
        
        # Get top-k predictions instead of just the final item
        final_preds = pred.get_top_k_predictions(self.basket_size)
        if not final_preds:
            return 0.0
        
        # Convert predictions and ground truth to sets
        predicted_set = set(final_preds)
        ground_truth_set = set(self.action_space)
        
        # Calculate base metrics
        true_positives = len(predicted_set.intersection(ground_truth_set))
        false_positives = len(predicted_set - ground_truth_set)
        false_negatives = len(ground_truth_set - predicted_set)
        
        # Calculate precision and recall
        precision = true_positives / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = true_positives / len(ground_truth_set) if len(ground_truth_set) > 0 else 0
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate diversity score
        unique_items = len(set(final_preds))
        diversity_score = unique_items / len(final_preds)
        
        # Calculate position-based rewards
        position_rewards = 0
        for i, pred in enumerate(final_preds):
            if pred in ground_truth_set:
                position_rewards += 1.0 / (i + 1)  # Higher reward for correct early predictions
        
        # Calculate confidence-weighted reward
        confidence_reward = 0
        for pred in final_preds:
            if pred in ground_truth_set:
                pred_idx = combined_preds.index(pred)
                confidence_reward += combined_confs[pred_idx]
        
        # Calculate penalty for false positives
        false_positive_penalty = 0.2 * false_positives
        
        # Combine all components with weights
        weights = {
            'f1': 0.4,           # Base accuracy
            'diversity': 0.2,    # Diversity of predictions
            'position': 0.2,     # Position-based rewards
            'confidence': 0.2    # Confidence-weighted rewards
        }
        
        # Calculate total reward
        total_reward = (
            f1 * weights['f1'] +
            diversity_score * weights['diversity'] +
            position_rewards * weights['position'] +
            confidence_reward * weights['confidence'] -
            false_positive_penalty
        )
        
        # Apply non-linear scaling to make rewards more meaningful
        if total_reward > 0:
            total_reward = math.sqrt(total_reward) * 2
        else:
            total_reward = total_reward * 3
        
        # Add bonus for multiple correct predictions
        if true_positives > 1:
            total_reward += 0.5 * true_positives
        
        # Cap the reward between -2 and 10
        total_reward = min(max(total_reward, -2.0), 10.0)
        
        return total_reward

class NextBasketEncoder(nn.Module):
    """Enhanced encoder for user order history and temporal information using attention mechanisms"""
    def __init__(self, embedding_dim=64, hidden_dim=128, max_items=49688):
        super(NextBasketEncoder, self).__init__()
        
        # Item embedding layer with larger dimension
        self.item_embedding = nn.Embedding(max_items+1, embedding_dim, padding_idx=0)
        
        # Self-attention for item interactions within a basket
        self.item_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4, batch_first=True)
        
        # Bidirectional GRU to capture order sequence information
        self.order_rnn = nn.GRU(
            embedding_dim, 
            hidden_dim, 
            batch_first=True, 
            bidirectional=True  # Use bidirectional to capture patterns in both directions
        )
        
        # Additional transformer layer for capturing complex patterns
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim*2,  # *2 because of bidirectional GRU
            nhead=4,
            dim_feedforward=hidden_dim*4,
            dropout=0.1,
            batch_first=True
        )
        
        # Enhanced temporal feature processing
        self.temporal_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim//4),
            nn.LayerNorm(hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.ReLU()
        )
        
        # Feature fusion with attention
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim*2 + hidden_dim//2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, order_history, days_between, days_since_last):
        batch_size = len(order_history)
        
        # Process each user's order history separately
        encoded_users = []
        
        for i in range(batch_size):
            user_orders = order_history[i]
            num_orders = len(user_orders)
            
            if num_orders == 0:
                # If no order history, use zeros
                order_encoding = torch.zeros(1, self.order_rnn.hidden_size * 2)  # *2 for bidirectional
            else:
                # Enhanced representation for each order with self-attention
                order_embeddings = []
                for order in user_orders:
                    # Ensure order is a list of items, not a single item
                    if not isinstance(order, (list, tuple, np.ndarray)):
                        # If somehow a single integer gets here, wrap it in a list
                        order = [order]
                        
                    # Convert order to tensor and get embeddings
                    order_tensor = torch.tensor(order, dtype=torch.long)
                    item_embeds = self.item_embedding(order_tensor)
                    
                    # Apply self-attention if there are multiple items
                    if len(order) > 1:
                        # Self-attention for item interactions
                        attn_output, _ = self.item_attention(
                            item_embeds.unsqueeze(0),
                            item_embeds.unsqueeze(0),
                            item_embeds.unsqueeze(0)
                        )
                        item_embeds = attn_output.squeeze(0)
                    
                    # Enhanced order representation (average of item embeddings)
                    order_embed = item_embeds.mean(dim=0, keepdim=True)
                    order_embeddings.append(order_embed)
                
                # Stack all orders
                order_embeddings = torch.cat(order_embeddings, dim=0).unsqueeze(0)  # [1, num_orders, embed_dim]
                
                # Apply bidirectional GRU to capture sequential patterns
                gru_output, _ = self.order_rnn(order_embeddings)
                
                # Apply transformer layer to capture complex patterns
                if gru_output.size(1) > 1:  # Only if we have multiple orders
                    transformer_output = self.transformer_layer(gru_output)
                    # Global average pooling
                    order_encoding = transformer_output.mean(dim=1)  # [1, hidden_size*2]
                else:
                    order_encoding = gru_output.mean(dim=1)  # [1, hidden_size*2]
            
            # Enhanced temporal features
            # Ensure proper indexing by checking if i is within range for days_between
            if i < len(days_between):
                current_days_between = days_between[i]
                days_mean = torch.tensor(current_days_between, dtype=torch.float).mean().reshape(1, 1)
                days_std = torch.tensor(current_days_between, dtype=torch.float).std().reshape(1, 1) if len(current_days_between) > 1 else torch.zeros(1, 1)
            else:
                # Use default values if index is out of range
                days_mean = torch.zeros(1, 1)
                days_std = torch.zeros(1, 1)
            
            # Ensure proper indexing for days_since_last
            if i < len(days_since_last):
                recency = torch.tensor(days_since_last[i], dtype=torch.float).reshape(1, 1)
            else:
                recency = torch.zeros(1, 1)
            
            # Create expanded temporal features
            temporal_features = torch.cat([
                days_mean,  # Average days between orders
                recency,    # Days since last order
            ], dim=1)
            
            temporal_encoding = self.temporal_encoder(temporal_features)
            
            # Combine features with attention-based fusion
            user_encoding = self.combiner(torch.cat([order_encoding, temporal_encoding], dim=1))
            encoded_users.append(user_encoding)
        
        # Stack all users
        return torch.cat(encoded_users, dim=0)  # [batch_size, hidden_size]

class NextBasketPredictor(nn.Module):
    """Enhanced next basket prediction model with advanced architecture"""
    def __init__(self, embedding_dim=128, hidden_dim=256, max_items=49688, basket_size=10):
        super(NextBasketPredictor, self).__init__()
        
        # Improved encoder with larger dimensions and attention mechanisms
        self.encoder = NextBasketEncoder(embedding_dim, hidden_dim, max_items)
        
        # Enhanced actor network with residual connections
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(0.2),
            ResidualBlock(hidden_dim*2),  # Add residual block for better gradient flow
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim*2, max_items+1),
            nn.Softmax(dim=-1)
        )
        
        # Initialize weights for better convergence
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.414)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Improved critic network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim),  # Add residual block
            nn.Linear(hidden_dim, 1)
        )
        
        self.max_items = max_items
        self.basket_size = basket_size
    
    def forward(self, order_history, days_between, days_since_last):
        # Encode user data
        encoding = self.encoder(order_history, days_between, days_since_last)
        
        # Get item probabilities and value
        item_probs = self.actor(encoding)
        value = self.critic(encoding)
        
        return item_probs, value
    
    def act(self, order_history, days_between, days_since_last):
        """Generate next basket prediction without training"""
        with torch.no_grad():
            item_probs, _ = self(order_history, days_between, days_since_last)
            
            # Get top-k items (basket_size)
            _, top_items = torch.topk(item_probs, self.basket_size)
            
            return top_items.cpu().numpy()

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
    
    def forward(self, x):
        return x + self.block(x)  # Residual connection

def find_latest_gnn4_model():
    """
    Find the most recent GNN4 model directory.
    
    Returns:
        str: Path to the most recent GNN4 model directory
    """
    # Get all model directories in model_checkpoints
    model_dirs = glob.glob("model_checkpoints/*")
    
    if not model_dirs:
        raise FileNotFoundError("No models found in model_checkpoints directory")
    
    # Sort by creation time (newest first)
    latest_dir = max(model_dirs, key=os.path.getctime)
    print(f"Found latest model: {latest_dir}")
    
    # Check if we found a directory or a file
    if os.path.isfile(latest_dir):
        # If we found a file, get its parent directory
        latest_dir = os.path.dirname(latest_dir)
    
    return latest_dir

def find_latest_ppo_model():
    """
    Find the most recent PPO model file.
    
    Returns:
        str: Name of the most recent PPO model file
    """
    # Get all PPO model files
    ppo_files = glob.glob("models/*.pt")
    
    if not ppo_files:
        raise FileNotFoundError("No PPO models found in models directory")
    
    # Sort by creation time (newest first)
    latest_file = max(ppo_files, key=os.path.getctime)
    model_name = os.path.splitext(os.path.basename(latest_file))[0]
    print(f"Found latest PPO model: {model_name}")
    
    return model_name

def plot_model_weights(ppo_agent, gnn4, show_plot=False):
    """
    Plot the weights of PPO and GNN4 models over time.
    
    Args:
        ppo_agent: The PPO agent instance
        gnn4: The GNN4 model instance
        show_plot: Whether to display the plot interactively
    """
    # Get PPO weights
    ppo_weights = []
    for name, param in ppo_agent.model.named_parameters():
        if 'weight' in name:
            ppo_weights.append(param.data.cpu().numpy().flatten())
    ppo_weights = np.concatenate(ppo_weights)
    
    # Get GNN4 weights
    gnn4_weights = []
    for name, param in gnn4.named_parameters():
        if 'weight' in name:
            gnn4_weights.append(param.data.cpu().numpy().flatten())
    gnn4_weights = np.concatenate(gnn4_weights)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot weights over time
    plt.plot(ppo_weights, 'b-', label='PPO Weights', alpha=0.7)
    plt.plot(gnn4_weights, 'r-', label='GNN4 Weights', alpha=0.7)
    plt.title('Model Weights Over Time')
    plt.xlabel('Weight Index')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('model_weights_over_time.png', dpi=300)
    print("Model weights over time plot saved to 'model_weights_over_time.png'")
    plt.close()  # Always close the figure to prevent display

def save_model_weights(ppo_agent, gnn4, episode, save_dir="weight_history"):
    """
    Save the current weights of both models for later analysis.
    
    Args:
        ppo_agent: The PPO agent instance
        gnn4: The GNN4 model instance
        episode: Current episode number
        save_dir: Directory to save weight history
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get PPO weights
    ppo_weights = []
    for name, param in ppo_agent.model.named_parameters():
        if 'weight' in name:
            ppo_weights.append(param.data.cpu().numpy().flatten())
    ppo_weights = np.concatenate(ppo_weights)
    
    # Get GNN4 weights
    gnn4_weights = []
    for name, param in gnn4.named_parameters():
        if 'weight' in name:
            gnn4_weights.append(param.data.cpu().numpy().flatten())
    gnn4_weights = np.concatenate(gnn4_weights)
    
    # Save weights
    np.save(os.path.join(save_dir, f"ppo_weights_ep{episode}.npy"), ppo_weights)
    np.save(os.path.join(save_dir, f"gnn4_weights_ep{episode}.npy"), gnn4_weights)

def plot_weight_history(save_dir="weight_history"):
    """
    Plot the evolution of model weights over training episodes.
        
    Args:
        save_dir: Directory containing saved weight history
    """
    # Get all weight files
    ppo_files = sorted(glob.glob(os.path.join(save_dir, "ppo_weights_ep*.npy")))
    gnn4_files = sorted(glob.glob(os.path.join(save_dir, "gnn4_weights_ep*.npy")))
    
    if not ppo_files or not gnn4_files:
        print("No weight history files found")
        return
    
    # Extract episode numbers
    episodes = [int(f.split('ep')[1].split('.')[0]) for f in ppo_files]
    
    # Load and process weights
    ppo_weights_history = []
    gnn4_weights_history = []
    
    for ppo_file, gnn4_file in zip(ppo_files, gnn4_files):
        ppo_weights = np.load(ppo_file)
        gnn4_weights = np.load(gnn4_file)
        
        # Calculate mean and std of weights
        ppo_weights_history.append({
            'mean': np.mean(ppo_weights),
            'std': np.std(ppo_weights)
        })
        gnn4_weights_history.append({
            'mean': np.mean(gnn4_weights),
            'std': np.std(gnn4_weights)
        })
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Ensure all arrays have the same length
    min_length = min(len(episodes), len(ppo_weights_history), len(gnn4_weights_history))
    episodes = episodes[:min_length]
    ppo_weights_history = ppo_weights_history[:min_length]
    gnn4_weights_history = gnn4_weights_history[:min_length]
            
    # Plot PPO weights
    plt.subplot(2, 1, 1)
    ppo_means = [w['mean'] for w in ppo_weights_history]
    ppo_stds = [w['std'] for w in ppo_weights_history]
    plt.plot(episodes, ppo_means, 'b-', label='PPO Mean Weights')
    plt.fill_between(episodes, 
                    np.array(ppo_means) - np.array(ppo_stds),
                    np.array(ppo_means) + np.array(ppo_stds),
                    alpha=0.2, color='b')
    plt.title('PPO Model Weight Evolution')
    plt.xlabel('Episode')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.grid(True)
            
    # Plot GNN4 weights
    plt.subplot(2, 1, 2)
    gnn4_means = [w['mean'] for w in gnn4_weights_history]
    gnn4_stds = [w['std'] for w in gnn4_weights_history]
    plt.plot(episodes, gnn4_means, 'r-', label='GNN4 Mean Weights')
    plt.fill_between(episodes,
                    np.array(gnn4_means) - np.array(gnn4_stds),
                    np.array(gnn4_means) + np.array(gnn4_stds),
                    alpha=0.2, color='r')
    plt.title('GNN4 Model Weight Evolution')
    plt.xlabel('Episode')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.grid(True)
        
    # Save the figure
    plt.tight_layout()
    plt.savefig('model_weights_evolution.png', dpi=300)
    print("Model weights evolution plot saved to 'model_weights_evolution.png'")
    plt.close()

def load_models(gnn4_model_dir=None, ppo_model_name=None):
    """
    Load both GNN4 and PPO models.
    
    Args:
        gnn4_model_dir: Path to the model directory (e.g. "model_checkpoints/20240315_123456")
                      If None, will find the latest model
        ppo_model_name: Name of the PPO model file (e.g. "model")
                       If None, will find the latest model
    
    Returns:
        tuple: (gnn4, graph_data, ppo_agent)
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find latest models if not specified
    if gnn4_model_dir is None:
        gnn4_model_dir = find_latest_gnn4_model()
    if ppo_model_name is None:
        ppo_model_name = find_latest_ppo_model()
    
    # Load GNN4 model and data
    print(f"Loading model from {gnn4_model_dir}")
    
    # Load the mappings and scores file first
    with open(os.path.join(gnn4_model_dir, 'mappings_and_scores.json'), 'r') as f:
        mappings_and_scores = json.load(f)
    
    # Convert string keys back to tuples for relevant dictionaries
    def convert_keys_back(d: dict, is_tuple: bool = False) -> dict:
        if is_tuple:
            return {tuple(map(lambda x: int(float(x)), k.split('_'))): v for k, v in d.items()}
        return d
    
    # Convert user and product IDs to integers
    user_to_idx = {int(float(k)): v for k, v in mappings_and_scores['user_to_idx'].items()}
    product_to_idx = {int(float(k)): v for k, v in mappings_and_scores['product_to_idx'].items()}
    
    # Convert relationship data
    relationship_strengths = convert_keys_back(mappings_and_scores['relationship_strengths'], True)
    confidence_scores = convert_keys_back(mappings_and_scores['confidence_scores'], True)
    regularity_scores = convert_keys_back(mappings_and_scores['regularity_scores'], True)
    user_product_freq = convert_keys_back(mappings_and_scores['user_product_freq'], True)
    
    # Convert other mappings
    user_total_items = {int(float(k)): v for k, v in mappings_and_scores['user_total_items'].items()}
    product_total_purchases = {int(float(k)): v for k, v in mappings_and_scores['product_total_purchases'].items()}
    
    # Load the model state
    checkpoint = torch.load(os.path.join(gnn4_model_dir, 'model.pt'), map_location=device)
    model_config = checkpoint['model_config']
    
    # Create and load model
    gnn4 = BipartiteGNN(
        num_features=model_config['num_features'],
        hidden_channels=model_config['hidden_channels'],
        num_classes=model_config['num_classes'],
        device=device
    )
    gnn4.load_state_dict(checkpoint['model_state_dict'])
    gnn4.to(device)
    
    # Create graph data structure
    graph_data = {
        'user_to_idx': user_to_idx,
        'product_to_idx': product_to_idx,
        'reverse_user_mapping': {v: k for k, v in user_to_idx.items()},
        'reverse_product_mapping': {v: k for k, v in product_to_idx.items()},
        'relationship_strengths': relationship_strengths,
        'confidence_scores': confidence_scores,
        'regularity_scores': regularity_scores,
        'user_product_freq': user_product_freq,
        'user_total_items': user_total_items,
        'product_total_purchases': product_total_purchases
    }
    
    # Create PPO environment
    env = NBREnvironment()
    
    # Create PPO agent with the same parameters as used during training
    ppo_agent = PPOAgent(
        env,
        hidden_dim=256,
        embedding_dim=128,
        batch_size=128,
        lr=5e-4,
        gamma=0.98,
        epsilon=0.2,
        epochs=5
    )
    
    # Load the saved model state
    saved_state = torch.load(os.path.join(ppo_agent.save_dir, f"{ppo_model_name}.pt"), map_location=device)
    
    # Create a new state dict with the correct dimensions
    new_state_dict = {}
    for key, value in saved_state.items():
        if 'item_embedding.weight' in key or 'actor.9.weight' in key or 'actor.9.bias' in key:
            # For these layers, we need to handle the size mismatch
            if 'item_embedding.weight' in key:
                # Create new embedding layer with correct size
                new_embedding = torch.zeros(env.max_product_id + 1, value.size(1), device=device)
                # Copy weights for existing items
                new_embedding[:value.size(0)] = value
                new_state_dict[key] = new_embedding
            elif 'actor.9.weight' in key:
                # Create new weight matrix with correct size
                new_weight = torch.zeros(env.max_product_id + 1, value.size(1), device=device)
                # Copy weights for existing items
                new_weight[:value.size(0)] = value
                new_state_dict[key] = new_weight
            elif 'actor.9.bias' in key:
                # Create new bias vector with correct size
                new_bias = torch.zeros(env.max_product_id + 1, device=device)
                # Copy biases for existing items
                new_bias[:value.size(0)] = value
                new_state_dict[key] = new_bias
        else:
            # For other layers, just copy the weights as is
            new_state_dict[key] = value
    
    # Load the modified state dict
    ppo_agent.model.load_state_dict(new_state_dict)
    
    # Set models to evaluation mode
    gnn4.eval()
    ppo_agent.eval_mode()
    
    # Plot model weights
    plot_model_weights(ppo_agent, gnn4)
    
    return gnn4, graph_data, ppo_agent

def update_weights(weights, confidences, correct, eta=0.1, lower=0.01, upper=0.05):
    for i in range(len(weights)):
        adjust = random.uniform(lower, upper)
        if correct[i]:
            weights[i] += eta * adjust * confidences[i]  # Reward
        else:
            weights[i] -= eta * adjust * confidences[i]  # Penalize
        weights[i] = max(weights[i], 0.01)
    
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]
    return weights

class MetaLearner:
    """Meta-learner that can process all users independently"""
    def __init__(self, train_file="trueVers/instacart_train_baskets.csv", test_file="trueVers/instacart_test_baskets.csv", basket_size=10):
        self.train_df = pd.read_csv(train_file)
        self.test_df = pd.read_csv(test_file)
        self.basket_size = basket_size
        self.max_product_id = 49688
        
        # Get unique users
        self.users = sorted(self.train_df['user_id'].unique())
        
        # Initialize predictor
        self.predictor = NextBasketPredictor(
            embedding_dim=128,
            hidden_dim=256,
            max_items=self.max_product_id,
            basket_size=self.basket_size
        )
        
        # Initialize encoder
        self.encoder = NextBasketEncoder(
            embedding_dim=64,
            hidden_dim=128,
            max_items=self.max_product_id
        )
        
        # Set to evaluation mode
        self.predictor.eval()
        self.encoder.eval()
        
        # Initialize PPO and GNN4 agents
        self.ppo_agent = None
        self.gnn4 = None
        self.graph_data = None
        
        # Initialize model weights with equal values of 1.0
        self.model_weights = {
            'ppo': 1.0,  # Start with weight 1.0 for PPO
            'gnn4': 1.0   # Start with weight 1.0 for GNN4
        }
        
        # Initialize model confidences (will be updated per item)
        self.model_confidences = {
            'ppo': 1.0,
            'gnn4': 1.0
        }
        
        # Initialize model accuracies
        self.model_accuracies = {
            'ppo': 0.5,  # Start with neutral accuracy
            'gnn4': 0.5
        }
        
        # Initialize optimized hyperparameters
        self.temporal_score_boost = 0.29573045626385774
        self.diversity_bonus = 1.3662239906915075
        self.confidence_threshold = 0.7385776714886988
        self.confidence_boost = 1.9056948036472252
        self.historical_context_weight = 0.1553712582582795
        self.recency_decay = 0.8500912536133545
        self.alpha = 0.2010599012877006
        self.positive_reward_scale = 1.5763661065935268
        self.negative_reward_scale = 3.8888995346820474
        self.learning_rate = 0.0010286257802440046
    
    def set_agents(self, ppo_agent, gnn4, graph_data):
        """Set the PPO and GNN4 agents for the meta-learner"""
        self.ppo_agent = ppo_agent
        self.gnn4 = gnn4
        self.graph_data = graph_data
        
        # Set agents to evaluation mode
        self.ppo_agent.eval_mode()
    
    def _get_user_observation(self, user_id):
        """Create observation for a user"""
        # Get user's order history from training data
        user_orders = self.train_df[self.train_df['user_id'] == user_id].sort_values('order_id')
        
        # Extract order history (list of lists of products)
        order_history = []
        for _, row in user_orders.iterrows():
            # Extract non-zero product IDs from the row
            products = [int(p) for p in row.iloc[2:] if p > 0]
            if products:  # Only add non-empty orders
                order_history.append(products)
        
        # Extract days between orders
        days_between = user_orders['days_since_prior_order'].fillna(0).tolist()
        
        # Get days since last order from test data
        test_order = self.test_df[self.test_df['user_id'] == user_id]
        days_since_last = test_order['days_since_prior_order'].values[0] if not test_order.empty else 0.0

        return {
            "user_id": user_id,
            "order_history": order_history,
            "days_between_orders": np.array(days_between, dtype=np.float32),
            "days_since_last": np.array([days_since_last], dtype=np.float32)
        }
    
    def _save_predictions_to_csv(self, predictions, ppo_predictions, gnn4_predictions):
        """Save predictions to a CSV file"""
        # Create directory if it doesn't exist
        os.makedirs("10b_preds", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"10b_preds/meta_learner_predictions_{timestamp}.csv"
        
        with open(filename, 'w') as f:
            # Write header
            header = ['user_id'] + \
                    [f'meta_product{i+1}' for i in range(self.basket_size)] + \
                    [f'ppo_product{i+1}' for i in range(self.basket_size)] + \
                    [f'gnn4_product{i+1}' for i in range(self.basket_size)]
            f.write(','.join(header) + '\n')
            
            # Write predictions for each user
            for user_id in sorted(predictions.keys()):
                meta_preds = predictions[user_id]
                ppo_preds = ppo_predictions[user_id]
                gnn4_preds = gnn4_predictions[user_id]
                
                # Format row
                row = [str(user_id)] + \
                      [str(p) for p in meta_preds] + \
                      [str(p) for p in ppo_preds] + \
                      [str(p) for p in gnn4_preds]
                f.write(','.join(row) + '\n')
        
        print(f"Predictions saved to {filename}")

    def process_all_users(self, batch_size=64, save_predictions=True):
        """
        Process all users in the dataset, using PPO and GNN4 predictions as hints for the meta-learner.
        Maintains learned progress from training and uses PPO-style observation format.
        """
        if self.ppo_agent is None or self.gnn4 is None:
            raise ValueError("PPO and GNN4 models must be set before processing users")
        
        # Load best weights from training if available
        try:
            with open('meta_learner_weights.json', 'r') as f:
                saved_state = json.load(f)
                self.model_weights = saved_state['weights']
                self.model_accuracies = saved_state['accuracies']
                print("Loaded best weights from training")
        except FileNotFoundError:
            print("No saved weights found, using current weights")
        
        # Pre-allocate dictionaries with expected size
        num_users = len(self.users)
        all_predictions = {}
        all_confidences = {}
        all_ppo_predictions = {}
        all_gnn4_predictions = {}
        
        # Initialize model performance tracking
        model_correct = {'ppo': 0, 'gnn4': 0}
        model_total = {'ppo': 0, 'gnn4': 0}
        
        # Pre-compute test data dictionary for faster lookups
        test_data_dict = {}
        for _, row in self.test_df.iterrows():
            user_id = row['user_id']
            purchases = [int(p) for p in row.iloc[2:] if p > 0]
            if purchases:
                test_data_dict[user_id] = purchases
        
        # Pre-compute user observations and historical data for all users
        print("Pre-computing user observations and historical data...")
        all_observations = {}
        all_historical_data = {}
        for user_id in self.users:
            # Get user's order history
            user_orders = self.train_df[self.train_df['user_id'] == user_id].sort_values('order_id')
            
            # Store historical data
            historical_data = {
                'order_history': [],
                'purchase_frequencies': defaultdict(int),
                'last_purchases': set(),
                'days_between_orders': []
            }
            
            # Process each order
            for _, row in user_orders.iterrows():
                products = [int(p) for p in row.iloc[2:] if p > 0]
                if products:
                    historical_data['order_history'].append(products)
                    for product in products:
                        historical_data['purchase_frequencies'][product] += 1
                        historical_data['last_purchases'].add(product)
            
                days = row['days_since_prior_order']
                if pd.notna(days):
                    historical_data['days_between_orders'].append(days)
            
            # Create PPO-style observation
            observation = {
                "user_id": user_id,
                "order_history": historical_data['order_history'],
                "days_between_orders": np.array(historical_data['days_between_orders'], dtype=np.float32),
                "days_since_last": np.array([user_orders['days_since_prior_order'].iloc[-1] if not user_orders.empty else 0.0], dtype=np.float32)
            }
            
            all_observations[user_id] = observation
            all_historical_data[user_id] = historical_data
        
        # Process users in batches
        for i in range(0, num_users, batch_size):
            batch_users = self.users[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            # Print progress with safe division
            if (i + batch_size) % 100 == 0:
                ppo_acc = model_correct['ppo'] / model_total['ppo'] if model_total['ppo'] > 0 else 0
                gnn4_acc = model_correct['gnn4'] / model_total['gnn4'] if model_total['gnn4'] > 0 else 0
                print(f"\nProcessed {i + batch_size}/{num_users} users")
                print(f"Model weights - PPO: {self.model_weights['ppo']:.3f}, GNN4: {self.model_weights['gnn4']:.3f}")
                print(f"Model accuracy - PPO: {ppo_acc:.3f}, GNN4: {gnn4_acc:.3f}")
            
            # Get pre-computed observations and historical data for batch
            batch_observations = [all_observations[user_id] for user_id in batch_users]
            batch_historical = [all_historical_data[user_id] for user_id in batch_users]
            
            # Process batch with all models
            with torch.no_grad():
                # Get PPO predictions
                ppo_predictions = {}
                ppo_confidences = {}
                for obs in batch_observations:
                    user_id = obs["user_id"]
                    action, confidence = self.ppo_agent.predict_with_confidence(obs)
                    ppo_predictions[user_id] = action[:self.basket_size]
                    ppo_confidences[user_id] = confidence[:self.basket_size]
                
                # Get GNN4 predictions
                gnn4_predictions = {}
                gnn4_confidences = {}
                for obs in batch_observations:
                    user_id = obs["user_id"]
                    if user_id in self.graph_data['user_to_idx']:
                        # Get top products based on relationship strengths
                        user_relationships = {
                            p: self.graph_data['relationship_strengths'].get((user_id, p), 0)
                            for p in self.graph_data['product_to_idx'].keys()
                        }
                        
                        # Get top products
                        top_products = sorted(
                            user_relationships.items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:self.basket_size]
                        
                        gnn4_predictions[user_id] = [p for p, _ in top_products]
                        gnn4_confidences[user_id] = [s for _, s in top_products]
            
            # Process each user in batch
            for j, user_id in enumerate(batch_users):
                # Get ground truth and historical data
                ground_truth = test_data_dict.get(user_id, [])
                historical_data = batch_historical[j]
                observation = batch_observations[j]
                
                if not ground_truth:
                    continue
                
                # Get days since last order from PPO-style observation
                days_since_last = observation['days_since_last'][0]
                
                # Calculate temporal relevance score based on PPO's observation format
                temporal_score = 1.0
                if len(observation['days_between_orders']) > 0:
                    avg_days = np.mean(observation['days_between_orders'])
                    std_days = np.std(observation['days_between_orders'])
                    if std_days > 0:
                        z_score = (days_since_last - avg_days) / std_days
                        temporal_score = np.exp(-0.5 * z_score**2)  # Gaussian weighting
                    else:
                        temporal_score = 1.0 if abs(days_since_last - avg_days) < 7 else 0.5
                
                # Convert ground truth to set for faster lookups
                gt_set = set(ground_truth)
                
                # Get predictions
                ppo_preds = ppo_predictions.get(user_id, [])
                gnn4_preds = gnn4_predictions.get(user_id, [])
                
                # Update model performance tracking
                ppo_hits = len(set(ppo_preds).intersection(gt_set))
                gnn4_hits = len(set(gnn4_preds).intersection(gt_set))
                model_correct['ppo'] += ppo_hits
                model_correct['gnn4'] += gnn4_hits
                model_total['ppo'] += len(ground_truth)
                model_total['gnn4'] += len(ground_truth)
                
                # Create weighted scoring for each prediction using learned weights
                weighted_scores = {}
                
                # Score PPO predictions with enhanced logic
                for pred, conf in zip(ppo_preds, ppo_confidences[user_id]):
                    base_score = self.model_weights['ppo']
                    # Add confidence-based weighting
                    base_score *= (1 + conf * self.confidence_boost)
                    
                    # Add historical context with decay
                    if pred in historical_data['purchase_frequencies']:
                        freq = historical_data['purchase_frequencies'][pred]
                        recency = 1.0
                        if pred in historical_data['last_purchases']:  # Now using set for O(1) lookup
                            recency = self.recency_decay  # Use optimized recency decay
                        base_score += freq * self.historical_context_weight * recency
                    
                    # Add diversity bonus
                    if pred not in weighted_scores:
                        base_score *= self.diversity_bonus  # Use optimized diversity bonus
                    
                    # Add confidence threshold bonus
                    if conf > self.confidence_threshold:  # Use optimized confidence threshold
                        base_score *= self.confidence_boost  # Use optimized confidence boost
                    
                    weighted_scores[pred] = max(weighted_scores.get(pred, 0), base_score)
                
                # Score GNN4 predictions with enhanced logic
                if user_id in gnn4_confidences:  # Add check for user_id in gnn4_confidences
                    for pred, conf in zip(gnn4_preds, gnn4_confidences[user_id]):
                        base_score = self.model_weights['gnn4']
                        # Add confidence-based weighting
                        base_score *= (1 + conf * self.confidence_boost)
                        
                        # Add historical context with decay
                        if pred in historical_data['purchase_frequencies']:
                            freq = historical_data['purchase_frequencies'][pred]
                            recency = 1.0
                            if pred in historical_data['last_purchases']:  # Now using set for O(1) lookup
                                recency = self.recency_decay  # Use optimized recency decay
                            base_score += freq * self.historical_context_weight * recency
                        
                        # Add diversity bonus
                        if pred not in weighted_scores:
                            base_score *= self.diversity_bonus  # Use optimized diversity bonus
                        
                        # Add confidence threshold bonus
                        if conf > self.confidence_threshold:  # Use optimized confidence threshold
                            base_score *= self.confidence_boost  # Use optimized confidence boost
                        
                        weighted_scores[pred] = max(weighted_scores.get(pred, 0), base_score)
                else:
                    # If no GNN4 confidences available, use default confidence of 0.5
                    for pred in gnn4_preds:
                        base_score = self.model_weights['gnn4'] * 0.5  # Use default confidence
                        
                        # Add historical context with decay
                        if pred in historical_data['purchase_frequencies']:
                            freq = historical_data['purchase_frequencies'][pred]
                            recency = 1.0
                            if pred in historical_data['last_purchases']:
                                recency = self.recency_decay  # Use optimized recency decay
                            base_score += freq * self.historical_context_weight * recency
                        
                        # Add diversity bonus
                        if pred not in weighted_scores:
                            base_score *= self.diversity_bonus  # Use optimized diversity bonus
                        
                        weighted_scores[pred] = max(weighted_scores.get(pred, 0), base_score)
                
                # Get top predictions based on weighted scores
                meta_preds = sorted(
                    weighted_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:self.basket_size]
                
                # Extract just the predictions
                meta_preds = [p for p, _ in meta_preds]
                
                # Store results
                all_predictions[user_id] = meta_preds
                all_confidences[user_id] = [weighted_scores[p] for p in meta_preds]
                all_ppo_predictions[user_id] = ppo_preds
                all_gnn4_predictions[user_id] = gnn4_preds
            
            # Clear memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Save predictions to CSV if requested
        if save_predictions:
            self._save_predictions_to_csv(all_predictions, all_ppo_predictions, all_gnn4_predictions)
        
        return all_predictions, all_confidences, all_ppo_predictions, all_gnn4_predictions

    def analyze_datasets(self):
        """Analyze dataset patterns to enhance learning"""
        print("Analyzing dataset patterns...")
        
        # Count item frequencies in training set
        train_items = []
        for _, row in self.train_df.iterrows():
            items = [int(p) for p in row.iloc[2:] if p > 0]
            train_items.extend(items)
        
        train_counts = pd.Series(train_items).value_counts()
        self.train_item_counts = train_counts
        
        # Count item frequencies in test set
        test_items = []
        for _, row in self.test_df.iterrows():
            items = [int(p) for p in row.iloc[2:] if p > 0]
            test_items.extend(items)
            
        test_counts = pd.Series(test_items).value_counts()
        self.test_item_counts = test_counts
        
        # Calculate overlap between train and test sets
        common_items = set(train_counts.index) & set(test_counts.index)
        train_only = set(train_counts.index) - set(test_counts.index)
        test_only = set(test_counts.index) - set(train_counts.index)
        
        print(f"Items in both train and test: {len(common_items)}")
        print(f"Items only in train: {len(train_only)}")
        print(f"Items only in test: {len(test_only)}")
        
        return {
            "common_items": common_items,
            "train_only": train_only,
            "test_only": test_only
        }

    def train_simple(self, num_episodes=100, batch_size=16, subset_size=500, num_folds=5):
        """
        Enhanced training with better historical data utilization and proper episode handling.
        """
        print(f"\nStarting meta-learner training for {num_episodes} episodes...")
        print(f"Using bootstrapped subset of {subset_size} users with batch size {batch_size}")
        
        # Start with equal weights or load previous weights if they exist
        try:
            with open('meta_learner_weights.json', 'r') as f:
                saved_state = json.load(f)
                self.model_weights = saved_state['weights']
                self.model_accuracies = saved_state['accuracies']
                print("Loaded previous weights and accuracies")
        except FileNotFoundError:
            self.model_weights = {'ppo': 0.5, 'gnn4': 0.5}
            self.model_accuracies = {'ppo': 0.5, 'gnn4': 0.5}
            print("Starting with default weights")
        
        # Initialize performance tracking
        episode_rewards = []
        episode_weights = {'ppo': [], 'gnn4': []}
        episode_accuracies = {'ppo': [], 'gnn4': []}
        
        # Pre-compute test data dictionary for faster lookups
        test_data_dict = {}
        for _, row in self.test_df.iterrows():
            user_id = int(row['user_id'])  # Convert to integer
            purchases = [int(p) for p in row.iloc[2:] if p > 0]
            if purchases:
                test_data_dict[user_id] = purchases
        
        # Get list of users with ground truth data
        valid_users = list(test_data_dict.keys())
        print(f"Found {len(valid_users)} users with ground truth data")
        
        if not valid_users:
            print("ERROR: No users with ground truth data found!")
            return self
        
        # Pre-compute user observations and historical data for all valid users
        print("Pre-computing user observations and historical data...")
        all_observations = {}
        all_historical_data = {}
        for user_id in valid_users:
            # Get user's order history
            user_orders = self.train_df[self.train_df['user_id'] == user_id].sort_values('order_id')
            
            # Store historical data
            historical_data = {
                'order_history': [],
                'purchase_frequencies': defaultdict(int),
                'last_purchases': set(),  # Changed to set for faster lookups
                'days_between_orders': []
            }
            
            # Process each order
            for _, row in user_orders.iterrows():
                products = [int(p) for p in row.iloc[2:] if p > 0]
                if products:
                    historical_data['order_history'].append(products)
                    for product in products:
                        historical_data['purchase_frequencies'][product] += 1
                        historical_data['last_purchases'].add(product)  # Add to set
                
                days = row['days_since_prior_order']
                if pd.notna(days):
                    historical_data['days_between_orders'].append(days)
            
            all_observations[user_id] = self._get_user_observation(user_id)
            all_historical_data[user_id] = historical_data
        
        # Track best model combinations
        best_ppo_weight = 0.5
        best_gnn4_weight = 0.5
        best_accuracy = 0.0
        
        # Train for specified number of episodes
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            
            # Randomly select users for this episode
            episode_users = np.random.choice(valid_users, size=subset_size, replace=True)
            print(f"Selected {len(set(episode_users))} unique users for this episode")
            
            # Initialize tracking variables for this episode
            model_correct = {'ppo': 0, 'gnn4': 0, 'meta': 0}
            model_total = {'ppo': 0, 'gnn4': 0, 'meta': 0}
            total_reward = 0
            processed_users = 0
            
            # Process users in batches
            for i in range(0, len(episode_users), batch_size):
                batch_users = episode_users[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                # Get pre-computed observations and historical data for batch
                batch_observations = [all_observations[int(user_id)] for user_id in batch_users]  # Convert to int
                batch_historical = [all_historical_data[int(user_id)] for user_id in batch_users]  # Convert to int
                
                # Process batch with all models
                with torch.no_grad():
                    # Get PPO predictions
                    ppo_predictions = {}
                    ppo_confidences = {}
                    for obs in batch_observations:
                        user_id = int(obs["user_id"])  # Convert to int
                        action, confidence = self.ppo_agent.predict_with_confidence(obs)
                        ppo_predictions[user_id] = action[:self.basket_size]
                        ppo_confidences[user_id] = confidence[:self.basket_size]
                    
                    # Get GNN4 predictions
                    gnn4_predictions = {}
                    gnn4_confidences = {}
                    for obs in batch_observations:
                        user_id = int(obs["user_id"])  # Convert to int
                        if user_id in self.graph_data['user_to_idx']:
                            # Get top products based on relationship strengths
                            user_relationships = {
                                p: self.graph_data['relationship_strengths'].get((user_id, p), 0)
                                for p in self.graph_data['product_to_idx'].keys()
                            }
                            
                            # Get top products
                            top_products = sorted(
                                user_relationships.items(),
                                key=lambda x: x[1],
                                reverse=True
                            )[:self.basket_size]
                            
                            gnn4_predictions[user_id] = [p for p, _ in top_products]
                            gnn4_confidences[user_id] = [s for _, s in top_products]
                
                # Process each user in batch
                for j, user_id in enumerate(batch_users):
                    user_id = int(user_id)  # Convert to int
                    # Get ground truth and historical data
                    ground_truth = test_data_dict.get(user_id, [])
                    historical_data = batch_historical[j]
                    
                    if not ground_truth:
                        continue
                    
                    # Convert ground truth to set for faster lookups
                    gt_set = set(ground_truth)
                    
                    # Get predictions
                    ppo_preds = ppo_predictions.get(user_id, [])
                    gnn4_preds = gnn4_predictions.get(user_id, [])
                    
                    # Calculate hits for individual models
                    ppo_hits = len(set(ppo_preds).intersection(gt_set))
                    gnn4_hits = len(set(gnn4_preds).intersection(gt_set))
                    
                    # Create weighted scoring for each prediction
                    weighted_scores = {}
                    
                    # Score PPO predictions with enhanced logic
                    for pred, conf in zip(ppo_preds, ppo_confidences[user_id]):
                        base_score = self.model_weights['ppo']
                        # Add confidence-based weighting
                        base_score *= (1 + conf * self.confidence_boost)
                        
                        # Add historical context with decay
                        if pred in historical_data['purchase_frequencies']:
                            freq = historical_data['purchase_frequencies'][pred]
                            recency = 1.0
                            if pred in historical_data['last_purchases']:  # Now using set for O(1) lookup
                                recency = self.recency_decay  # Use optimized recency decay
                            base_score += freq * self.historical_context_weight * recency
                        
                        # Add diversity bonus
                        if pred not in weighted_scores:
                            base_score *= self.diversity_bonus  # Use optimized diversity bonus
                        
                        # Add confidence threshold bonus
                        if conf > self.confidence_threshold:  # Use optimized confidence threshold
                            base_score *= self.confidence_boost  # Use optimized confidence boost
                        
                        weighted_scores[pred] = max(weighted_scores.get(pred, 0), base_score)
                    
                    # Score GNN4 predictions with enhanced logic
                    if user_id in gnn4_confidences:  # Add check for user_id in gnn4_confidences
                        for pred, conf in zip(gnn4_preds, gnn4_confidences[user_id]):
                            base_score = self.model_weights['gnn4']
                            # Add confidence-based weighting
                            base_score *= (1 + conf * self.confidence_boost)
                            
                            # Add historical context with decay
                            if pred in historical_data['purchase_frequencies']:
                                freq = historical_data['purchase_frequencies'][pred]
                                recency = 1.0
                                if pred in historical_data['last_purchases']:  # Now using set for O(1) lookup
                                    recency = self.recency_decay  # Use optimized recency decay
                                base_score += freq * self.historical_context_weight * recency
                            
                            # Add diversity bonus
                            if pred not in weighted_scores:
                                base_score *= self.diversity_bonus  # Use optimized diversity bonus
                            
                            # Add confidence threshold bonus
                            if conf > self.confidence_threshold:  # Use optimized confidence threshold
                                base_score *= self.confidence_boost  # Use optimized confidence boost
                            
                            weighted_scores[pred] = max(weighted_scores.get(pred, 0), base_score)
                    else:
                        # If no GNN4 confidences available, use default confidence of 0.5
                        for pred in gnn4_preds:
                            base_score = self.model_weights['gnn4'] * 0.5  # Use default confidence
                            
                            # Add historical context with decay
                            if pred in historical_data['purchase_frequencies']:
                                freq = historical_data['purchase_frequencies'][pred]
                                recency = 1.0
                                if pred in historical_data['last_purchases']:
                                    recency = self.recency_decay  # Use optimized recency decay
                                base_score += freq * self.historical_context_weight * recency
                            
                            # Add diversity bonus
                            if pred not in weighted_scores:
                                base_score *= self.diversity_bonus  # Use optimized diversity bonus
                            
                            weighted_scores[pred] = max(weighted_scores.get(pred, 0), base_score)
                    
                    # Get top predictions based on weighted scores
                    meta_preds = sorted(
                        weighted_scores.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:self.basket_size]
                    
                    # Extract just the predictions
                    meta_preds = [p for p, _ in meta_preds]
                    
                    # Calculate meta hits
                    meta_hits = len(set(meta_preds).intersection(gt_set))
                    model_correct['meta'] += meta_hits
                    model_total['meta'] += len(ground_truth)
                    
                    # Update model performance tracking
                    model_correct['ppo'] += ppo_hits
                    model_correct['gnn4'] += gnn4_hits
                    model_total['ppo'] += len(ground_truth)
                    model_total['gnn4'] += len(ground_truth)
                    
                    # Print detailed information for first few users
                    if batch_num <= 2 and j < 3:
                        print(f"\nUser {user_id}:")
                        print(f"  Ground Truth: {ground_truth}")
                        print(f"  Historical Purchases: {historical_data['purchase_frequencies']}")
                        print(f"  PPO Preds ({ppo_hits}/{len(ground_truth)} hits): {ppo_preds}")
                        print(f"  GNN4 Preds ({gnn4_hits}/{len(ground_truth)} hits): {gnn4_preds}")
                        print(f"  Meta Preds ({meta_hits}/{len(ground_truth)} hits): {meta_preds}")
                    
                    # Calculate reward as improvement over individual models
                    reward = meta_hits - max(ppo_hits, gnn4_hits)
                    
                    # Apply optimized reward scaling
                    if reward > 0:
                        reward *= self.positive_reward_scale
                    else:
                        reward *= self.negative_reward_scale
                    
                    total_reward += reward
                    processed_users += 1
                
                # Update weights based on performance with more aggressive updates
                if model_total['ppo'] > 0 and model_total['gnn4'] > 0:
                    # Calculate current accuracies
                    ppo_acc = model_correct['ppo'] / model_total['ppo']
                    gnn4_acc = model_correct['gnn4'] / model_total['gnn4']
                    meta_acc = model_correct['meta'] / model_total['meta'] if model_total['meta'] > 0 else 0
                    
                    # Update model accuracies with more aggressive moving average
                    alpha = 0.2  # Increased from 0.05
                    self.model_accuracies['ppo'] = (1 - alpha) * self.model_accuracies['ppo'] + alpha * ppo_acc
                    self.model_accuracies['gnn4'] = (1 - alpha) * self.model_accuracies['gnn4'] + alpha * gnn4_acc
                    
                    # Adjust weights based on accuracy ratio with more aggressive updates
                    total_accuracy = self.model_accuracies['ppo'] + self.model_accuracies['gnn4']
                    if total_accuracy > 0:
                        new_ppo_weight = self.model_accuracies['ppo'] / total_accuracy
                        new_gnn4_weight = self.model_accuracies['gnn4'] / total_accuracy
                        
                        # Apply more aggressive weight updates
                        self.model_weights['ppo'] = (1 - alpha) * self.model_weights['ppo'] + alpha * new_ppo_weight
                        self.model_weights['gnn4'] = (1 - alpha) * self.model_weights['gnn4'] + alpha * new_gnn4_weight
                    
                    # Print batch summary
                    print(f"\nBatch {batch_num} Summary:")
                    print(f"PPO Accuracy: {ppo_acc:.4f}")
                    print(f"GNN4 Accuracy: {gnn4_acc:.4f}")
                    print(f"Meta Accuracy: {meta_acc:.4f}")
                    print(f"Current Weights - PPO: {self.model_weights['ppo']:.3f}, GNN4: {self.model_weights['gnn4']:.3f}")
            
            # Track best weights for this episode
            if model_total['meta'] > 0:
                meta_acc = model_correct['meta'] / model_total['meta']
                if meta_acc > best_accuracy:
                    best_accuracy = meta_acc
                    best_ppo_weight = self.model_weights['ppo']
                    best_gnn4_weight = self.model_weights['gnn4']
                    
                    # Save best weights to file
                    with open('meta_learner_weights.json', 'w') as f:
                        json.dump({
                            'weights': self.model_weights,
                            'accuracies': self.model_accuracies,
                            'best_accuracy': best_accuracy
                        }, f)
            
            # Calculate average reward and store episode statistics
            avg_reward = total_reward / processed_users if processed_users > 0 else 0
            episode_rewards.append(avg_reward)
            episode_weights['ppo'].append(self.model_weights['ppo'])
            episode_weights['gnn4'].append(self.model_weights['gnn4'])
            episode_accuracies['ppo'].append(self.model_accuracies['ppo'])
            episode_accuracies['gnn4'].append(self.model_accuracies['gnn4'])
            
            # Print episode summary
            print(f"\n{'='*80}")
            print(f"Episode {episode + 1}/{num_episodes} Summary")
            print(f"{'='*80}")
            print(f"Average Reward: {avg_reward:.4f}")
            print(f"PPO Accuracy: {self.model_accuracies['ppo']:.4f}")
            print(f"GNN4 Accuracy: {self.model_accuracies['gnn4']:.4f}")
            print(f"Meta Accuracy: {model_correct['meta'] / model_total['meta'] if model_total['meta'] > 0 else 0:.4f}")
            print(f"Model Weights - PPO: {self.model_weights['ppo']:.3f}, GNN4: {self.model_weights['gnn4']:.3f}")
            
            # Clear memory after each episode
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Set to best weights found across all episodes
        print(f"\nBest weights found across all episodes - PPO: {best_ppo_weight:.3f}, GNN4: {best_gnn4_weight:.3f}")
        print(f"Best accuracy: {best_accuracy:.4f}")
        self.model_weights['ppo'] = best_ppo_weight
        self.model_weights['gnn4'] = best_gnn4_weight
        
        # Store the episode statistics
        self.episode_rewards = episode_rewards
        self.episode_weights = episode_weights
        self.episode_accuracies = episode_accuracies
        
        return self

    def _calculate_reward_simple(self, ground_truth, ppo_preds, gnn4_preds):
        """A much simpler reward function based directly on hits"""
        if not ground_truth:
            return 0.0
        
        # Convert to sets for faster intersection
        gt_set = set(ground_truth)
        ppo_set = set(ppo_preds)
        gnn4_set = set(gnn4_preds)
        
        # Calculate hits
        ppo_hits = len(ppo_set.intersection(gt_set))
        gnn4_hits = len(gnn4_set.intersection(gt_set))
        
        # Create weighted scoring for each prediction
        weighted_scores = {}
        
        # Score PPO predictions with enhanced logic
        for pred in ppo_preds:
            # Base score is just the model weight
            score = self.model_weights['ppo']
            # Massive bonus for correct predictions
            if pred in gt_set:
                score *= 10  # 10x multiplier for correct predictions
            weighted_scores[pred] = score
        
        # Score GNN4 predictions with enhanced logic
        for pred in gnn4_preds:
            # Base score is just the model weight
            score = self.model_weights['gnn4']
            # Massive bonus for correct predictions
            if pred in gt_set:
                score *= 10  # 10x multiplier for correct predictions
            if pred in weighted_scores:
                # If prediction exists, take the higher score
                weighted_scores[pred] = max(weighted_scores[pred], score)
            else:
                weighted_scores[pred] = score
        
        # Get top predictions based on weighted scores
        meta_preds = sorted(
            weighted_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.basket_size]
        
        # Extract just the predictions
        meta_preds = [p for p, _ in meta_preds]
        
        # Calculate meta hits
        meta_hits = len(set(meta_preds).intersection(gt_set))
        
        # Reward is improvement over individual models
        base_reward = meta_hits - max(ppo_hits, gnn4_hits)
        
        # Apply scaling
        if base_reward > 0:
            return base_reward * 2
        else:
            return base_reward * 3

# Main execution
if __name__ == "__main__":
    # Create environment
    basket_size = 20
    env = NBREnvironment(basket_size=basket_size)
    
    # Load both models first
    gnn4, graph_data, ppo_agent = load_models(gnn4_model_dir=f"model_checkpoints_b{basket_size}_mls", ppo_model_name=f"final_model_b{basket_size}_mls")
    
    # Create meta-learner
    meta_learner = MetaLearner(basket_size=basket_size)
    
    # Set the agents for the meta-learner
    meta_learner.set_agents(ppo_agent, gnn4, graph_data)
    
    send_email("Starting training on metalearner", "Training has started on metalearner")
    
    # Train meta-learner with optimized hyperparameters
    print("\nStarting meta-learner training with optimized hyperparameters...")
    meta_learner.train_simple(
        num_episodes=48,  # From optimization run
        batch_size=37,    # From optimization run
        subset_size=51    # experiences_per_episode from optimization run
    ).process_all_users(save_predictions=True)
    
    # Print final model weights and recalls
    print("\nFinal Model Weights:")
    print(f"PPO: {meta_learner.model_weights['ppo']:.3f}")
    print(f"GNN4: {meta_learner.model_weights['gnn4']:.3f}")
    print("\nFinal Model Recalls:")
    print(f"PPO: {meta_learner.model_accuracies['ppo']:.3f}")
    print(f"GNN4: {meta_learner.model_accuracies['gnn4']:.3f}")
    
    send_email("Training on metalearner completed", "Training has completed on metalearner")

