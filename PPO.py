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
from mail import send_email
from datetime import datetime
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
    def __init__(self, train_file="trueVers/19k_train_baskets.csv", test_file="trueVers/19k_test_baskets.csv", basket_size=10):
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
        
        # Calculate max_product_id from training data
        # Get all product columns (excluding user_id and order_id)
        product_columns = [col for col in self.train_df.columns if col not in ['user_id', 'order_id']]
        # Get max product ID from all product columns and ensure it's an integer
        self.max_product_id = int(self.train_df[product_columns].max().max())
        print(f"Max product ID: {self.max_product_id}")
        
        # Define observation space
        self.observation_space = spaces.Dict({
            "user_id": spaces.Discrete(len(self.users)),  # User ID
            "order_history": spaces.Box(
                low=0,  # Changed from 1 to 0 to allow padding
                high=self.max_product_id,
                shape=(100, 50),  # Fixed shape: (max_orders, max_items_per_order)
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
                high=31,
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
        """Calculate reward based on individual item predictions"""
        test_order = self.test_df[self.test_df['user_id'] == self.current_user]
        
        if test_order.empty:
            return 0  # no reward change for no test order
        
        ground_truth = [int(p) for p in test_order.iloc[0, 2:] if p > 0]
        
        if not ground_truth:
            return 0  # no reward change for empty ground truth
        
        ground_truth_set = set(ground_truth)
        predicted_set = set(action)
        
        # Calculate intersection and precision/recall
        intersection = len(ground_truth_set.intersection(predicted_set))
        
        # Return -1 if there are no correct predictions
        if intersection == 0:
            return -1.0
            
        precision = intersection / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = intersection / len(ground_truth_set) if len(ground_truth_set) > 0 else 0
        
        # Base reward on F1 score (non-negative)
        f1 = 2 * (precision * recall) / (precision + recall)
        base_reward = f1 * 10.0  # Scale to [0, 10]
        
        # Add position-based rewards
        position_rewards = 0
        for i, item in enumerate(action):
            if item in ground_truth_set:
                position_rewards += 1.0 / (i + 1)  # Higher reward for correct early predictions
        
        total_reward = base_reward + position_rewards
        
        # print(f"Prediction: {action}")
        # print(f"Ground truth: {ground_truth}")
        # print(f"F1 score: {f1}")
        # print(f"Total reward: {total_reward}\n")
        
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
        # Handle empty or None order_history
        if order_history is None or len(order_history) == 0:
            device = next(self.parameters()).device
            return torch.zeros(1, self.combiner[0].out_features, device=device)
        
        batch_size = len(order_history)
        device = next(self.parameters()).device
        
        # Process each user's order history separately
        encoded_users = []
        
        for i in range(batch_size):
            user_orders = order_history[i]
            num_orders = len(user_orders) if user_orders is not None else 0
            
            if num_orders == 0:
                # If no order history, use zeros for order encoding
                order_encoding = torch.zeros(1, self.order_rnn.hidden_size * 2, device=device)
            else:
                # Enhanced representation for each order with self-attention
                order_embeddings = []
                for order in user_orders:
                    # Skip None orders
                    if order is None:
                        continue
                        
                    # Convert single integer to list
                    if isinstance(order, (int, np.integer)):
                        order = [order]
                    # Skip empty orders
                    elif len(order) == 0:
                        continue
                    
                    # Convert order to tensor and get embeddings
                    order_tensor = torch.tensor(order, dtype=torch.long, device=device)
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
                
                # If no valid order embeddings, use zeros
                if not order_embeddings:
                    order_encoding = torch.zeros(1, self.order_rnn.hidden_size * 2, device=device)
                else:
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
            if i < len(days_between) and days_between[i] is not None and len(days_between[i]) > 0:
                current_days_between = days_between[i]
                days_mean = torch.tensor(current_days_between, dtype=torch.float, device=device).mean().reshape(1, 1)
                days_std = torch.tensor(current_days_between, dtype=torch.float, device=device).std().reshape(1, 1) if len(current_days_between) > 1 else torch.zeros(1, 1, device=device)
            else:
                # Use default values if index is out of range
                days_mean = torch.zeros(1, 1, device=device)
                days_std = torch.zeros(1, 1, device=device)
            
            # Ensure proper indexing for days_since_last
            if i < len(days_since_last) and days_since_last[i] is not None:
                recency = torch.tensor(days_since_last[i], dtype=torch.float, device=device).reshape(1, 1)
            else:
                recency = torch.zeros(1, 1, device=device)
            
            # Create expanded temporal features
            temporal_features = torch.cat([
                days_mean,  # Average days between orders
                recency,    # Days since last order
            ], dim=1)
            
            temporal_encoding = self.temporal_encoder(temporal_features)
            
            # Combine features with attention-based fusion
            user_encoding = self.combiner(torch.cat([order_encoding, temporal_encoding], dim=1))
            encoded_users.append(user_encoding)
        
        # Ensure we have at least one encoded user
        if not encoded_users:
            # If no encoded users, create a zero tensor with the correct shape
            device = next(self.parameters()).device
            return torch.zeros(1, self.combiner[0].out_features, device=device)
        
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


# Add a Residual Block for better gradient flow
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


class PPOAgent:
    """PPO agent for next basket recommendation with enhanced learning capabilities"""
    def __init__(self, env, hidden_dim=256, embedding_dim=128, lr=3e-4, gamma=0.99, 
                 epsilon=0.2, epochs=5, batch_size=128, save_dir='models', best_model_dir=None, num_workers=None,
                 seed=42):
        # Set seed for reproducibility
        self.seed = seed
        set_seed(seed)
        
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_dir = save_dir
        
        # Create best model directory with basket size
        self.best_model_dir = f"PPO_model_{env.basket_size}_mls" if best_model_dir is None else best_model_dir
        self.training = True
        
        # Convergence checking parameters
        self.convergence_window = 10  # Number of episodes to check for convergence
        self.convergence_threshold = 0.02  # Minimum change in average reward to consider converged (2%)
        self.min_episodes = 50  # Minimum number of episodes before checking convergence
        self.recent_rewards = []  # Store recent rewards for convergence checking
        
        # Create enhanced model with larger dimensions
        self.model = NextBasketPredictor(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            max_items=env.max_product_id,
            basket_size=env.basket_size
        )
        
        # Use AdamW with cosine annealing for better optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-5  # Increased epsilon for better numerical stability
        )
        
        # Add cosine annealing scheduler with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=5,  # Restart every 5 episodes
            T_mult=2,  # Double the restart interval after each restart
            eta_min=lr/50  # Lower minimum learning rate
        )
        
        # Create directories if they don't exist
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)
        
        # For tracking rewards over time
        self.episode_rewards = []
        self.cumulative_rewards = []
        self.avg_rewards = []
        self.f1_scores = []
        
        # Track best model performance
        self.best_reward = float('-inf')
        
        # For pattern detection, use a frequency counter for items
        self.item_frequencies = {}
        
        # For train-test data similarity analysis
        self.train_item_counts = None
        self.test_item_counts = None
        
        # Epsilon-based exploration parameters
        self.exploration_epsilon = 1.0  # Start with moderate epsilon
        self.epsilon_decay = 0.995  # Slower decay for better exploration
        self.epsilon_min = 0.1  # Minimum epsilon value
        self.ordered_item_threshold = 3.0  # Lower threshold to encourage more exploration
        self.exploration_noise = 0.1  # Initial exploration noise
        self.noise_decay = 0.995  # Noise decay rate
        
        # Track unique items seen per user
        self.user_item_history = {}
        
        # Use a single process for now to avoid memory issues
        self.num_workers = 1
    
    def actions_to_csv(self,train_data_file='full_train_baskets.csv', output_file='ppo_model_predictions.csv', minUser = 1, maxUser = 206210, basket_size = 10):
        """
        Process all users from the training data and save predictions to a CSV file.
        
        Args:
            train_data_file: Path to the CSV file containing training data
            output_file: Path to save the predictions CSV
            minUser: Minimum user ID to process
            maxUser: Maximum user ID to process
            basket_size: Number of items to predict for each user
        """
        # Set agent to evaluation mode
        was_training = self.training
        self.eval_mode()
        
        # Create output file with timestamp
        fileName = f"PPO_actions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        print(f"Creating predictions file: {fileName}")
        
        # Get unique users in the specified range
        all_users = sorted(self.env.train_df['user_id'].unique())
        users_in_range = [user for user in all_users if minUser <= user <= maxUser]
        print(f"Found {len(users_in_range)} users in range {minUser}-{maxUser}")
        
        # Create CSV file with header
        with open(fileName, 'w') as f:
            # Write header
            header = ['user_id'] + [f'product{i+1}' for i in range(basket_size)]
            f.write(','.join(header) + '\n')
            
            # Process each user
            for i, user_id in enumerate(users_in_range):
                if i % 100 == 0:
                    print(f"Processing user {i}/{len(users_in_range)} (user_id: {user_id})")
                
                # Get user's order history from training data
                user_orders = self.env.train_df[self.env.train_df['user_id'] == user_id].sort_values('order_id')
                
                # Skip users with no order history
                if user_orders.empty:
                    continue
                
                # Create observation for the user
                obs = self.env._get_user_observation(user_id)
                
                # Get prediction and confidence scores
                predictions, _ = self.predict_with_confidence(obs)
                
                # Ensure we have the right number of predictions
                if len(predictions) < basket_size:
                    # Pad with zeros if needed
                    predictions = np.pad(predictions, (0, basket_size - len(predictions)), 'constant')
                elif len(predictions) > basket_size:
                    # Truncate if we have too many
                    predictions = predictions[:basket_size]
                
                # Write to CSV
                row = [str(user_id)] + [str(p) for p in predictions]
                f.write(','.join(row) + '\n')
        
        print(f"Predictions saved to {fileName}")
        
        # Restore previous training mode
        if was_training:
            self.train_mode()
        
        return fileName
    
    def analyze_datasets(self):
        """Analyze dataset patterns to enhance learning"""
        print("Analyzing dataset patterns...")
        
        # Count item frequencies in training set
        train_items = []
        for _, row in self.env.train_df.iterrows():
            items = [int(p) for p in row.iloc[2:] if p > 0]
            train_items.extend(items)
        
        train_counts = pd.Series(train_items).value_counts()
        self.train_item_counts = train_counts
        
        # Count item frequencies in test set
        test_items = []
        for _, row in self.env.test_df.iterrows():
            items = [int(p) for p in row.iloc[2:] if p > 0]
            test_items.extend(items)
            
        test_counts = pd.Series(test_items).value_counts()
        self.test_item_counts = test_counts
        
        # Find most common items in training set
        top_items = train_counts.head(50).index.tolist()
        print(f"Top 50 most frequent items: {top_items}")
        
        # Calculate overlap between train and test sets
        common_items = set(train_counts.index) & set(test_counts.index)
        train_only = set(train_counts.index) - set(test_counts.index)
        test_only = set(test_counts.index) - set(train_counts.index)
        
        print(f"Items in both train and test: {len(common_items)}")
        print(f"Items only in train: {len(train_only)}")
        print(f"Items only in test: {len(test_only)}")
        
        return {
            "top_items": top_items,
            "common_items": common_items,
            "train_only": train_only,
            "test_only": test_only
        }
        
    def preprocess_observation(self, obs):
        """Prepare observation for model input with additional feature engineering"""
        order_history = obs["order_history"]
        days_between = obs["days_between_orders"]
        days_since_last = obs["days_since_last"]
        
        # During training, update item frequency counter
        if self.training:
            for basket in order_history:
                for item in basket:
                    self.item_frequencies[item] = self.item_frequencies.get(item, 0) + 1
        
        return (order_history, days_between, days_since_last)
    
    def get_action(self, obs):
        order_history, days_between, days_since_last = self.preprocess_observation(obs)
        user_id = obs["user_id"]
        
        with torch.no_grad():
            # Ensure proper list wrapping for single observation
            if not isinstance(order_history, list):
                order_history = [order_history]
            if not isinstance(days_between, list):
                days_between = [days_between]
            if not isinstance(days_since_last, list):
                days_since_last = [days_since_last]
                
            item_probs, _ = self.model(order_history, days_between, days_since_last)
            
            # Remove padding token
            item_probs = item_probs[0, 1:]
            
            # Get user's previous items
            user_items = set()
            if order_history and len(order_history) > 0 and len(order_history[0]) > 0:
                for basket in order_history[0]:  # Access the first (and only) element
                    # Handle case where basket might be a single integer instead of an iterable
                    if isinstance(basket, (list, tuple, np.ndarray)):
                        user_items.update(basket)
                    else:
                        # If basket is a single integer, add it directly
                        user_items.add(basket)
            
            # Track user's item history for more personalized recommendations
            if user_id not in self.user_item_history:
                self.user_item_history[user_id] = set()
            self.user_item_history[user_id].update(user_items)
            
            # If no user history, use popular items + exploration
            if not user_items:
                if hasattr(self, 'train_item_counts') and self.train_item_counts is not None:
                    # Boost probabilities for popular items
                    pop_items_mask = torch.zeros_like(item_probs)
                    for item, count in self.train_item_counts.items()[:100]:  # Use top 100 popular items
                        if 1 <= item <= self.env.max_product_id:
                            pop_items_mask[item-1] = count / self.train_item_counts.max()
                    
                    # Blend model predictions with popularity
                    item_probs = item_probs * (1.0 + pop_items_mask)
                
                # Add exploration noise during training
                if self.training:
                    # Add Dirichlet noise for exploration
                    exploration_noise = torch.tensor(np.random.dirichlet([0.6] * len(item_probs)), dtype=torch.float32)
                    item_probs = 0.75 * item_probs + 0.25 * exploration_noise
                
                # Get top items and their probabilities for recommendation
                top_probs, top_indices = torch.topk(item_probs, self.env.basket_size)
                
                # Normalize confidence scores to make them more meaningful
                # Scale to a range that's more intuitive (0.1 to 1.0)
                min_prob = 0.1
                max_prob = 1.0
                normalized_probs = min_prob + (max_prob - min_prob) * (top_probs - top_probs.min()) / (top_probs.max() - top_probs.min() + 1e-8)
                
                actions = top_indices.tolist()
                actions = np.array(actions) + 1
                confidences = normalized_probs.tolist()
                return actions, confidences
            
            # Create epsilon-based strategy for users with history
            
            # Create mask for previously bought items
            prev_items_mask = torch.zeros_like(item_probs)
            for item in user_items:
                if item < len(prev_items_mask):
                    prev_items_mask[item-1] = 1.0
            
            # Identify previously ordered items and their probabilities
            prev_indices = torch.nonzero(prev_items_mask).squeeze(-1)
            prev_items_count = len(prev_indices)
            
            # Check if we should explore new items based on epsilon
            explore_new_items = False
            if self.training:
                if prev_items_count < self.env.basket_size:
                    # Always explore if we don't have enough previous items
                    explore_new_items = True
                elif self.exploration_epsilon <= self.ordered_item_threshold:
                    # Start exploring new items if epsilon is below threshold
                    explore_new_items = True
            
            # In evaluation mode, still use some new items
            elif prev_items_count < self.env.basket_size or np.random.random() < 0.2:
                explore_new_items = True
            
            # Extract probabilities for previous items
            if prev_items_count > 0:
                prev_probs = item_probs[prev_indices]
                
                # Select the highest probability items from previous purchases
                k = min(self.env.basket_size, prev_items_count)
                top_prev_probs, top_prev_indices = torch.topk(prev_probs, k)
                selected_prev_items = [prev_indices[i].item() for i in top_prev_indices]
                
                # Normalize previous item confidence scores
                min_prob = 0.1
                max_prob = 1.0
                normalized_prev_probs = min_prob + (max_prob - min_prob) * (top_prev_probs - top_prev_probs.min()) / (top_prev_probs.max() - top_prev_probs.min() + 1e-8)
                selected_prev_confidences = normalized_prev_probs.tolist()
            else:
                selected_prev_items = []
                selected_prev_confidences = []
                
            # If we have enough previous items and shouldn't explore, use only those
            if len(selected_prev_items) >= self.env.basket_size and not explore_new_items:
                actions = np.array(selected_prev_items[:self.env.basket_size]) + 1
                confidences = selected_prev_confidences[:self.env.basket_size]
                return actions, confidences
            
            # Otherwise, fill the remaining spots with new items
            remaining_spots = self.env.basket_size - len(selected_prev_items)
            
            if remaining_spots > 0:
                # Create mask for new items (those not previously purchased)
                new_items_mask = 1.0 - prev_items_mask
                
                # If training, use popularity to guide new item selection
                if self.training and hasattr(self, 'train_item_counts') and self.train_item_counts is not None:
                    # Create frequency-based popularity mask
                    pop_boost = torch.zeros_like(item_probs)
                    for item, count in self.train_item_counts.items():
                        if 1 <= item <= self.env.max_product_id:
                            pop_boost[item-1] = count / self.train_item_counts.max()
                    
                    # Boost new items based on their popularity
                    new_item_probs = item_probs * new_items_mask * (1.0 + 0.5 * pop_boost)
                else:
                    # Just use model predictions for new items
                    new_item_probs = item_probs * new_items_mask
                
                # Zero out already selected items
                for idx in selected_prev_items:
                    new_item_probs[idx] = 0
                
                # Add exploration noise during training
                if self.training:
                    # Get non-zero indices
                    valid_indices = torch.nonzero(new_item_probs).squeeze(-1)
                    if len(valid_indices) > 0:
                        # Add noise only to valid indices
                        noise = torch.zeros_like(new_item_probs)
                        noise_values = torch.tensor(np.random.dirichlet([0.5] * len(valid_indices)), dtype=torch.float32)
                        noise[valid_indices] = noise_values
                        new_item_probs = 0.8 * new_item_probs + 0.2 * noise
                
                # Select top new items
                if new_item_probs.sum() > 0:
                    top_new_probs, new_indices = torch.topk(new_item_probs, remaining_spots)
                    new_items = new_indices.tolist()
                    
                    # Normalize new item confidence scores
                    min_prob = 0.1
                    max_prob = 1.0
                    normalized_new_probs = min_prob + (max_prob - min_prob) * (top_new_probs - top_new_probs.min()) / (top_new_probs.max() - top_new_probs.min() + 1e-8)
                    new_confidences = normalized_new_probs.tolist()
                else:
                    # Fallback to raw predictions if no valid new items
                    top_new_probs, fallback_indices = torch.topk(item_probs, remaining_spots)
                    new_items = fallback_indices.tolist()
                    
                    # Normalize fallback confidence scores
                    min_prob = 0.1
                    max_prob = 1.0
                    normalized_fallback_probs = min_prob + (max_prob - min_prob) * (top_new_probs - top_new_probs.min()) / (top_new_probs.max() - top_new_probs.min() + 1e-8)
                    new_confidences = normalized_fallback_probs.tolist()
                
                # Combine previous and new items
                all_indices = selected_prev_items + new_items
                all_confidences = selected_prev_confidences + new_confidences
                actions = np.array(all_indices) + 1
                confidences = all_confidences
            else:
                actions = np.array(selected_prev_items) + 1
                confidences = selected_prev_confidences
            
            # Update exploration epsilon during training
            if self.training:
                self.exploration_epsilon *= self.epsilon_decay
                self.exploration_epsilon = max(self.exploration_epsilon, self.epsilon_min)
            
            return actions, confidences
    
    def train_step(self, obs_batch, action_batch, reward_batch, next_obs_batch, done_batch):
        """Perform one PPO update step with improved optimization"""
        # Preprocess all observations
        order_history_batch = [obs["order_history"] for obs in obs_batch]
        days_between_batch = [obs["days_between_orders"] for obs in obs_batch]
        days_since_last_batch = [obs["days_since_last"] for obs in obs_batch]
        
        next_order_history_batch = [obs["order_history"] for obs in next_obs_batch]
        next_days_between_batch = [obs["days_between_orders"] for obs in next_obs_batch]
        next_days_since_last_batch = [obs["days_since_last"] for obs in next_obs_batch]
        
        # Convert rewards and dones to tensors
        rewards = torch.FloatTensor(reward_batch)
        dones = torch.FloatTensor(done_batch)
        
        # Get current item probabilities and values
        with torch.no_grad():
            curr_item_probs, curr_values = self.model(
                order_history_batch,
                days_between_batch,
                days_since_last_batch
            )
            curr_values = curr_values.squeeze(-1)
            
            # Get values for next observations
            _, next_values = self.model(
                next_order_history_batch,
                next_days_between_batch,
                next_days_since_last_batch
            )
            next_values = next_values.squeeze(-1)
        
        # Calculate advantages and returns
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Enhanced GAE calculation with better lambda and normalization
        gae = 0
        lambda_gae = 0.95  # GAE lambda parameter
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = curr_values[t+1]
                next_non_terminal = 1.0
            
            # Calculate TD error
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - curr_values[t]
            
            # Update GAE
            gae = delta + self.gamma * lambda_gae * next_non_terminal * gae
            
            # Store advantage
            advantages[t] = gae
            
            # Calculate returns
            returns[t] = advantages[t] + curr_values[t]
        
        # Normalize advantages with better numerical stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Clip advantages to prevent extreme values
        advantages = torch.clamp(advantages, -10.0, 10.0)
        
        # Prepare action data - convert to one-hot masks
        action_masks = []
        for action in action_batch:
            # Create a mask for current action
            mask = torch.zeros(1, self.env.max_product_id + 1)
            
            # Mark selected items as 1
            for item in action:
                if 1 <= item <= self.env.max_product_id:  # Ensure valid item ID
                    mask[0, item] = 1
                    
            action_masks.append(mask)
        
        action_masks = torch.cat(action_masks, dim=0)
        
        # Store old action probabilities
        with torch.no_grad():
            old_item_probs, _ = self.model(
                order_history_batch,
                days_between_batch,
                days_since_last_batch
            )
            # Calculate probabilities for the chosen actions
            old_probs_selected = torch.sum(old_item_probs * action_masks, dim=1) / (torch.sum(action_masks, dim=1) + 1e-8)
        
        # Track losses for reporting
        epoch_actor_losses = []
        epoch_critic_losses = []
        epoch_entropies = []
        
        # Multiple epochs of PPO updates
        for epoch in range(self.epochs):
            # Get new probabilities and values with fresh computation graph
            new_item_probs, new_values = self.model(
                order_history_batch,
                days_between_batch,
                days_since_last_batch
            )
            new_values = new_values.squeeze(-1)
            
            # Calculate new probabilities for chosen actions
            new_probs_selected = torch.sum(new_item_probs * action_masks, dim=1) / (torch.sum(action_masks, dim=1) + 1e-8)
            
            # Calculate ratio (importance sampling)
            ratio = new_probs_selected / (old_probs_selected + 1e-8)
            
            # Calculate surrogate losses for PPO with better clipping
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages.detach()
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss with Huber loss and value clipping
            value_pred_clipped = curr_values + (new_values - curr_values).clamp(-self.epsilon, self.epsilon)
            value_losses = (new_values - returns.detach()).pow(2)
            value_losses_clipped = (value_pred_clipped - returns.detach()).pow(2)
            critic_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
            
            # Entropy bonus for exploration with adaptive coefficient
            entropy = -torch.sum(new_item_probs * torch.log(new_item_probs + 1e-8), dim=1).mean()
            entropy_coef = max(0.01, 0.05 * (1.0 - epoch / self.epochs))  # Decrease entropy bonus over epochs
            
            # L2 regularization for better generalization
            l2_reg = 0.0
            for param in self.model.parameters():
                l2_reg += torch.norm(param)
            
            # Combine losses with better weighting
            total_loss = (
                actor_loss + 
                0.5 * critic_loss - 
                entropy_coef * entropy + 
                0.01 * l2_reg  # Add L2 regularization
            )
            
            # Optimize with gradient clipping
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()

            # Store losses
            epoch_actor_losses.append(actor_loss.item())
            epoch_critic_losses.append(critic_loss.item())
            epoch_entropies.append(entropy.item())
        
        # Print training statistics
        print(f"\nTraining Stats:")
        print(f"Actor Loss: {np.mean(epoch_actor_losses):.4f}")
        print(f"Critic Loss: {np.mean(epoch_critic_losses):.4f}")
        print(f"Entropy: {np.mean(epoch_entropies):.4f}\n")
    
    def collect_experiences(self, obs):
        """Collect experiences sequentially"""
        action, confidences = self.get_action(obs)
        next_obs, reward, done, _, _ = self.env.step(action)
        return obs, action, confidences, reward, next_obs, done

    def check_convergence(self):
        """Check if the model has converged based on sum of rewards over past 10 episodes"""
        if len(self.recent_rewards) < self.convergence_window:
            return False
            
        # Calculate sum of rewards over the window
        recent_sum = np.sum(self.recent_rewards[-self.convergence_window:])
        prev_sum = np.sum(self.recent_rewards[-2*self.convergence_window:-self.convergence_window])
        
        # Calculate percentage change
        if prev_sum == 0:
            return False
            
        change = abs(recent_sum - prev_sum) / abs(prev_sum)
        
        # Print convergence metrics
        print(f"\nConvergence Check:")
        print(f"Recent sum of rewards: {recent_sum:.4f}")
        print(f"Previous sum of rewards: {prev_sum:.4f}")
        print(f"Change: {change:.4f}")
        print(f"Threshold: {self.convergence_threshold}")
        
        return change < self.convergence_threshold

    def train(self, num_episodes=100):
        """Train the agent with sequential experience collection"""
        # Set seed for reproducibility at the start of training
        set_seed(self.seed)
        
        # Analyze datasets for patterns before training
        dataset_patterns = self.analyze_datasets()
        
        # Reset exploration epsilon at the start of training
        self.exploration_epsilon = 1.0
        
        total_cumulative_reward = 0
        self.recent_rewards = []  # Reset recent rewards
        
        for episode in range(num_episodes):
            # Set seed for each episode to ensure reproducibility
            set_seed(self.seed + episode)
            
            obs_batch = []
            action_batch = []
            confidence_batch = []
            reward_batch = []
            next_obs_batch = []
            done_batch = []
            f1_scores = []
            
            obs, _ = self.env.reset()
            episode_reward = 0
            steps = 0
            
            print(f"Episode {episode+1}/{num_episodes}: Collecting experiences...")
            print(f"Current exploration epsilon: {self.exploration_epsilon:.2f}")
            
            # Collect experiences sequentially
            while len(obs_batch) < self.batch_size:
                obs, action, confidences, reward, next_obs, done = self.collect_experiences(obs)
                
                obs_batch.append(obs)
                action_batch.append(action)
                confidence_batch.append(confidences)
                reward_batch.append(reward)
                next_obs_batch.append(next_obs)
                done_batch.append(done)
                
                episode_reward += reward
                steps += 1
                
                # Calculate F1 score
                order_history = obs["order_history"]
                user_id = obs["user_id"]
                test_order = self.env.test_df[self.env.test_df['user_id'] == user_id]
                
                if not test_order.empty:
                    ground_truth = [int(p) for p in test_order.iloc[0, 2:] if p > 0]
                    if ground_truth:
                        ground_truth_set = set(ground_truth)
                        predicted_set = set(action)
                        
                        intersection = len(ground_truth_set.intersection(predicted_set))
                        precision = intersection / len(predicted_set) if len(predicted_set) > 0 else 0
                        recall = intersection / len(ground_truth_set) if len(ground_truth_set) > 0 else 0
                        
                        if precision + recall > 0:
                            f1 = 2 * (precision * recall) / (precision + recall)
                            f1_scores.append(f1)
                
                if done:
                    obs, _ = self.env.reset()
                else:
                    obs = next_obs
            
            # Calculate average reward for this episode
            avg_episode_reward = episode_reward / steps
            self.recent_rewards.append(avg_episode_reward)
            
            print(f"Episode {episode+1} - Collected {self.batch_size} experiences. Performing PPO update...")
            
            # Perform PPO update
            self.train_step(
                obs_batch,
                action_batch,
                reward_batch,
                next_obs_batch,
                done_batch
            )
            
            # Update metrics
            total_cumulative_reward += episode_reward
            self.episode_rewards.append(episode_reward)
            self.cumulative_rewards.append(total_cumulative_reward)
            self.avg_rewards.append(avg_episode_reward)
            
            avg_f1 = np.mean(f1_scores) if f1_scores else 0
            self.f1_scores.append(avg_f1)
            
            self.scheduler.step()
            
            print(f"Episode {episode+1}, Average Reward: {avg_episode_reward:.4f}, Avg F1: {avg_f1:.4f}")
            print(f"Cumulative Reward: {total_cumulative_reward:.4f}, Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Check for convergence after minimum episodes
            if episode >= self.min_episodes and self.check_convergence():
                print(f"\nModel converged after {episode+1} episodes!")
                print("Stopping training early...")
                break
            
            if avg_episode_reward > self.best_reward:
                self.best_reward = avg_episode_reward
                try:
                    self.save_model("best_model", is_best=True)  # Save best model in specified directory
                    print(f"New best model saved with reward: {self.best_reward:.4f}")
                except Exception as e:
                    print(f"Failed to save best model: {e}")
                
            if (episode + 1) % 10 == 0:
                try:
                    self.save_model(f"model_ep{episode+1}")  # Regular checkpoints in default directory
                    self.plot_rewards(show_plot=False, include_f1=True)
                except Exception as e:
                    print(f"Failed to save model or plot rewards: {e}")
                
                if len(self.item_frequencies) > 0:
                    top_predicted = sorted(self.item_frequencies.items(), key=lambda x: x[1], reverse=True)[:20]
                    print(f"Top 20 frequently predicted items: {top_predicted}")
        
        # Final plot
        # try:
        #     self.plot_rewards(show_plot=True, include_f1=True)
        # except Exception as e:
        #     print(f"Failed to create final reward plot: {e}")
    
    def predict_with_confidence(self, obs):
        """
        Generate prediction and confidence scores for a given observation.
        This function is designed to be used by meta learners or ensemble methods
        to get both predictions and confidence levels for a given state.
        
        IMPORTANT: This function does NOT update the agent's state or weights.
        It's completely safe to use in ensemble methods without affecting the model.
        
        Args:
            obs: The observation/state to predict for
            
        Returns:
            tuple: (predictions, confidence_scores)
                - predictions: numpy array of predicted item IDs
                - confidence_scores: list of confidence scores for each prediction
        """
        # Store current training state
        was_training = self.training
        
        # Temporarily set to evaluation mode
        self.eval_mode()
        
        # Disable gradient computation to prevent any updates
        with torch.no_grad():
            # Get prediction and confidence scores
            predictions, confidence_scores = self.get_action(obs)
        
        # Restore previous training mode
        if was_training:
            self.train_mode()
            
        return predictions, confidence_scores

    def act(self, obs):
        """Generate next basket prediction without learning (for testing)"""
        order_history, days_between, days_since_last = self.preprocess_observation(obs)
        
        # Ensure proper list wrapping for single observation
        if not isinstance(order_history, list):
            order_history = [order_history]
        if not isinstance(days_between, list):
            days_between = [days_between]
        if not isinstance(days_since_last, list):
            days_since_last = [days_since_last]
        
        return self.model.act(order_history, days_between, days_since_last)[0]
    
    def save_model(self, name="model", is_best=False):
        """Save model weights with error handling"""
        try:
            # Use best_model_dir if this is the best model
            save_path = os.path.join(self.best_model_dir if is_best else self.save_dir, f"{name}.pt")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save model with error handling
            try:
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved to {save_path}")
            except (RuntimeError, IOError) as e:
                print(f"Error saving model to {save_path}: {e}")
                # Try alternative save location
                alt_path = os.path.join(os.getcwd(), f"{name}_backup.pt")
                print(f"Attempting to save to alternative location: {alt_path}")
                torch.save(self.model.state_dict(), alt_path)
                print(f"Model saved to alternative location: {alt_path}")
                
        except Exception as e:
            print(f"Failed to save model: {e}")
            print("Continuing training without saving model")

    def load_model(self, name="model"):
        """Load model weights"""
        path = os.path.join(self.save_dir, f"{name}.pt")
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
        return self

    def train_mode(self):
        """Set agent to training mode"""
        self.training = True
        self.model.train()

    def eval_mode(self):
        """Set agent to evaluation mode"""
        self.training = False
        self.model.eval()

    def plot_rewards(self, show_plot=False, include_f1=False):
        """Enhanced plot of rewards with additional metrics"""
        if include_f1:
            plt.figure(figsize=(15, 10))
            
            # Plot episode rewards
            plt.subplot(2, 2, 1)
            plt.plot(self.episode_rewards, 'b-')
            plt.title('Reward per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.grid(True)
            
            # Plot cumulative rewards
            plt.subplot(2, 2, 2)
            plt.plot(self.cumulative_rewards, 'r-')
            plt.title('Cumulative Reward over Episodes')
            plt.xlabel('Episode')
            plt.ylabel('Cumulative Reward')
            plt.grid(True)
            
            # Plot average rewards
            plt.subplot(2, 2, 3)
            plt.plot(self.avg_rewards, 'g-')
            plt.title('Average Reward per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            plt.grid(True)
            
            # Plot F1 scores
            plt.subplot(2, 2, 4)
            plt.plot(self.f1_scores, 'm-')
            plt.title('Average F1 Score per Episode')
            plt.xlabel('Episode')
            plt.ylabel('F1 Score')
            plt.grid(True)
        else:
            plt.figure(figsize=(12, 6))
            
            # Plot episode rewards
            plt.subplot(1, 2, 1)
            plt.plot(self.episode_rewards, 'b-')
            plt.title('Reward per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.grid(True)
            
            # Plot cumulative rewards
            plt.subplot(1, 2, 2)
            plt.plot(self.cumulative_rewards, 'r-')
            plt.title('Cumulative Reward over Episodes')
            plt.xlabel('Episode')
            plt.ylabel('Cumulative Reward')
            plt.grid(True)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'reward_plot.png'), dpi=300)
        print(f"Reward plot saved to {os.path.join(self.save_dir, 'reward_plot.png')}")
        
        # Option to display the plot interactively
        if show_plot:
            plt.show()
        else:
            plt.close()


# Test function for evaluation
def evaluate_model(agent, env, num_users=100):
    """Evaluate model on test users"""
    # Set agent to evaluation mode
    agent.eval_mode()
    
    rewards = []
    
    # Reset environment
    obs, _ = env.reset()
    
    print(f"Evaluating model on {num_users} users...")
    for i in range(num_users):
        if i > 0 and i % 10 == 0:
            print(f"Evaluated {i}/{num_users} users...")
            
        # Get prediction without learning
        action = agent.act(obs)
        
        # Take step in environment
        next_obs, reward, _, _, _ = env.step(action)
        
        rewards.append(reward)
        obs = next_obs
    
    avg_reward = sum(rewards) / len(rewards)
    print(f"Evaluation on {num_users} users:")
    print(f"Average F1 Score: {avg_reward:.4f}")
    
    # Set agent back to training mode
    agent.train_mode()
    return avg_reward

# Main execution
if __name__ == "__main__":

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    send_email("PPOv3 training started", "PPOv3 training started")
    # Create environment
    basket_size = 10
    env = NBREnvironment(basket_size=basket_size)
    
    # Create enhanced agent with improved exploration
    agent = PPOAgent(
        env, 
        hidden_dim=256,
        embedding_dim=128,
        batch_size=128,
        lr=0.0008388913219637349,
        gamma=0.9562257186205783,
        epsilon=0.2,
        epochs=5,
        save_dir='models'  # Default directory for regular checkpoints
    )
    
    # Configure exploration parameters
    agent.exploration_epsilon = 1.0  # Start with moderate epsilon
    agent.epsilon_decay = 0.995      # Slower decay for better exploration
    agent.epsilon_min = 0.1          # Minimum epsilon value
    agent.ordered_item_threshold = 3.0  # Lower threshold to encourage more exploration
    agent.exploration_noise = 0.1      # Initial exploration noise
    agent.noise_decay = 0.995         # Noise decay rate
    
    # Pre-analyze datasets to identify patterns
    dataset_patterns = agent.analyze_datasets()
    print("\nStarting training with epsilon-based exploration strategy...")
    print(f"Exploration parameters: epsilon={agent.exploration_epsilon}, decay={agent.epsilon_decay}")
    print(f"Will start exploring new items when epsilon drops below {agent.ordered_item_threshold}")
    
    # Train agent
    agent.train(num_episodes=1000)
    
    # Save final model
    agent.save_model(f"final_model_b{basket_size}_mls")
    
    # Plot rewards with F1 scores
    #agent.plot_rewards(show_plot=True, include_f1=True)
    
    # Evaluate model on test users
    print("\nPerforming evaluation on test users...")
    rewards = evaluate_model(agent, env, num_users=50)
    
    # Print final item frequency statistics
    if len(agent.item_frequencies) > 0:
        print("\nTop 20 most frequently predicted items during training:")
        top_predicted = sorted(agent.item_frequencies.items(), key=lambda x: x[1], reverse=True)[:20]
        for item, count in top_predicted:
            print(f"Item {item}: {count} times")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    send_email("PPOv3 training finished", f"PPOv3 training finished at {timestamp}")
    agent.actions_to_csv(output_file=f"PPO_actions_{timestamp}.csv")
    # Recommend top items for a sample of users
    print("\nSample recommendations for a few test users:")
    for _ in range(5):
        obs, _ = env.reset()
        user_id = obs["user_id"]
        order_history = obs["order_history"]
        action, confidences = agent.get_action(obs)
        
        print(f"\nUser ID: {user_id}")
        print(f"Order history (last 3 orders): {order_history[-3:] if len(order_history) > 3 else order_history}")
        print(f"Recommended items: {action}")
        print(f"Confidence scores: {[f'{conf:.4f}' for conf in confidences]}")
        
        # Check which items were previously ordered by this user
        user_items = set()
        for basket in order_history:
            if isinstance(basket, (list, tuple, np.ndarray)):
                user_items.update(basket)
            else:
                user_items.add(basket)
        
        previously_ordered = [item for item in action if item in user_items]
        new_items = [item for item in action if item not in user_items]
        
        print(f"Previously ordered items: {previously_ordered} ({len(previously_ordered)}/{len(action)})")
        print(f"New items: {new_items} ({len(new_items)}/{len(action)})")