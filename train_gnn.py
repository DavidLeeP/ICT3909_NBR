import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque, namedtuple
import gc
import os
import pandas as pd
from datetime import datetime
from mail import send_email
import math
import pickle

# Define GNN Model
class GNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=32, output_dim=32):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.use_checkpointing = False

    def forward(self, x, edge_index):
        # Validate and normalize edge indices
        if edge_index is not None:
            # Ensure edge indices are within bounds
            max_idx = x.size(0) - 1
            edge_index = edge_index.clamp(0, max_idx)
            
            # Remove self-loops and duplicate edges
            edge_index = torch.unique(edge_index, dim=1)
            
            # Ensure edge indices are contiguous
            unique_nodes = torch.unique(edge_index)
            node_mapping = torch.zeros(max_idx + 1, dtype=torch.long, device=edge_index.device)
            node_mapping[unique_nodes] = torch.arange(len(unique_nodes), device=edge_index.device)
            edge_index = node_mapping[edge_index]
            
            # Update x to only include nodes present in edge_index
            x = x[unique_nodes]
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Define RL Agent
class RLAgent(torch.nn.Module):
    def __init__(self, state_dim, action_dim, epsilon=5.0, epsilon_min=0.1, decay_rate=0.999):
        super(RLAgent, self).__init__()
        # Split network into two paths
        self.state_path = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU()
        )
        self.history_path = torch.nn.Sequential(
            torch.nn.Linear(action_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU()
        )
        self.final_layer = torch.nn.Linear(256, action_dim)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate

    def forward(self, state, history=None):
        # Ensure state has batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        state_features = self.state_path(state)
        
        if history is not None:
            # Ensure history has batch dimension
            if history.dim() == 1:
                history = history.unsqueeze(0)
            history_features = self.history_path(history)
            combined = torch.cat([state_features, history_features], dim=-1)
        else:
            # If no history, use zeros with same shape as state_features
            history_features = torch.zeros_like(state_features)
            combined = torch.cat([state_features, history_features], dim=-1)
            
        return self.final_layer(combined)

    def get_top_actions(self, state, top_k=10, user_history=None, blacklist=None, whitelist=None):
        with torch.no_grad():
            # Create history embedding
            if user_history:
                history_embedding = torch.zeros(len(product_nodes), device=state.device)
                for item in user_history:
                    if item in reverse_product_mapping:
                        idx = list(reverse_product_mapping.keys())[list(reverse_product_mapping.values()).index(item)]
                        if idx < len(history_embedding):
                            history_embedding[idx] = 1.0
                history_embedding = history_embedding.unsqueeze(0)  # Add batch dimension
            else:
                history_embedding = None

            q_values = self.forward(state, history_embedding)
            q_values = q_values.squeeze(0)  # Remove batch dimension for action selection
            
            # Apply whitelist boost
            if whitelist:
                for item in whitelist:
                    if item < len(q_values):
                        q_values[item] += 5.0  # Increased boost
            
            # Apply blacklist penalty
            if blacklist:
                for item in blacklist:
                    if item < len(q_values):
                        q_values[item] -= 10.0  # Increased penalty
            
            # Get valid indices
            valid_indices = torch.arange(len(q_values), device=state.device)
            if blacklist:
                valid_indices = valid_indices[~torch.isin(valid_indices, torch.tensor(list(blacklist), device=state.device))]
            
            if len(valid_indices) == 0:
                return random.sample(range(len(q_values)), min(top_k, len(q_values)))
            
            if random.random() < self.epsilon:
                # Exploration with history bias
                valid_q_values = q_values[valid_indices]
                if user_history:
                    # Boost probabilities for items in user history
                    history_mask = torch.zeros_like(valid_q_values)
                    for item in user_history:
                        if item in reverse_product_mapping:
                            idx = list(reverse_product_mapping.keys())[list(reverse_product_mapping.values()).index(item)]
                            if idx in valid_indices:
                                history_mask[valid_indices == idx] = 2.0
                    valid_q_values = valid_q_values + history_mask
                
                probs = F.softmax(valid_q_values, dim=0)
                selected_indices = torch.multinomial(probs, min(top_k, len(valid_indices)))
                return valid_indices[selected_indices].tolist()
            else:
                # Exploitation
                valid_q_values = q_values[valid_indices]
                top_values, top_indices = torch.topk(valid_q_values, k=min(top_k, len(valid_indices)))
                return valid_indices[top_indices].tolist()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.decay_rate, self.epsilon_min)

Transition = namedtuple('Transition', ('state', 'action', 'reward'))

class UserSpecificMemory:
    def __init__(self, capacity, batch_size, device):
        self.memory = defaultdict(lambda: deque(maxlen=capacity))
        self.batch_size = batch_size
        self.total_capacity = capacity
        self.current_size = 0
        self.device = device
        # Add blacklist and whitelist tracking
        self.blacklist = defaultdict(set)
        self.whitelist = defaultdict(set)
        self.max_list_size = 1000  # Maximum number of items to keep in each list

    def append(self, user_id, state, action, reward, true_items=None):
        if self.current_size >= self.total_capacity:
            oldest_user = next(iter(self.memory))
            self.current_size -= len(self.memory[oldest_user])
            del self.memory[oldest_user]
            # Also clear lists for oldest user
            if oldest_user in self.blacklist:
                del self.blacklist[oldest_user]
            if oldest_user in self.whitelist:
                del self.whitelist[oldest_user]
        
        # Ensure state is on CPU for storage
        if state.is_cuda:
            state = state.cpu()
        
        self.memory[user_id].append(Transition(state, action, reward))
        self.current_size += 1

        # Update blacklist and whitelist if true items are provided
        if true_items is not None:
            if action not in true_items:
                self.blacklist[user_id].add(action)
                # Limit blacklist size
                if len(self.blacklist[user_id]) > self.max_list_size:
                    self.blacklist[user_id] = set(list(self.blacklist[user_id])[-self.max_list_size:])
            else:
                self.whitelist[user_id].add(action)
                # Limit whitelist size
                if len(self.whitelist[user_id]) > self.max_list_size:
                    self.whitelist[user_id] = set(list(self.whitelist[user_id])[-self.max_list_size:])

    def sample_batch(self, user_id):
        if user_id not in self.memory or len(self.memory[user_id]) == 0:
            return None
        
        sample_size = min(self.batch_size, len(self.memory[user_id]))
        batch = random.sample(list(self.memory[user_id]), sample_size)
        
        # Move tensors to the correct device
        states = torch.stack([transition.state for transition in batch]).to(self.device)
        actions = torch.tensor([transition.action for transition in batch], 
                             dtype=torch.long, device=self.device)
        rewards = torch.tensor([transition.reward for transition in batch], 
                             dtype=torch.float32, device=self.device)
        
        return states, actions.unsqueeze(1), rewards.unsqueeze(1)

    def clear_old_experiences(self, user_id):
        if user_id in self.memory:
            self.current_size -= len(self.memory[user_id])
            del self.memory[user_id]
            # Also clear lists
            if user_id in self.blacklist:
                del self.blacklist[user_id]
            if user_id in self.whitelist:
                del self.whitelist[user_id]

    def get_high_reward_actions(self, user_id, threshold=10):
        if user_id not in self.memory:
            return []
        
        high_reward_actions = []
        for transition in self.memory[user_id]:
            if transition.reward > threshold:
                high_reward_actions.append(transition.action)
        
        return list(set(high_reward_actions))

    def get_whitelist_actions(self, user_id):
        """Get actions from the whitelist for a user"""
        return list(self.whitelist.get(user_id, set()))

    def get_blacklist_actions(self, user_id):
        """Get actions from the blacklist for a user"""
        return list(self.blacklist.get(user_id, set()))

def calculate_diversity_reward(predictions, true_items, previous_actions=None, blacklist=None, whitelist=None, purchase_history=None):
    """
    Calculate a reward that strongly penalizes incorrect predictions while maintaining diversity.
    """
    if not true_items or not predictions:
        return 0.0, 0.0, 0.0
    
    # Calculate F1 score for accuracy
    true_set = set(true_items)
    pred_set = set(predictions)
    true_positives = len(true_set.intersection(pred_set))
    false_positives = len(pred_set - true_set)
    
    if true_positives == 0:
        precision = 0.0
        recall = 0.0
    else:
        precision = true_positives / len(predictions)
        recall = true_positives / len(true_set)
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate diversity score
    unique_items = len(set(predictions))
    diversity_score = unique_items / len(predictions)
    
    # Calculate user-specific scores with increased weight for previous baskets
    user_specific_score = 0.0
    if purchase_history:
        historical_overlap = len(set(predictions).intersection(purchase_history))
        user_specific_score = historical_overlap / len(predictions)
    
    # Calculate blacklist and whitelist scores
    blacklist_penalty = 0.0
    whitelist_bonus = 0.0
    
    if blacklist:
        blacklisted_items = len(set(predictions).intersection(blacklist))
        blacklist_penalty = blacklisted_items / len(predictions)
    
    if whitelist:
        whitelisted_items = len(set(predictions).intersection(whitelist))
        whitelist_bonus = whitelisted_items / len(predictions)
    
    # Strongly weight accuracy and correct predictions
    accuracy_weight = 0.7
    diversity_weight = 0.15
    user_specific_weight = 0.15
    blacklist_weight = 0.3
    whitelist_weight = 0.3
    
    # Calculate base reward
    base_reward = (
        f1_score * accuracy_weight +
        diversity_score * diversity_weight +
        user_specific_score * user_specific_weight -
        blacklist_penalty * blacklist_weight +
        whitelist_bonus * whitelist_weight
    )
    
    # Apply non-linear scaling to make rewards more meaningful
    if base_reward > 0:
        total_reward = math.sqrt(base_reward) * 2
    else:
        total_reward = base_reward * 3
    
    # Add a significant bonus for correct predictions
    correct_predictions = len(set(predictions).intersection(true_items))
    if correct_predictions > 0:
        total_reward += 0.5 * correct_predictions
    else:
        total_reward -= 1.0
    
    # Add penalty for false positives
    total_reward -= 0.2 * false_positives
    
    # Cap the maximum reward and minimum penalty
    total_reward = min(max(total_reward, -2.0), 1.0)
    
    return f1_score, diversity_score, total_reward

def sample_predictions(gnn, agent, data, test_data_dict, reverse_user_mapping, reverse_product_mapping, product_nodes, device, top_k=10):
    """
    Sample predictions for 5 random users and display their true purchases and predictions.
    Both true purchases and predictions are sorted in ascending order for easier comparison.
    """
    # Get 5 random users from the test data
    sample_users = random.sample(list(test_data_dict.keys()), min(5, len(test_data_dict)))
    
    print("\nSample Predictions:")
    print("-" * 80)
    
    for user_id in sample_users:
        # Get true purchases and sort them
        true_purchases = sorted(test_data_dict[user_id])
        
        # Get user node index
        user_node = next((k for k, v in reverse_user_mapping.items() if v == user_id), None)
        if user_node is None:
            continue
            
        # Get predictions
        with torch.no_grad():
            state = gnn(data.x, data.edge_index)[user_node].to(device)
            predicted_indices = agent.get_top_actions(state, top_k=top_k)
            predicted_products = [reverse_product_mapping.get(int(product_nodes[idx]), -1) for idx in predicted_indices]
            predicted_products = [p for p in predicted_products if p != -1]
            # Sort predictions
            predicted_products = sorted(predicted_products)
        
        # Print results
        print(f"User ID: {user_id}")
        print(f"True Purchases: {true_purchases}")
        print(f"Predictions: {predicted_products}")
        print(f"Correct Predictions: {sorted([p for p in predicted_products if p in true_purchases])}")
        print("-" * 80)

def train_rl_agent(data, device, user_nodes, product_nodes, reverse_user_mapping, reverse_product_mapping,
                   episodes=10000, learning_rate=0.001, top_k=10, batch_size=128, gamma=0.99,
                   experiences_per_episode=32, ground_truth_file='test_data.csv'):
    """
    Train a reinforcement learning agent using a graph neural network.
    
    Args:
        data: PyG Data object containing the graph
        device: torch device to use (cuda or cpu)
        user_nodes: List of user node indices
        product_nodes: List of product node indices
        reverse_user_mapping: Dictionary mapping node indices to user IDs
        reverse_product_mapping: Dictionary mapping node indices to product IDs
        episodes: Number of training episodes
        learning_rate: Learning rate for the optimizer
        top_k: Number of top actions to consider
        batch_size: Batch size for training
        gamma: Discount factor for future rewards
        experiences_per_episode: Number of experiences to collect per episode
        ground_truth_file: Path to the CSV file containing ground truth data
    """
    # Get input feature dimension from data
    input_dim = data.x.size(1)
    hidden_dim = 32
    output_dim = 32
    
    # Set memory management settings
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Reduce memory usage
    memory = UserSpecificMemory(capacity=25000, batch_size=batch_size, device=device)
    gnn = GNN(num_features=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    agent = RLAgent(state_dim=output_dim, action_dim=len(product_nodes), epsilon=3.5, epsilon_min=0.1, decay_rate=0.995).to(device)
    optimizer = optim.Adam(list(gnn.parameters()) + list(agent.parameters()), lr=learning_rate)
    episodeList, rewardList = [], []
    totalReward = 0
    avg_rewards = []  # Track average rewards for plotting

    # Track previous actions for each user
    user_previous_actions = defaultdict(set)

    last_cleanup = 0
    cleanup_interval = 50

    send_email("Training started", "Training started at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Get only the user nodes that exist in our reverse mapping
    valid_user_nodes = [node for node in user_nodes if int(node) in reverse_user_mapping]
    print(f"Total user nodes: {len(user_nodes)}")
    print(f"Valid user nodes: {len(valid_user_nodes)}")
    print(f"First few user nodes: {valid_user_nodes[:5]}")
    print(f"First few reverse mappings: {list(reverse_user_mapping.items())[:5]}")
    print(f"Number of reverse mappings: {len(reverse_user_mapping)}")

    if not valid_user_nodes:
        raise ValueError("No valid user nodes found!")

    # Load test data once at the start
    print("Loading test data...")
    test_data = pd.read_csv(ground_truth_file)
    test_data_dict = {}  # Cache for user test data
    purchase_counts = []
    for _, row in test_data.iterrows():
        user_id = row['user_id']
        purchases = [int(p) for p in row.iloc[2:] if p > 0]
        if purchases:  # Only store if there are actual purchases
            test_data_dict[user_id] = purchases
            purchase_counts.append(len(purchases))
    
    avg_purchases = sum(purchase_counts) / len(purchase_counts) if purchase_counts else 0
    max_purchases = max(purchase_counts) if purchase_counts else 0
    min_purchases = min(purchase_counts) if purchase_counts else 0
    
    print(f"Loaded test data for {len(test_data_dict)} users")
    print(f"Purchase statistics:")
    print(f"  Average purchases per user: {avg_purchases:.2f}")
    print(f"  Maximum purchases per user: {max_purchases}")
    print(f"  Minimum purchases per user: {min_purchases}")
    
    # Debug: Check product mapping
    product_in_test = set()
    for purchases in test_data_dict.values():
        product_in_test.update(purchases)
    
    mapped_products_counter = 0
    for prod_id in product_in_test:
        for mapped_id in product_mapping.values():
            if mapped_id == prod_id:
                mapped_products_counter += 1
                break
    
    print(f"Products in test data: {len(product_in_test)}")
    print(f"Products found in mapping: {mapped_products_counter}")
    print(f"First few test products: {list(product_in_test)[:10]}")
    print(f"First few product mappings: {list(product_mapping.items())[:10]}")
    
    # Analyze the reverse mapping
    rev_mapped_products = set(reverse_product_mapping.values())
    overlap = product_in_test.intersection(rev_mapped_products)
    print(f"Products in reverse mapping: {len(rev_mapped_products)}")
    print(f"Products that overlap with test data: {len(overlap)} ({len(overlap)/len(product_in_test)*100:.2f}%)")
    
    # Debug - show first few items of each data structure
    print("\nData structure samples:")
    print(f"Product nodes: {product_nodes[:5]}")
    print(f"Reverse product mapping sample: {list(reverse_product_mapping.items())[:5]}")
    
    # Load historical data
    print("Loading historical data...")
    historical_data = pd.read_csv(ground_truth_file)
    historical_dict = {}  # Cache for user historical data
    for _, row in historical_data.iterrows():
        user_id = row['user_id']
        purchases = [int(p) for p in row.iloc[2:] if p > 0]
        if purchases:  # Only store if there are actual purchases
            historical_dict[user_id] = purchases
    
    # Pre-compute GNN embeddings in chunks
    print("Pre-computing GNN embeddings...")
    with torch.no_grad():
        gnn_embeddings = []
        batch_size = 10000
        for i in range(0, data.x.size(0), batch_size):
            end = min(i + batch_size, data.x.size(0))
            x_batch = data.x[i:end].to(device)
            edge_index_batch = data.edge_index.to(device)
            embeddings = gnn(x_batch, edge_index_batch)
            gnn_embeddings.append(embeddings.cpu())
            del x_batch, embeddings
            torch.cuda.empty_cache()
        gnn_embeddings = torch.cat(gnn_embeddings, dim=0).to(device)
    
    # Training loop
    for episode in range(episodes):
        episode_reward = 0
        episode_correct = 0
        episode_total = 0
        episode_f1_scores = []
        episode_diversity_scores = []
        
        if episode % 50 == 0:
            users_to_clear = random.sample(list(memory.memory.keys()), min(25, len(memory.memory)))
            for user in users_to_clear:
                memory.clear_old_experiences(user)
            gc.collect()
            torch.cuda.empty_cache()
        
        # Process multiple experiences per episode
        for exp_idx in range(experiences_per_episode):
            mapped_user_node = int(random.choice(valid_user_nodes))
            original_user_node = reverse_user_mapping[mapped_user_node]
            state = gnn_embeddings[mapped_user_node].detach()
            
            # Get test data from cache
            true_purchases = test_data_dict.get(original_user_node, None)
            if not true_purchases:
                continue
            
            # Get user's purchase history
            user_history = user_previous_actions.get(original_user_node, set())
            
            # Get blacklist and whitelist for this user
            blacklist = set(memory.get_blacklist_actions(original_user_node))
            whitelist = set(memory.get_whitelist_actions(original_user_node))
            
            # Get actions from agent
            state_for_agent = state.to(device)
            with torch.no_grad():
                new_actions = agent.get_top_actions(state_for_agent, top_k=top_k, 
                                                  user_history=user_history,
                                                  blacklist=blacklist,
                                                  whitelist=whitelist)
            
            # Convert actions to product IDs
            mapped_products = [product_nodes[idx] for idx in new_actions]
            original_products = [reverse_product_mapping.get(int(mp), -1) for mp in mapped_products]
            original_products = [p for p in original_products if p != -1]
            
            # Calculate rewards
            f1_score, diversity_score, reward = calculate_diversity_reward(
                original_products, true_purchases, user_history, blacklist, whitelist, user_history
            )
            
            # Accumulate rewards and metrics
            episode_reward += reward
            episode_f1_scores.append(f1_score)
            episode_diversity_scores.append(diversity_score)
            
            # Update previous actions
            user_previous_actions[original_user_node].update(original_products)
            if len(user_previous_actions[original_user_node]) > 50:
                user_previous_actions[original_user_node] = set(list(user_previous_actions[original_user_node])[-50:])
            
            # Store experience
            memory.append(original_user_node, state, new_actions[0], reward, true_purchases)
            
            # Train on a batch of experiences
            batch = memory.sample_batch(original_user_node)
            if batch:
                batch_states, batch_actions, batch_rewards = batch
                optimizer.zero_grad()
                
                current_q_values = agent(batch_states)
                
                with torch.no_grad():
                    next_q_values = agent(batch_states)
                    max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
                    target_q_values = batch_rewards + gamma * max_next_q_values
                
                loss = F.smooth_l1_loss(current_q_values, target_q_values)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(gnn.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Clear memory
                del batch_states, batch_actions, batch_rewards, current_q_values, next_q_values, max_next_q_values, target_q_values, loss
                torch.cuda.empty_cache()
        
        # Update GNN embeddings less frequently and in chunks
        if episode % 50 == 0:
            with torch.no_grad():
                gnn_embeddings = []
                for i in range(0, data.x.size(0), batch_size):
                    end = min(i + batch_size, data.x.size(0))
                    x_batch = data.x[i:end].to(device)
                    edge_index_batch = data.edge_index.to(device)
                    embeddings = gnn(x_batch, edge_index_batch)
                    gnn_embeddings.append(embeddings.cpu())
                    del x_batch, embeddings
                    torch.cuda.empty_cache()
                gnn_embeddings = torch.cat(gnn_embeddings, dim=0).to(device)
        
        agent.decay_epsilon()
        
        # Update total reward
        totalReward += episode_reward
        
        # Print progress less frequently
        if episode % 50 == 0:
            accuracy = episode_correct / episode_total if episode_total > 0 else 0
            print(f"\nEpisode {episode}:")
            print(f"  Average F1 Score: {sum(episode_f1_scores)/len(episode_f1_scores):.4f}")
            print(f"  Average Diversity Score: {sum(episode_diversity_scores)/len(episode_diversity_scores):.4f}")
            print(f"  Average Reward per Experience: {episode_reward/experiences_per_episode:.2f}")
            print(f"  Total Episode Reward: {episode_reward:.2f}")
            print(f"  Cumulative Total Reward: {totalReward:.2f}")
            print(f"  Accuracy: {accuracy:.2%}")
            print(f"  Current epsilon: {agent.epsilon:.4f}")
            
            # Debug on occasional episodes
            if episode % 1000 == 0:
                # Count correct items
                test_user = valid_user_nodes[0]
                original_test_user = reverse_user_mapping[test_user]
                true_items = test_data_dict.get(original_test_user, [])
                
                if true_items:
                    with torch.no_grad():
                        test_state = gnn_embeddings[test_user].to(device)
                        actions = agent.get_top_actions(test_state, top_k=top_k)
                        mapped_prods = [product_nodes[idx] for idx in actions]
                        original_prods = [reverse_product_mapping.get(int(p), -1) for p in mapped_prods]
                        original_prods = [p for p in original_prods if p != -1]
                        
                        correct = [p for p in original_prods if p in true_items]
                        f1_score, diversity_score, reward = calculate_diversity_reward(
                            original_prods, true_items, user_previous_actions.get(original_test_user, set()), memory.get_blacklist_actions(original_test_user), memory.get_whitelist_actions(original_test_user), user_previous_actions.get(original_test_user, set())
                        )
                        print(f"\n  Debug test user {original_test_user}:")
                        print(f"    True items: {true_items[:5]}...")
                        print(f"    Predicted: {original_prods[:5]}...")
                        print(f"    Correct: {correct}")
                        print(f"    Test F1: {f1_score:.4f}")
                        print(f"    Test Diversity: {diversity_score:.4f}")
                        print(f"    Test Reward: {reward:.4f}")
            
            # Sample predictions every 100 episodes
            sample_predictions(gnn, agent, data, test_data_dict, reverse_user_mapping, 
                             reverse_product_mapping, product_nodes, device, top_k)
            
            episodeList.append(episode)
            rewardList.append(totalReward)  # Store cumulative reward instead of episode reward

    print("Training Complete!")
    send_email("Training Complete!", "Training Complete at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return agent, gnn, episodeList, rewardList

def save_model(gnn, agent, graph_data, model_dir):
    """Save the GNN and agent models along with graph data."""
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save graph data using pickle
    graph_path = os.path.join(model_dir, 'graph.pt')
    with open(graph_path, 'wb') as f:
        pickle.dump(graph_data, f)
    
    # Save model states with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(model_dir, f'model_{timestamp}.pth')
    
    checkpoint = {
        'gnn_state_dict': gnn.state_dict(),
        'agent_state_dict': agent.state_dict(),
        'timestamp': timestamp
    }
    
    torch.save(checkpoint, model_path)
    print(f"Models saved to {model_dir}")
    print(f"Model timestamp: {timestamp}")

def load_model(model_dir, device):
    """Load the latest GNN and agent models from the specified directory."""
    # Check if model directory exists
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} does not exist")
    
    # Define expected file paths
    model_path = os.path.join(model_dir, 'model.pt')
    graph_path = os.path.join(model_dir, 'graph.pt')
    
    # Check if required files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph data file not found at {graph_path}")
    
    # Load the graph data with proper handling for PyTorch 2.6
    try:
        # First try loading with weights_only=False
        graph_data = torch.load(graph_path, weights_only=False, map_location=device)
    except Exception as e:
        print(f"First load attempt failed: {str(e)}")
        try:
            # If that fails, try with safe_globals
            from torch.serialization import safe_globals
            import numpy as np
            with safe_globals(['numpy._core.multiarray.scalar', 'numpy.dtype', 'numpy.ndarray']):
                graph_data = torch.load(graph_path, map_location=device)
        except Exception as e:
            print(f"Second load attempt failed: {str(e)}")
            # If both fail, try one last time with minimal settings
            graph_data = torch.load(graph_path, weights_only=False, map_location=device)
    
    # Create GNN and agent instances with correct dimensions
    gnn = GNN(num_features=16, hidden_dim=32, output_dim=32).to(device)  # Changed output_dim to 32
    agent = RLAgent(state_dim=32, action_dim=len(graph_data['product_nodes']),  # Changed state_dim to 32
                   epsilon_min=0.05, epsilon=2).to(device)
    
    # Load model states
    try:
        checkpoint = torch.load(model_path, map_location=device)
        gnn.load_state_dict(checkpoint['gnn_state_dict'])
        agent.load_state_dict(checkpoint['agent_state_dict'])
    except Exception as e:
        raise RuntimeError(f"Failed to load model states from {model_path}: {str(e)}")
    
    print(f"Models loaded successfully from {model_dir}")
    
    return gnn, agent, graph_data

if __name__ == "__main__":
    # Load the preprocessed graph data
    print("Loading graph data...")
    try:
        # First try loading with weights_only=False
        graph_data = torch.load('graph_data.pt', weights_only=False, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"First load attempt failed: {str(e)}")
        try:
            # If that fails, try with safe_globals
            from torch.serialization import safe_globals
            import numpy as np
            with safe_globals(['numpy._core.multiarray.scalar', 'numpy.dtype', 'numpy.ndarray']):
                graph_data = torch.load('graph_data.pt', map_location=torch.device('cpu'))
        except Exception as e:
            print(f"Second load attempt failed: {str(e)}")
            # If both fail, try one last time with minimal settings
            graph_data = torch.load('graph_data.pt', weights_only=False, map_location=torch.device('cpu'))
    
    print("Graph data loaded successfully!")
    
    # Extract components
    edge_index = graph_data['edge_index']
    x = graph_data['x']
    user_nodes = graph_data['user_nodes']
    product_nodes = graph_data['product_nodes']
    user_mapping = graph_data['user_mapping']
    product_mapping = graph_data['product_mapping']
    reverse_user_mapping = graph_data['reverse_user_mapping']
    reverse_product_mapping = graph_data['reverse_product_mapping']
    
    # Convert numpy values to Python integers if needed
    if isinstance(user_nodes[0], (np.integer, np.floating)):
        user_nodes = [int(node) for node in user_nodes]
    if isinstance(product_nodes[0], (np.integer, np.floating)):
        product_nodes = [int(node) for node in product_nodes]
    
    # Convert mappings if needed
    reverse_user_mapping = {int(k): int(v) for k, v in reverse_user_mapping.items()}
    reverse_product_mapping = {int(k): int(v) for k, v in reverse_product_mapping.items()}
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Move data to device
    data = data.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    
    # Define paths
    ground_truth_file = "/mnt/megastore/UNI/y3/fyp stuff/completedVers/full_test_baskets.csv"
    
    # Train the model
    print("Starting training...")
    agent, gnn, episodeList, rewardList = train_rl_agent(
        data=data,
        device=device,
        user_nodes=user_nodes,
        product_nodes=product_nodes,
        reverse_user_mapping=reverse_user_mapping,
        reverse_product_mapping=reverse_product_mapping,
        episodes=1001,  # Reduced number of episodes since we process more experiences per episode
        experiences_per_episode=64,  # Process 64 experiences per episode
        ground_truth_file=ground_truth_file
    )
    
    # Save the trained models and graph data
    model_dir = save_model(gnn, agent, graph_data, model_dir)
    
    # Plot training progress
    plt.figure(figsize=(12, 8))
    plt.plot(episodeList, rewardList, 'b-', label='Total Reward')
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.legend()
    
    # Save the plot first
    plot_path = os.path.join(model_dir, 'training_progress.png')
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")
    
    # Try to show the plot if in an interactive environment
    # try:
    #     plt.show(block=True)  # block=True keeps the plot window open
    # except Exception as e:
    #     print(f"Could not display plot interactively: {e}")
    #     print("Plot has been saved to file instead.")
    
    plt.close() 

    