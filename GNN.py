import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, asdict
import gc
import json
import os
from typing import Dict, Tuple, List, Any

@dataclass
class HyperParameters:
    # Model hyperparameters
    learning_rate: float = 0.006610595090609882
    hidden_dim: int = 39
    output_dim: int = 60
    batch_size: int = 34
    
    # Strength calculation weights
    strength_freq_weight: float = 0.6
    strength_user_affinity_weight: float = 0.2
    strength_product_affinity_weight: float = 0.2
    
    # Confidence calculation weights
    confidence_freq_weight: float = 0.4
    confidence_consistency_weight: float = 0.3
    confidence_recency_weight: float = 0.2
    confidence_strength_weight: float = 0.1
    
    # Purchase consistency parameters
    consistency_monthly_threshold: int = 30  # days
    consistency_increase_factor: float = 1.1
    consistency_decrease_factor: float = 0.9
    consistency_default: float = 0.5
    
    # Frequency normalization
    frequency_cap: int = 5  # purchases
    
    # Time decay parameters
    recency_decay_period: int = 365  # days
    
    # Combined score weights
    combined_strength_weight: float = 0.5
    combined_confidence_weight: float = 0.5
    
    # Recommendation parameters
    num_recommendations: int = 10  # Number of items to recommend
    
    def __post_init__(self):
        # Validate weights sum to 1.0
        strength_weights = [self.strength_freq_weight, self.strength_user_affinity_weight, 
                          self.strength_product_affinity_weight]
        confidence_weights = [self.confidence_freq_weight, self.confidence_consistency_weight,
                            self.confidence_recency_weight, self.confidence_strength_weight]
        combined_weights = [self.combined_strength_weight, self.combined_confidence_weight]
        
        assert abs(sum(strength_weights) - 1.0) < 1e-6, "Strength weights must sum to 1.0"
        assert abs(sum(confidence_weights) - 1.0) < 1e-6, "Confidence weights must sum to 1.0"
        assert abs(sum(combined_weights) - 1.0) < 1e-6, "Combined weights must sum to 1.0"
        assert self.num_recommendations > 0, "Number of recommendations must be positive"
        assert self.hidden_dim > 0, "Hidden dimension must be positive"
        assert self.output_dim > 0, "Output dimension must be positive"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"

def clean_memory():
    """Helper function to force garbage collection and clean PyTorch memory"""
    try:
        # Force garbage collection first
        gc.collect()
        
        # Clear PyTorch's memory caches
        if torch.cuda.is_available():
            try:
                # Synchronize all CUDA devices first
                with torch.cuda.device('cuda'):
                    torch.cuda.synchronize()
                    
                    # Empty cache for each device
                    torch.cuda.empty_cache()
                    
                    # Reset peak memory stats if available
                    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                        torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass  # Silently ignore CUDA cleanup errors
    except Exception:
        pass  # Silently ignore memory cleanup errors

def clean_tensors(*tensors):
    """Helper function to clean up PyTorch tensors"""
    try:
        for t in tensors:
            if isinstance(t, torch.Tensor):
                try:
                    # If tensor is on CUDA, move to CPU first
                    if t.is_cuda:
                        with torch.cuda.device(t.device):
                            t_cpu = t.cpu()
                            del t  # Delete CUDA tensor
                            del t_cpu  # Delete CPU tensor
                    else:
                        del t
                except Exception:
                    pass  # Silently ignore tensor cleanup errors
    except Exception:
        pass  # Silently ignore tensor cleanup errors
    finally:
        # Always try to clean memory
        clean_memory()

def calculate_relationship_strength(user_product_freq, user_total_items, product_total_purchases, df, params=None):
    """
    Calculate relationship strength and confidence scores between users and products.
    Returns relationship strength, confidence scores, and regularity scores
    """
    if params is None:
        params = HyperParameters()
        
    strengths = {}
    confidence_scores = {}
    regularity_scores = defaultdict(float)
    
    # Track purchase dates for each user-product pair
    purchase_dates = defaultdict(list)
    user_basket_counts = defaultdict(int)
    
    # First pass: collect all purchase dates
    for _, row in df.iterrows():
        user_id = row['user_id']
        day = row['days_since_prior_order']
        user_basket_counts[user_id] += 1
        products = row[[col for col in df.columns if col.startswith('item')]][row != 0].values
        
        for product_id in products:
            purchase_dates[(user_id, product_id)].append(day)
    
    clean_memory()  # Clean up after first pass
    
    # Calculate regularity scores
    for (user_id, product_id), dates in purchase_dates.items():
        freq = len(dates)
        if freq > 1:
            dates.sort()
            time_between_purchases = sum(abs(dates[i] - dates[i-1]) for i in range(1, len(dates)))
            baskets = user_basket_counts[user_id]
            regularity_score = (freq * time_between_purchases) / (baskets * baskets)
            regularity_scores[(user_id, product_id)] = regularity_score
    
    clean_memory()  # Clean up after regularity calculation
    
    max_freq = max(user_product_freq.values())
    user_last_purchase = defaultdict(dict)
    user_purchase_consistency = defaultdict(dict)
    
    # Process orders chronologically for time-based features
    df_sorted = df.sort_values('days_since_prior_order')
    for _, row in df_sorted.iterrows():
        user_id = row['user_id']
        day = row['days_since_prior_order']
        products = row[[col for col in df.columns if col.startswith('item')]][row != 0].values
        
        for product_id in products:
            if product_id not in user_last_purchase[user_id]:
                user_last_purchase[user_id][product_id] = day
                user_purchase_consistency[user_id][product_id] = 1.0
            else:
                days_diff = abs(day - user_last_purchase[user_id][product_id])
                if days_diff <= params.consistency_monthly_threshold:
                    user_purchase_consistency[user_id][product_id] *= params.consistency_increase_factor
                else:
                    user_purchase_consistency[user_id][product_id] *= params.consistency_decrease_factor
                user_purchase_consistency[user_id][product_id] = min(1.0, user_purchase_consistency[user_id][product_id])
                user_last_purchase[user_id][product_id] = day
    
    del df_sorted  # Remove sorted dataframe from memory
    clean_memory()  # Clean up after consistency calculation
    
    # Calculate final scores
    for (user_id, product_id), freq in user_product_freq.items():
        # Strength calculation
        norm_freq = freq / max_freq
        user_affinity = freq / user_total_items[user_id]
        product_affinity = freq / product_total_purchases[product_id]
        
        strength = (params.strength_freq_weight * norm_freq + 
                   params.strength_user_affinity_weight * user_affinity +
                   params.strength_product_affinity_weight * product_affinity)
        strengths[(user_id, product_id)] = strength
        
        # Confidence calculation
        purchase_consistency = user_purchase_consistency[user_id].get(product_id, params.consistency_default)
        frequency_confidence = min(1.0, freq / params.frequency_cap)
        recency_factor = 1.0
        
        if user_id in user_last_purchase and product_id in user_last_purchase[user_id]:
            days_since_last = user_last_purchase[user_id][product_id]
            recency_factor = np.exp(-days_since_last / params.recency_decay_period) if days_since_last > 0 else 1.0
        
        confidence = (params.confidence_freq_weight * frequency_confidence +
                     params.confidence_consistency_weight * purchase_consistency +
                     params.confidence_recency_weight * recency_factor +
                     params.confidence_strength_weight * strength)
        confidence_scores[(user_id, product_id)] = confidence
    
    # Clean up large temporary data structures
    del user_last_purchase, user_purchase_consistency, purchase_dates
    clean_memory()
    
    return strengths, confidence_scores, regularity_scores

def load_data(file_path='trueVers/19k_train_baskets.csv', params=None):
    # Read the CSV file
    print("Loading data from CSV...")
    df = pd.read_csv(file_path)
    clean_memory()
    
    print("Processing item columns...")
    item_cols = [col for col in df.columns if col.startswith('item')]
    
    # Create edges between users and products
    edges = []
    unique_users = set()
    unique_products = set()
    
    # Track purchase frequencies using memory-efficient defaultdict
    user_product_freq = defaultdict(int)
    user_total_items = defaultdict(int)
    product_total_purchases = defaultdict(int)
    
    print("Processing baskets...")
    # Process in chunks to reduce memory usage
    chunk_size = 1000
    last_reported_user = None
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        for _, row in chunk.iterrows():
            user_id = row['user_id']
            unique_users.add(user_id)
            
            # Print progress every 2000 users
            if last_reported_user is None or len(unique_users) >= last_reported_user + 2000:
                print(f"Processing user {user_id} (Total users so far: {len(unique_users)})")
                last_reported_user = len(unique_users)
            
            products = row[item_cols][row[item_cols] != 0].values
            unique_products.update(products)
            
            for product_id in products:
                edges.append([user_id, product_id])
                user_product_freq[(user_id, product_id)] += 1
                user_total_items[user_id] += 1
                product_total_purchases[product_id] += 1
        
        if i % (chunk_size * 10) == 0:  # Clean memory every 10 chunks
            clean_memory()
    
    print(f"Finished processing all {len(unique_users)} users")
    
    print("Calculating relationship scores...")
    relationship_strengths, confidence_scores, regularity_scores = calculate_relationship_strength(
        user_product_freq, user_total_items, product_total_purchases, df, params)
    
    # Free up the DataFrame as it's no longer needed
    del df
    clean_memory()
    
    print("Creating mapping dictionaries...")
    user_to_idx = {user: idx for idx, user in enumerate(sorted(unique_users))}
    product_to_idx = {product: idx + len(user_to_idx) for idx, product in enumerate(sorted(unique_products))}
    
    print("Converting edges to tensors...")
    # Process edges in chunks to reduce peak memory usage
    chunk_size = 10000
    edge_chunks = []
    weight_chunks = []
    
    for i in range(0, len(edges), chunk_size):
        chunk = edges[i:i+chunk_size]
        mapped_chunk = [[user_to_idx[user_id], product_to_idx[product_id]] for user_id, product_id in chunk]
        weights_chunk = [relationship_strengths[(user_id, product_id)] for user_id, product_id in chunk]
        
        edge_chunks.append(torch.tensor(mapped_chunk, dtype=torch.long))
        weight_chunks.append(torch.tensor(weights_chunk, dtype=torch.float))
    
    # Concatenate chunks
    edge_index = torch.cat(edge_chunks, dim=0).t()
    edge_weight = torch.cat(weight_chunks, dim=0)
    
    # Clean up chunks
    del edge_chunks, weight_chunks, edges
    clean_memory()
    
    print("Creating node features...")
    num_nodes = len(user_to_idx) + len(product_to_idx)
    x = torch.zeros((num_nodes, 2), dtype=torch.float)
    x[:len(user_to_idx), 0] = 1
    x[len(user_to_idx):, 1] = 1
    
    node_type = torch.zeros(num_nodes, dtype=torch.long)
    node_type[len(user_to_idx):] = 1
    
    print("Creating PyTorch Geometric Data object...")
    data = MemoryOptimizedData(x=x, edge_index=edge_index, edge_weight=edge_weight, node_type=node_type)
    
    # Clean up intermediate tensors
    clean_tensors(x, edge_index, edge_weight, node_type)
    
    return (data, list(unique_users), list(unique_products), user_to_idx, product_to_idx, 
            user_product_freq, user_total_items, product_total_purchases, 
            relationship_strengths, confidence_scores, regularity_scores)

class MemoryOptimizedData(Data):
    """Extended PyTorch Geometric Data class with memory optimization"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def to_device(self, device):
        """Move data to specified device with memory optimization"""
        for key, item in self:
            if torch.is_tensor(item):
                self[key] = item.to(device)
                # Clear CPU memory if moved to GPU
                if device.type == 'cuda':
                    del item
                    clean_memory()
        return self
    
    def cleanup(self):
        """Clean up tensor memory"""
        for key, item in self:
            if torch.is_tensor(item):
                del item
        clean_memory()

class BipartiteGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, device='cpu'):
        super(BipartiteGNN, self).__init__()
        self.device = device
        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        
        # Create separate convolutions for users and products
        self.user_conv1 = SAGEConv((num_features, num_features), hidden_channels)
        self.user_conv2 = SAGEConv((hidden_channels, hidden_channels), num_classes)
        
        self.product_conv1 = SAGEConv((num_features, num_features), hidden_channels)
        self.product_conv2 = SAGEConv((hidden_channels, hidden_channels), num_classes)
        
        # Move all layers to device
        self.to(device)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for saving"""
        return {
            'num_features': self.num_features,
            'hidden_channels': self.hidden_channels,
            'num_classes': self.num_classes
        }
        
    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        # Process user nodes
        x_user = x_dict['user']
        x_product = x_dict['product']
        
        # User -> Product
        edge_index = edge_index_dict[('user', 'buys', 'product')]
        
        # Ensure edge indices are properly typed and within bounds
        edge_index = edge_index.long()
        
        # Get the number of users and products
        num_users = x_user.size(0)
        num_products = x_product.size(0)
        
        # Validate edge indices
        if edge_index.size(1) > 0:  # Only validate if we have edges
            # Check user indices (first row of edge_index)
            user_indices = edge_index[0]
            if torch.any(user_indices >= num_users) or torch.any(user_indices < 0):
                raise ValueError(f"User indices out of bounds. Max index: {user_indices.max()}, "
                               f"Number of users: {num_users}")
            
            # Check product indices (second row of edge_index)
            product_indices = edge_index[1]
            if torch.any(product_indices >= num_products) or torch.any(product_indices < 0):
                raise ValueError(f"Product indices out of bounds. Max index: {product_indices.max()}, "
                               f"Number of products: {num_products}")
        
        try:
            # First layer - user features
            # For user nodes, we aggregate from products
            x_user_1 = self.user_conv1(
                (x_user, x_product),  # (source, target)
                edge_index
            )
            x_user_1 = F.relu(x_user_1)
            x_user_1 = F.dropout(x_user_1, p=0.5, training=self.training)
            
            # First layer - product features
            # For product nodes, we aggregate from users
            x_product_1 = self.product_conv1(
                (x_product, x_user),  # (source, target)
                edge_index.flip([0])  # Flip edge direction
            )
            x_product_1 = F.relu(x_product_1)
            x_product_1 = F.dropout(x_product_1, p=0.5, training=self.training)
            
            # Second layer
            x_user_2 = self.user_conv2(
                (x_user_1, x_product_1),
                edge_index
            )
            
            x_product_2 = self.product_conv2(
                (x_product_1, x_user_1),
                edge_index.flip([0])
            )
            
            # Ensure all computations are complete before returning
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            return {'user': x_user_2, 'product': x_product_2}
            
        except Exception as e:
            print(f"Error during forward pass: {str(e)}")
            raise e
    
    def cleanup(self):
        """Clean up model's memory"""
        try:
            # First move all parameters to CPU
            if next(self.parameters(), None) is not None:
                try:
                    with torch.cuda.device(self.device):
                        self.cpu()
                except Exception:
                    pass  # Silently ignore CPU transfer errors
            
            # Clear gradients with set_to_none for better memory cleanup
            try:
                self.zero_grad(set_to_none=True)
            except Exception:
                pass  # Silently ignore gradient cleanup errors
            
            # Delete parameters and buffers
            try:
                for name, param in list(self.named_parameters()):
                    if hasattr(param, 'grad') and param.grad is not None:
                        param.grad = None
                    try:
                        delattr(self, name)
                    except Exception:
                        pass  # Silently ignore parameter deletion errors
                
                for name, buffer in list(self.named_buffers()):
                    try:
                        delattr(self, name)
                    except Exception:
                        pass  # Silently ignore buffer deletion errors
            except Exception:
                pass  # Silently ignore parameter/buffer cleanup errors
            
            # Clear the conv layers
            try:
                if hasattr(self, 'user_conv1'): delattr(self, 'user_conv1')
                if hasattr(self, 'user_conv2'): delattr(self, 'user_conv2')
                if hasattr(self, 'product_conv1'): delattr(self, 'product_conv1')
                if hasattr(self, 'product_conv2'): delattr(self, 'product_conv2')
            except Exception:
                pass  # Silently ignore conv layer cleanup errors
            
        except Exception:
            pass  # Silently ignore model cleanup errors
        finally:
            # Always try to clean memory at the end
            clean_memory()

def get_top_recommendations(user_id, user_relationships, params, sort_key_func, label):
    """Helper function to get top N recommendations sorted by a specific metric"""
    sorted_products = sorted(
        user_relationships.items(),
        key=sort_key_func,
        reverse=True
    )[:params.num_recommendations]
    
    print(f"  Top {params.num_recommendations} {label}:")
    return sorted_products

def save_model_state(
    save_dir: str,
    model: BipartiteGNN,
    params: HyperParameters,
    user_to_idx: Dict[int, int],
    product_to_idx: Dict[int, int],
    relationship_strengths: Dict[Tuple[int, int], float],
    confidence_scores: Dict[Tuple[int, int], float],
    regularity_scores: Dict[Tuple[int, int], float],
    user_product_freq: Dict[Tuple[int, int], int],
    user_total_items: Dict[int, int],
    product_total_purchases: Dict[int, int]
) -> None:
    """
    Save model state and associated data to disk
    
    Args:
        save_dir: Directory to save the model state and data
        model: The BipartiteGNN model
        params: HyperParameters instance
        user_to_idx: User ID to index mapping
        product_to_idx: Product ID to index mapping
        relationship_strengths: Dictionary of relationship strengths
        confidence_scores: Dictionary of confidence scores
        regularity_scores: Dictionary of regularity scores
        user_product_freq: Dictionary of purchase frequencies
        user_total_items: Dictionary of total items per user
        product_total_purchases: Dictionary of total purchases per product
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model configuration and state
    model_config = model.get_config()
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config
    }, os.path.join(save_dir, 'model.pt'))
    
    # Save hyperparameters
    with open(os.path.join(save_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(asdict(params), f, indent=2)
    
    # Convert tuple keys to strings for JSON serialization
    def convert_dict_keys(d: Dict) -> Dict:
        if isinstance(next(iter(d.keys())), tuple):
            return {f"{k[0]}_{k[1]}": v for k, v in d.items()}
        return d
    
    # Save mappings and scores
    mappings_and_scores = {
        'user_to_idx': user_to_idx,
        'product_to_idx': product_to_idx,
        'relationship_strengths': convert_dict_keys(relationship_strengths),
        'confidence_scores': convert_dict_keys(confidence_scores),
        'regularity_scores': convert_dict_keys(regularity_scores),
        'user_product_freq': convert_dict_keys(user_product_freq),
        'user_total_items': user_total_items,
        'product_total_purchases': product_total_purchases
    }
    
    with open(os.path.join(save_dir, 'mappings_and_scores.json'), 'w') as f:
        json.dump(mappings_and_scores, f, indent=2)
    
    print(f"Model and associated data saved to {save_dir}")

def load_model_state(save_dir: str, device: str = 'cpu') -> Tuple[
    BipartiteGNN,
    HyperParameters,
    Dict[int, int],
    Dict[int, int],
    Dict[Tuple[int, int], float],
    Dict[Tuple[int, int], float],
    Dict[Tuple[int, int], float],
    Dict[Tuple[int, int], int],
    Dict[int, int],
    Dict[int, int]
]:
    """
    Load model state and associated data from disk
    
    Args:
        save_dir: Directory containing the saved model state and data
        device: Device to load the model onto ('cpu' or 'cuda')
        
    Returns:
        Tuple containing:
        - Loaded BipartiteGNN model
        - HyperParameters instance
        - User ID to index mapping
        - Product ID to index mapping
        - Dictionary of relationship strengths
        - Dictionary of confidence scores
        - Dictionary of regularity scores
        - Dictionary of purchase frequencies
        - Dictionary of total items per user
        - Dictionary of total purchases per product
    """
    # Load model state and config
    checkpoint = torch.load(os.path.join(save_dir, 'model.pt'), map_location=device)
    model_config = checkpoint['model_config']
    
    # Create and load model
    model = BipartiteGNN(
        num_features=model_config['num_features'],
        hidden_channels=model_config['hidden_channels'],
        num_classes=model_config['num_classes'],
        device=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Load hyperparameters
    with open(os.path.join(save_dir, 'hyperparameters.json'), 'r') as f:
        params_dict = json.load(f)
        params = HyperParameters(**params_dict)
    
    # Load mappings and scores
    with open(os.path.join(save_dir, 'mappings_and_scores.json'), 'r') as f:
        mappings_and_scores = json.load(f)
    
    # Convert string keys back to tuples for relevant dictionaries
    def convert_keys_back(d: Dict, is_tuple: bool = False) -> Dict:
        if is_tuple:
            return {tuple(map(int, k.split('_'))): v for k, v in d.items()}
        return d
    
    # Extract and convert data
    user_to_idx = {int(k): v for k, v in mappings_and_scores['user_to_idx'].items()}
    product_to_idx = {int(k): v for k, v in mappings_and_scores['product_to_idx'].items()}
    relationship_strengths = convert_keys_back(mappings_and_scores['relationship_strengths'], True)
    confidence_scores = convert_keys_back(mappings_and_scores['confidence_scores'], True)
    regularity_scores = convert_keys_back(mappings_and_scores['regularity_scores'], True)
    user_product_freq = convert_keys_back(mappings_and_scores['user_product_freq'], True)
    user_total_items = {int(k): v for k, v in mappings_and_scores['user_total_items'].items()}
    product_total_purchases = {int(k): v for k, v in mappings_and_scores['product_total_purchases'].items()}
    
    print(f"Model and associated data loaded from {save_dir}")
    
    return (
        model, params, user_to_idx, product_to_idx,
        relationship_strengths, confidence_scores, regularity_scores,
        user_product_freq, user_total_items, product_total_purchases
    )

if __name__ == "__main__":
    # Enable garbage collection
    gc.enable()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    basket_size = 20
    # Create hyperparameters with tuned values
    params = HyperParameters(
        # Model hyperparameters
        learning_rate=0.0001382053466928862,
        hidden_dim=65,
        output_dim=75,
        batch_size=53,
        
        # Strength weights (sum to 1.0)
        strength_freq_weight=0.3079666805612192,
        strength_user_affinity_weight=0.11024544221746542,
        strength_product_affinity_weight=0.5817878772213154,  # Adjusted to sum to 1.0
        
        # Confidence weights (sum to 1.0)
        confidence_freq_weight=0.1504595318915328,
        confidence_consistency_weight=0.3998343895144623,
        confidence_recency_weight=0.3999509643846626,
        confidence_strength_weight=0.0497551142093423,  # Adjusted to sum to 1.0
        
        # Other tuned parameters
        consistency_monthly_threshold=33,
        consistency_increase_factor=1.098487920027529,
        consistency_decrease_factor=0.9212579036779653,
        recency_decay_period=534,
        
        # Combined weights (sum to 1.0)
        combined_strength_weight=0.40321668753176243,
        combined_confidence_weight=0.5967833124682376,  # Adjusted to sum to 1.0
        
        num_recommendations=basket_size
    )
    
    try:
        # Load the data with regularity scores
        print("Starting data loading process...")
        results = load_data(params=params, file_path='trueVers/19k_train_baskets.csv')
        (data, unique_users, unique_products, user_to_idx, product_to_idx, 
         user_product_freq, user_total_items, product_total_purchases,
         relationship_strengths, confidence_scores, regularity_scores) = results
        
        # Move data to device
        data = data.to_device(device)
        clean_memory()
        
        print(f"Loaded bipartite graph:")
        print(f"Number of users: {len(unique_users)}")
        print(f"Number of products: {len(unique_products)}")
        print(f"Total nodes: {data.num_nodes}")
        print(f"Total edges (unique user-product pairs): {data.num_edges}")
        
        # Calculate additional network statistics
        max_user_id = max(unique_users)
        max_product_id = max(unique_products)
        
        # Calculate average products per user
        user_product_counts = defaultdict(int)
        for (user_id, _), freq in user_product_freq.items():
            user_product_counts[user_id] += 1
        avg_products_per_user = sum(user_product_counts.values()) / len(user_product_counts)
        
        # Calculate average baskets per user
        user_basket_counts = defaultdict(int)
        for user_id in unique_users:
            user_basket_counts[user_id] = sum(1 for (u, _), freq in user_product_freq.items() if u == user_id)
        avg_baskets_per_user = sum(user_basket_counts.values()) / len(user_basket_counts)
        
        # Calculate average basket size per user and overall
        user_basket_sizes = defaultdict(list)
        for user_id in unique_users:
            user_items = user_total_items[user_id]
            user_baskets = user_basket_counts[user_id]
            avg_basket_size = user_items / user_baskets
            user_basket_sizes[user_id] = avg_basket_size
        
        overall_avg_basket_size = sum(user_basket_sizes.values()) / len(user_basket_sizes)
        print(f"Average basket size per user: {overall_avg_basket_size:.2f}")
        
        # Calculate average items per basket
        total_items = sum(user_total_items.values())
        total_baskets = sum(user_basket_counts.values())
        avg_items_per_basket = total_items / total_baskets
        print(f"Overall average items per basket: {avg_items_per_basket:.2f}")
        
        print("\nAdditional Network Statistics:")
        print(f"Highest user_id: {max_user_id}")
        print(f"Highest product_id: {max_product_id}")
        print(f"Average unique products per user: {avg_products_per_user:.2f}")
        print(f"Average baskets per user: {avg_baskets_per_user:.2f}")
        
        # Initialize the model with tuned parameters
        model = BipartiteGNN(num_features=2,
                          hidden_channels=params.hidden_dim, 
                          num_classes=params.output_dim,
                          device=device)
        
        # Create optimizer with tuned learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
        
        print("\nModel created successfully!")
        print(f"Model parameters:")
        print(f"  Hidden dimension: {params.hidden_dim}")
        print(f"  Output dimension: {params.output_dim}")
        print(f"  Learning rate: {params.learning_rate}")
        print(f"  Batch size: {params.batch_size}")
        
        # Example of saving the model state
        save_dir = f"model_checkpoints_b{basket_size}_mls"
        save_model_state(
            save_dir=save_dir,
            model=model,
            params=params,
            user_to_idx=user_to_idx,
            product_to_idx=product_to_idx,
            relationship_strengths=relationship_strengths,
            confidence_scores=confidence_scores,
            regularity_scores=regularity_scores,
            user_product_freq=user_product_freq,
            user_total_items=user_total_items,
            product_total_purchases=product_total_purchases
        )
        
        # Example of loading the model state
        # (model, params, user_to_idx, product_to_idx,
        #  relationship_strengths, confidence_scores, regularity_scores,
        #  user_product_freq, user_total_items, product_total_purchases) = load_model_state(
        #     save_dir=save_dir,
        #     device=device
        # )
        
        # Print detailed statistics about the graph
        print(f"\nDetailed Statistics:")
        print(f"Average unique products per user: {data.num_edges / len(unique_users):.2f}")
        print(f"Average unique users per product: {data.num_edges / len(unique_products):.2f}")
        
        # Calculate average total items purchased per user
        total_items = sum(user_total_items.values())
        print(f"Average total items purchased per user: {total_items / len(unique_users):.2f}")
        
        # Print statistics for first few users as example
        print("\nExample User Statistics (first 5 users):")
        for user_id in sorted(unique_users)[:5]:
            unique_products_count = len([freq for (u, p), freq in user_product_freq.items() if u == user_id])
            total_items = user_total_items[user_id]
            print(f"User {user_id}:")
            print(f"  Unique products purchased: {unique_products_count}")
            print(f"  Total items purchased: {total_items}")
            print(f"  Average purchases per product: {total_items / unique_products_count:.2f}")
            
            # Get all products and their metrics for this user
            user_relationships = {p: (strength, confidence_scores[(user_id, p)], 
                                   regularity_scores[(user_id, p)], user_product_freq[(user_id, p)]) 
                                for (u, p), strength in relationship_strengths.items() if u == user_id}
            
            print("\n  [DEBUG] Regular Purchase Recommendations (by time-frequency pattern):")
            regular_products = get_top_recommendations(
                user_id, 
                {p: metrics[2] for p, metrics in user_relationships.items()},  # Just pass regularity scores
                params,
                lambda x: x[1],  # Sort by regularity score
                "regular purchases"
            )
            for product_id, reg_score in regular_products:
                freq = user_product_freq[(user_id, product_id)]
                print(f"    Product {product_id}: Regularity Score = {reg_score:.3f} "
                      f"(Purchased {freq} times)")
            
            print("\n  [DEBUG] Combined Score Recommendations (strength + confidence):")
            combined_scores = {p: (params.combined_strength_weight * metrics[0] + 
                                 params.combined_confidence_weight * metrics[1])
                              for p, metrics in user_relationships.items()}
            
            top_products = get_top_recommendations(
                user_id,
                combined_scores,
                params,
                lambda x: x[1],  # Sort by combined score
                "combined score"
            )
            for product_id, combined_score in top_products:
                metrics = user_relationships[product_id]
                print(f"    Product {product_id}: Combined Score = {combined_score:.3f} "
                      f"(Strength = {metrics[0]:.3f}, Confidence = {metrics[1]:.3f}, "
                      f"Purchased {metrics[3]} times)")
            
            print("\n  [DEBUG] Frequency-based Recommendations:")
            frequent_products = get_top_recommendations(
                user_id,
                {p: metrics[3] for p, metrics in user_relationships.items()},  # Just pass frequency
                params,
                lambda x: x[1],  # Sort by frequency
                "frequency"
            )
            for product_id, freq in frequent_products:
                metrics = user_relationships[product_id]
                print(f"    Product {product_id}: Purchased {freq} times "
                      f"(Strength = {metrics[0]:.3f}, Confidence = {metrics[1]:.3f}, "
                      f"Regularity = {metrics[2]:.3f})")
            
            print("\n" + "="*50 + "\n")  # Add separator between users
        
    finally:
        # Final cleanup
        if 'model' in locals():
            model.cleanup()
        if 'data' in locals():
            data.cleanup()
        clean_memory()
        print("Process completed. Memory cleaned up.")
