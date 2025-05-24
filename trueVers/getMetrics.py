import dask.dataframe as dd
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from datetime import datetime
import os
import matplotlib.pyplot as plt

# File paths
truthFile = "trueVers/instacart_test_baskets.csv"
# prediction_files = [
#     "meta_learner_predictions_20250512_235701.csv",
#     "meta_learner_predictions_20250513_033816.csv",
#     "meta_learner_predictions_20250513_103906.csv",
#     "meta_learner_predictions_20250513_112420.csv",
#     "meta_learner_predictions_20250513_120448.csv",
#     "meta_learner_predictions_20250513_132939.csv",
#     "meta_learner_predictions_20250513_141217.csv",
# ]

prediction_files = [
    "10b_preds/meta_learner_predictions_20250523_224032.csv",
    #"10b_preds/meta_learner_predictions_20250523_222034.csv",
    #"10b_preds/meta_learner_predictions_20250520_003839.csv",
    #"10b_preds/meta_learner_predictions_20250520_000939.csv",
    # "10b_preds/meta_learner_predictions_20250513_160718.csv",
    # "10b_preds/meta_learner_predictions_20250513_165533.csv",
    # "10b_preds/meta_learner_predictions_20250513_174218.csv",
    # "10b_preds/meta_learner_predictions_20250513_183224.csv",
    # "10b_preds/meta_learner_predictions_20250513_191813.csv",
    # "10b_preds/meta_learner_predictions_20250513_200458.csv",
]

def get_truth(truth_df, id: int):
    '''Get the ground truth list for a user, removing zeros'''
    filtered_df = truth_df[truth_df['user_id'] == id]
    
    # Extract all item columns (item1, item2, etc.)
    item_columns = [col for col in filtered_df.columns if col.startswith('item')]
    
    # Get all items for this user
    all_items = []
    for _, row in filtered_df.iterrows():
        items = row[item_columns].tolist()
        all_items.extend(items)
    
    # Remove zeros from truth list
    sl = [item for item in all_items if item != 0]
    return sl

def get_model_predictions(pred_df, id: int, model: str):
    '''Get the prediction list for a specific model and user'''
    filtered_df = pred_df[pred_df['user_id'] == id]
    
    # Get the appropriate product columns based on the model
    if model == 'Meta':
        product_columns = [col for col in filtered_df.columns if col.startswith('meta_product')]
    elif model == 'PPO':
        product_columns = [col for col in filtered_df.columns if col.startswith('ppo_product')]
    elif model == 'GNN':
        product_columns = [col for col in filtered_df.columns if col.startswith('gnn4_product')]
    else:
        raise ValueError(f"Unknown model: {model}")
    
    # Get all products for this user and model
    products = []
    for _, row in filtered_df.iterrows():
        prods = row[product_columns].tolist()
        products.extend(prods)
    
    return products

def get_recall(truth_list, pred_list):
    '''Calculate recall: number of correct predictions / number of items in truth list'''
    if not truth_list:  # If truth list is empty
        return 0
    correct = sum(1 for pred in pred_list if pred in truth_list)
    return correct / len(truth_list)

def get_bhr(truth_list, pred_list):
    '''Calculate bhr: 1 if any prediction is in truth list, 0 otherwise'''
    if not truth_list:  # If truth list is empty
        return 0
    return 1 if any(pred in truth_list for pred in pred_list) else 0

def get_kappa(pred_list1, pred_list2):
    '''Calculate Cohen's Kappa between two prediction lists'''
    # If either list is empty, return 0
    if not pred_list1 or not pred_list2:
        return 0.0
        
    # Convert predictions to binary vectors (1 if item is predicted, 0 if not)
    all_items = set(pred_list1 + pred_list2)
    if not all_items:  # If no items in either list
        return 0.0
        
    binary1 = [1 if item in pred_list1 else 0 for item in all_items]
    binary2 = [1 if item in pred_list2 else 0 for item in all_items]
    
    # If all predictions are the same (all 0s or all 1s), return 1.0
    if all(x == 0 for x in binary1 + binary2) or all(x == 1 for x in binary1 + binary2):
        return 1.0
    
    try:
        # Calculate Cohen's Kappa
        return cohen_kappa_score(binary1, binary2)
    except:
        # If any error occurs in calculation, return 0
        return 0.0

def get_precision(truth_list, pred_list):
    '''Calculate precision: number of correct predictions / number of predictions made'''
    if not pred_list:  # If prediction list is empty
        return 0
    correct = sum(1 for pred in pred_list if pred in truth_list)
    return correct / len(pred_list)

def calculate_metrics_for_file(pred_file):
    # Load the dataframes
    print(f"Loading truth data...")
    truth_df = pd.read_csv(truthFile)
    print(f"Loading predictions from {pred_file}...")
    pred_df = pd.read_csv(pred_file)
    
    # Get unique user IDs from both dataframes
    truth_users = truth_df['user_id'].unique()
    pred_users = pred_df['user_id'].unique()
    common_users = list(set(truth_users) & set(pred_users))
    
    print(f"Found {len(common_users)} common users between truth and prediction data")
    
    # Initialize metrics for each model
    metrics = {
        'GNN': {'recall': 0, 'precision': 0, 'bhr': 0},
        'PPO': {'recall': 0, 'precision': 0, 'bhr': 0},
        'Meta': {'recall': 0, 'precision': 0, 'bhr': 0},
        'kappa': 0
    }
    user_count = 0
    
    # Process each user
    for uid in common_users:
        # Get truth list
        truth_list = get_truth(truth_df, uid)
        
        # Skip if truth list is empty after removing zeros
        if not truth_list:
            continue
        
        # Get predictions for each model
        gnn_preds = get_model_predictions(pred_df, uid, 'GNN')
        ppo_preds = get_model_predictions(pred_df, uid, 'PPO')
        meta_preds = get_model_predictions(pred_df, uid, 'Meta')
        
        # Calculate metrics for each model
        for model, preds in [('GNN', gnn_preds), ('PPO', ppo_preds), ('Meta', meta_preds)]:
            recall = get_recall(truth_list, preds)
            precision = get_precision(truth_list, preds)
            bhr = get_bhr(truth_list, preds)
            metrics[model]['recall'] += recall
            metrics[model]['precision'] += precision
            metrics[model]['bhr'] += bhr
        
        # Calculate kappa between PPO and GNN
        metrics['kappa'] += get_kappa(ppo_preds, gnn_preds)
        
        user_count += 1
        
        # Print progress every 1000 users
        if user_count % 1000 == 0:
            print(f"Processed {user_count} users")
    
    # Calculate averages
    if user_count > 0:
        for model in ['GNN', 'PPO', 'Meta']:
            metrics[model]['recall'] /= user_count
            metrics[model]['precision'] /= user_count
            metrics[model]['bhr'] /= user_count
        metrics['kappa'] /= user_count
    
    return metrics, user_count

def plot_metrics(all_metrics, prediction_files):
    # Extract file names for x-axis labels (remove path and extension)
    file_names = [os.path.splitext(os.path.basename(f))[0] for f in prediction_files]
    
    # Prepare data for plotting
    recalls = {
        'GNN': [m['GNN']['recall'] for m in all_metrics],
        'PPO': [m['PPO']['recall'] for m in all_metrics],
        'Meta': [m['Meta']['recall'] for m in all_metrics]
    }
    
    precisions = {
        'GNN': [m['GNN']['precision'] for m in all_metrics],
        'PPO': [m['PPO']['precision'] for m in all_metrics],
        'Meta': [m['Meta']['precision'] for m in all_metrics]
    }
    
    bhrs = {
        'GNN': [m['GNN']['bhr'] for m in all_metrics],
        'PPO': [m['PPO']['bhr'] for m in all_metrics],
        'Meta': [m['Meta']['bhr'] for m in all_metrics]
    }
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot recalls
    for model, values in recalls.items():
        ax1.plot(file_names, values, marker='o', label=model)
    ax1.set_title('Recall Metrics Across Files')
    ax1.set_xlabel('Files')
    ax1.set_ylabel('Recall')
    ax1.grid(True)
    ax1.legend()
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot precisions
    for model, values in precisions.items():
        ax2.plot(file_names, values, marker='o', label=model)
    ax2.set_title('Precision Metrics Across Files')
    ax2.set_xlabel('Files')
    ax2.set_ylabel('Precision')
    ax2.grid(True)
    ax2.legend()
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot bhrs
    for model, values in bhrs.items():
        ax3.plot(file_names, values, marker='o', label=model)
    ax3.set_title('BHR Metrics Across Files')
    ax3.set_xlabel('Files')
    ax3.set_ylabel('BHR')
    ax3.grid(True)
    ax3.legend()
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('metrics_comparison.png')
    plt.close()

def calculate_metrics():
    all_metrics = []
    total_users = 0
    
    # Calculate metrics for each prediction file
    for pred_file in prediction_files:
        print(f"\nProcessing file: {pred_file}")
        metrics, user_count = calculate_metrics_for_file(pred_file)
        all_metrics.append(metrics)
        total_users += user_count
    
    # Aggregate results
    if all_metrics:
        aggregated_metrics = {
            'GNN': {'recall': 0, 'precision': 0, 'bhr': 0},
            'PPO': {'recall': 0, 'precision': 0, 'bhr': 0},
            'Meta': {'recall': 0, 'precision': 0, 'bhr': 0},
            'kappa': 0
        }
        
        # Sum up all metrics
        for metrics in all_metrics:
            for model in ['GNN', 'PPO', 'Meta']:
                aggregated_metrics[model]['recall'] += metrics[model]['recall']
                aggregated_metrics[model]['precision'] += metrics[model]['precision']
                aggregated_metrics[model]['bhr'] += metrics[model]['bhr']
            aggregated_metrics['kappa'] += metrics['kappa']
        
        # Calculate final averages
        num_files = len(all_metrics)
        for model in ['GNN', 'PPO', 'Meta']:
            aggregated_metrics[model]['recall'] /= num_files
            aggregated_metrics[model]['precision'] /= num_files
            aggregated_metrics[model]['bhr'] /= num_files
        aggregated_metrics['kappa'] /= num_files
        
        # Print final results
        print("\nAggregated Results Across All Files:")
        for model in ['GNN', 'PPO', 'Meta']:
            print(f"\n{model} Model Metrics:")
            print(f"Average Recall: {aggregated_metrics[model]['recall']:.4f}")
            print(f"Average Precision: {aggregated_metrics[model]['precision']:.4f}")
            print(f"Average BHR: {aggregated_metrics[model]['bhr']:.4f}")
        
        print(f"\nAverage Kappa between PPO and GNN: {aggregated_metrics['kappa']:.4f}")
        print(f"\nTotal number of users processed: {total_users}")
        
        # Create visualization
        plot_metrics(all_metrics, prediction_files)

        # Print raw metrics for each file
        print("\nRaw metrics per file:")
        for fname, metrics in zip(prediction_files, all_metrics):
            print(f"{fname}")
            for model in ['GNN', 'PPO', 'Meta']:
                print(f"  {model}: Recall={metrics[model]['recall']:.4f}, Precision={metrics[model]['precision']:.4f}, BHR={metrics[model]['bhr']:.4f}")
            print(f"  Kappa: {metrics['kappa']:.4f}")
        
        return aggregated_metrics
    
    return None

if __name__ == "__main__":
    calculate_metrics()



