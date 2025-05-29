import pandas as pd
import numpy as np

# Read the data files
orders = pd.read_csv('orders.csv')
order_products_prior = pd.read_csv('order_products__prior.csv')
order_products_train = pd.read_csv('order_products__train.csv')

# Get first 20000 unique users
unique_users = orders['user_id'].unique()[:19435]
orders = orders[orders['user_id'].isin(unique_users)]

# Merge orders with prior products
prior_orders = orders[orders['eval_set'] == 'prior']
prior_merged = pd.merge(prior_orders, order_products_prior, on='order_id')

# Merge orders with train products
train_orders = orders[orders['eval_set'] == 'train']
train_merged = pd.merge(train_orders, order_products_train, on='order_id')

# Combine the merged dataframes
combined_data = pd.concat([prior_merged, train_merged])

# Fill null values in time_since_last_basket with 0
combined_data['days_since_prior_order'] = combined_data['days_since_prior_order'].fillna(0)

# Create initial dataframe with exact columns
initial_data = pd.DataFrame({
    'user_id': combined_data['user_id'],
    'order_id': combined_data['order_id'],
    'days_since_prior_order': combined_data['days_since_prior_order'],
    'product_id': combined_data['product_id']
})

# Sort the data
initial_data = initial_data.sort_values(['user_id', 'order_id'])

# Filter out baskets that don't meet size requirements first
print("\nFiltering basket sizes...")
basket_sizes = initial_data.groupby(['user_id', 'order_id']).size()
valid_baskets = basket_sizes[(basket_sizes >= 3) & (basket_sizes <= 50)].index
initial_data = initial_data[initial_data.set_index(['user_id', 'order_id']).index.isin(valid_baskets)]

# First pass: Count frequencies and identify products to keep
product_counts = {}
for product_id in initial_data['product_id']:
    product_counts[product_id] = product_counts.get(product_id, 0) + 1

# Filter out products that appear less than 17 times
valid_products = {pid for pid, count in product_counts.items() if count >= 17}

print("Remapping products")
# Create new mapping for remaining products (first come first serve)
product_mapping = {}
current_index = 1
# Map products in order of first appearance
for product_id in initial_data['product_id'].unique():
    if product_id in valid_products:
        product_mapping[product_id] = current_index
        current_index += 1

# Filter the data to keep only valid products and apply mapping
final_data = initial_data[initial_data['product_id'].isin(valid_products)].copy()
final_data['product_id'] = final_data['product_id'].map(product_mapping)

# Find maximum basket size while processing users
print("\nProcessing users:")
current_user_id = None
current_order_id = None
index = 1
new_order_ids = []
max_basket_size = 0
current_basket_size = 0

print("Processing baskets...")
for _, row in final_data.iterrows():
    if current_user_id != row['user_id']:
        current_user_id = row['user_id']
        current_order_id = row['order_id']
        index = 1
        current_basket_size = 1
        print(f"Processing user_id: {current_user_id}")
    elif current_order_id != row['order_id']:
        current_order_id = row['order_id']
        index += 1
        max_basket_size = max(max_basket_size, current_basket_size)
        current_basket_size = 1
    else:
        current_basket_size += 1
    
    new_order_ids.append(index)

# Update max_basket_size with the last basket
max_basket_size = max(max_basket_size, current_basket_size)
final_data['order_id'] = new_order_ids

print("Reshaping data into basket format...")
# Reshape the data into basket format
basket_data = []
for (user_id, order_id), group in final_data.groupby(['user_id', 'order_id']):
    basket = {
        'user_id': user_id,
        'order_id': order_id,
        'days_since_prior_order': group['days_since_prior_order'].iloc[0]
    }
    # Add items as columns
    for i, product_id in enumerate(group['product_id'], 1):
        basket[f'item{i}'] = product_id
    # Fill remaining item columns with 0
    for i in range(len(group) + 1, max_basket_size + 1):
        basket[f'item{i}'] = 0
    basket_data.append(basket)

# Create the final basket dataframe
basket_df = pd.DataFrame(basket_data)

# Separate last order of each user into test set
print("\nSeparating test set...")
test_data = []
train_data = []

# Group by user_id and get the last order for each user
for user_id, user_group in basket_df.groupby('user_id'):
    # Sort by order_id to ensure we get the last order
    user_group = user_group.sort_values('order_id')
    # Get the last order
    last_order = user_group.iloc[-1]
    test_data.append(last_order)
    # Add all other orders to training data
    train_data.extend(user_group.iloc[:-1].to_dict('records'))

# Create train and test dataframes
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# Save the processed data
train_df.to_csv('trueVers/instacart_train_baskets.csv', index=False)
test_df.to_csv('trueVers/instacart_test_baskets.csv', index=False)

# Save the product mapping
pd.DataFrame(list(product_mapping.items()), columns=['original_id', 'new_id']).to_csv('product_mapping.csv', index=False)

print(f"\nProcessed {len(train_df)} training orders")
print(f"Processed {len(test_df)} test orders")
print(f"Total unique products after filtering: {len(product_mapping)}")
print(f"Maximum basket size: {max_basket_size}")
