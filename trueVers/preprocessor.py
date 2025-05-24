# This is the preprocessor for the instacart dataset.

import pandas as pd
import numpy as np

# Read instacart_future and create a set of user IDs
future_users = pd.read_csv('trueVers/instacart_future.csv', dtype={'user_id': int})
future_user_set = set(future_users['user_id'].unique())
print(f"Number of users in future dataset: {len(future_user_set)}")
print(f"Maximum user ID in future dataset: {max(future_user_set)}")
print(f"Minimum user ID in future dataset: {min(future_user_set)}")

user_order_d = pd.read_csv('orders.csv',
                         usecols=['user_id', 'order_number', 'order_id', 'eval_set', 'days_since_prior_order'],
                         dtype={'user_id': int})
print(f"Number of users in orders.csv: {len(user_order_d['user_id'].unique())}")

# Get the first 19435 users
first_users = sorted(user_order_d['user_id'].unique())[:19435]
print(f"Number of users to process: {len(first_users)}")
print(f"First user ID: {first_users[0]}")
print(f"Last user ID: {first_users[-1]}")

order_item_train = pd.read_csv('order_products__train.csv',
                               usecols=['order_id', 'product_id'])
order_item_prior = pd.read_csv('order_products__prior.csv',
                               usecols=['order_id', 'product_id'])
order_item = pd.concat([order_item_prior, order_item_train], ignore_index=True)

# Remap product IDs to start from 1
unique_products = sorted(order_item['product_id'].unique())
product_map = {old_id: new_id for new_id, old_id in enumerate(unique_products, 1)}
order_item['product_id'] = order_item['product_id'].map(product_map)

user_order = pd.merge(user_order_d, order_item, on='order_id', how='left')
print(f"Number of users after merge: {len(user_order['user_id'].unique())}")

user_order = user_order.dropna(how='any')
print(f"Number of users after dropna: {len(user_order['user_id'].unique())}")

# Fill NaN values in days_since_prior_order with 0
user_order['days_since_prior_order'] = user_order['days_since_prior_order'].fillna(0)

# Filter users to only include the first 19435
user_order = user_order[user_order['user_id'].isin(first_users)]
print(f"Number of users after filtering: {len(user_order['user_id'].unique())}")

# Debug missing users
all_users = set(range(1, max(future_user_set) + 1))
missing_users = all_users - set(user_order['user_id'].unique())
print(f"Number of missing users: {len(missing_users)}")
print(f"First 10 missing users: {sorted(list(missing_users))[:10]}")
print(f"Last 10 missing users: {sorted(list(missing_users))[-10:]}")

# Check if specific users are in the filtered data
for user in test_users:
    print(f"User {user} in filtered data: {user in user_order['user_id'].unique()}")

baskets = None
for user, user_data in user_order.groupby('user_id'):
    date_list = list(set(user_data['order_number'].tolist()))
    date_list = sorted(date_list)
    print(f"Processing user {user} with {len(date_list)} orders")
    date_num = 1
    for date in date_list:
        date_data = user_data[user_data['order_number'].isin([date])]
        date_item = list(set(date_data['product_id'].tolist()))
        item_num = len(date_item)
        days_since = date_data['days_since_prior_order'].iloc[0]  # Get days since prior order
        order_id = date_data['order_id'].iloc[0]  # Get order_id
        
        if baskets is None:
            baskets = pd.DataFrame({'user_id': pd.Series([user for i in range(item_num)]),
                                    'order_number': pd.Series([date_num for i in range(item_num)]),
                                    'order_id': pd.Series([order_id for i in range(item_num)]),
                                    'product_id': pd.Series(date_item),
                                    'eval_set': pd.Series(['prior' for i in range(item_num)]),
                                    'days_since_prior_order': pd.Series([days_since for i in range(item_num)])})
            date_num += 1
        else:
            if date == date_list[-1]:#if date is the last. then add a tag here
                temp = pd.DataFrame({'user_id': pd.Series([user for i in range(item_num)]),
                                        'order_number': pd.Series([date_num for i in range(item_num)]),
                                        'order_id': pd.Series([order_id for i in range(item_num)]),
                                        'product_id': pd.Series(date_item),
                                        'eval_set': pd.Series(['train' for i in range(item_num)]),
                                        'days_since_prior_order': pd.Series([days_since for i in range(item_num)])})
                date_num += 1
                baskets = pd.concat([baskets, temp], ignore_index=True)
            else:
                temp = pd.DataFrame({'user_id': pd.Series([user for i in range(item_num)]),
                                        'order_number': pd.Series([date_num for i in range(item_num)]),
                                        'order_id': pd.Series([order_id for i in range(item_num)]),
                                        'product_id': pd.Series(date_item),
                                        'eval_set': pd.Series(['prior' for i in range(item_num)]),
                                        'days_since_prior_order': pd.Series([days_since for i in range(item_num)])})
                date_num += 1
                baskets = pd.concat([baskets, temp], ignore_index=True)

print('total transactions:', len(baskets))
print(f"Number of unique users in baskets: {len(baskets['user_id'].unique())}")

# Check if specific users are in the baskets
for user in test_users:
    print(f"User {user} in baskets: {user in baskets['user_id'].unique()}")

# Find the maximum basket size
max_basket_size = baskets.groupby(['user_id', 'order_id'])['product_id'].count().max()

# Create a function to transform basket data
def transform_basket(group):
    items = group['product_id'].tolist()
    # Pad with zeros if needed
    items.extend([0] * (max_basket_size - len(items)))
    return pd.Series(items)

# Split data into last orders and other orders
last_orders = baskets.groupby('user_id').apply(lambda x: x[x['order_number'] == x['order_number'].max()]).reset_index(drop=True)
other_orders = baskets[~baskets.index.isin(last_orders.index)]

print(f"Number of users in last orders: {len(last_orders['user_id'].unique())}")
print(f"Number of users in other orders: {len(other_orders['user_id'].unique())}")

# Check if specific users are in the last orders
for user in test_users:
    print(f"User {user} in last orders: {user in last_orders['user_id'].unique()}")

# Transform both datasets
last_orders_transformed = last_orders.groupby(['user_id', 'order_id', 'days_since_prior_order']).apply(transform_basket).reset_index()
other_orders_transformed = other_orders.groupby(['user_id', 'order_id', 'days_since_prior_order']).apply(transform_basket).reset_index()

# Rename columns
item_columns = [f'item{i+1}' for i in range(max_basket_size)]
last_orders_transformed.columns = ['user_id', 'order_id', 'days_since_prior_order'] + item_columns
other_orders_transformed.columns = ['user_id', 'order_id', 'days_since_prior_order'] + item_columns

# Save to CSV files
last_orders_transformed.to_csv('instacart_test_baskets.csv', index=False)
other_orders_transformed.to_csv('instacart_train_baskets.csv', index=False)