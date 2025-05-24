import pandas as pd
import numpy as np
import dask.dataframe as dd

def getDataFrames(orders_path="orders.csv", opt_path="order_products__train.csv", opp_path="order_products__prior.csv"):
    # Step 1: Read the CSV files
    orders = pd.read_csv(orders_path)
    order_products_train = pd.read_csv(opt_path)
    order_products_prior = pd.read_csv(opp_path)

    # Step 2: Identify users who have a "train" entry but NOT "test" entry
    test_users = orders.loc[orders['eval_set'] == 'test', 'user_id'].unique()
    train_users = orders.loc[orders['eval_set'] == 'train', 'user_id'].unique()
    
    # Filter out users who have test entries
    filtered_train_users = [user for user in train_users if user not in test_users]

    # Step 3: Remove all entries for users with test entries and keep only those with train
    orders_prior = orders[(orders['eval_set'] == 'prior') & (orders['user_id'].isin(filtered_train_users))]
    orders_train = orders[(orders['eval_set'] == 'train') & (orders['user_id'].isin(filtered_train_users))]

    # Fill NaN values in 'days_since_prior_order' with 0 before merging
    orders_prior['days_since_prior_order'] = orders_prior['days_since_prior_order'].fillna(0)
    orders_train['days_since_prior_order'] = orders_train['days_since_prior_order'].fillna(0)

    # Merge the orders with the corresponding order products based on the eval_set
    merged_prior = pd.merge(orders_prior, order_products_prior, on='order_id')
    merged_train = pd.merge(orders_train, order_products_train, on='order_id')

    # Step 4: Select only the columns user_id, order_id, product_id, and days_since_prior_order
    merged_prior = merged_prior[['user_id', 'order_id', 'product_id', 'days_since_prior_order']]
    merged_train = merged_train[['user_id', 'order_id', 'product_id', 'days_since_prior_order']]

    # Step 5: Filter baskets by size (3-50 items)
    basket_sizes_prior = merged_prior.groupby('order_id').size()
    basket_sizes_train = merged_train.groupby('order_id').size()
    
    valid_orders_prior = basket_sizes_prior[(basket_sizes_prior >= 3) & (basket_sizes_prior <= 50)].index
    valid_orders_train = basket_sizes_train[(basket_sizes_train >= 3) & (basket_sizes_train <= 50)].index
    
    merged_prior = merged_prior[merged_prior['order_id'].isin(valid_orders_prior)]
    merged_train = merged_train[merged_train['order_id'].isin(valid_orders_train)]

    # Step 6: Remove items that appear less than 17 times
    item_counts_prior = merged_prior['product_id'].value_counts()
    item_counts_train = merged_train['product_id'].value_counts()
    
    valid_items = set(item_counts_prior[item_counts_prior >= 17].index) & set(item_counts_train[item_counts_train >= 17].index)
    
    merged_prior = merged_prior[merged_prior['product_id'].isin(valid_items)]
    merged_train = merged_train[merged_train['product_id'].isin(valid_items)]

    # Step 7: Ensure users appear in both dataframes
    users_prior = set(merged_prior['user_id'].unique())
    users_train = set(merged_train['user_id'].unique())
    common_users = users_prior & users_train
    
    merged_prior = merged_prior[merged_prior['user_id'].isin(common_users)]
    merged_train = merged_train[merged_train['user_id'].isin(common_users)]

    # Step 8: Remap product IDs to sequential integers starting from 1 based on first appearance
    # Combine both dataframes and sort by user_id and order_id to ensure consistent ordering
    combined_df = pd.concat([merged_prior, merged_train]).sort_values(['user_id', 'order_id'])
    
    # Get unique items in order of first appearance
    first_appearance_items = combined_df['product_id'].unique()
    
    # Create mapping dictionary (old_id -> new_id)
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(first_appearance_items, start=1)}
    
    # Apply mapping to both dataframes
    merged_prior['product_id'] = merged_prior['product_id'].map(id_mapping)
    merged_train['product_id'] = merged_train['product_id'].map(id_mapping)

    # Convert pandas DataFrame to dask DataFrame
    ddf_prior = dd.from_pandas(merged_prior, npartitions=16)
    ddf_train = dd.from_pandas(merged_train, npartitions=16)

    print(f"Prior orders dataframe: {ddf_prior}")
    print(f"Train orders dataframe: {ddf_train}")
    print(f"Number of users after filtering: {len(common_users)}")
    print(f"Number of valid items: {len(valid_items)}")
    print(f"Product IDs have been remapped from 1 to {len(valid_items)}")
        
    return ddf_prior, ddf_train

# sample of dataset is used to test processes faster and on smaller sets
def process_largest_basket_sample(ddf, user_ids, output_csv="SampledSet\largest_baskets.csv"):
    # Step 1: Filter dataset for selected user IDs
    filtered_ddf = ddf[ddf['user_id'].isin(user_ids)].compute()

    # Step 2: Get order counts
    order_counts = filtered_ddf.groupby("order_id")["product_id"].count()

    # Step 3: Find the largest order
    largest_order_id = order_counts.idxmax()
    max_basket_size = order_counts.max()

    print(f"Largest order ID: {largest_order_id} with {max_basket_size} items.")

    # Step 4: Aggregate products into lists per order
    order_groups = (
        filtered_ddf.groupby(["user_id", "order_id", "days_since_prior_order"])["product_id"]
        .apply(list)
        .reset_index()
    )

    order_groups["product_id"] = order_groups["product_id"].apply(
        lambda x: x + [0] * (max_basket_size - len(x))  # Fill shorter lists with 0
    )

    # Step 6: Convert product lists into separate columns
    product_cols = pd.DataFrame(order_groups["product_id"].to_list(), columns=[f"item{i+1}" for i in range(max_basket_size)])
    final_df = pd.concat([order_groups.drop(columns=["product_id"]), product_cols], axis=1)

    # Step 7: Save to CSV
    final_df.to_csv(output_csv, index=False)
    print(f"Saved CSV to {output_csv}")

    return final_df


# Example usage
#ddf,test_df = getDataFrames("NBR_/orders.csv", 'NBR_/order_products__train.csv', 'NBR_/order_products__prior.csv')#
ddf,test_df = getDataFrames("orders.csv", 'order_products__train.csv', 'order_products__prior.csv')

#randPredG_o(ddf)

user_ids = list(range(1,30488)) #np.random.choice(range(1, 206209), size=10, replace=False)  # Random 100 users
#print(user_ids)
df_result = process_largest_basket_sample(ddf, user_ids, "trueVers/19k_train_baskets.csv")
df_result = process_largest_basket_sample(test_df, user_ids, "trueVers/19k_test_baskets.csv")


#print(getDataFrames())