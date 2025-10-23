import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Path to your file
file_path = "amazon_dataset.json"

# Each line is a JSON object → use lines=True
df = pd.read_json(file_path, lines=True)

print("Shape:", df.shape)
print(df.head(3))
# Keep only the columns we need
df = df[['reviewerID', 'asin', 'overall', 'unixReviewTime']].rename(
    columns={
        'reviewerID': 'user_id',
        'asin': 'product_id',
        'overall': 'rating',
        'unixReviewTime': 'timestamp'
    }
)

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

df.head()

user_enc = LabelEncoder()
prod_enc = LabelEncoder()

df['user_id'] = user_enc.fit_transform(df['user_id'])
df['product_id'] = prod_enc.fit_transform(df['product_id'])

print(f"Unique users: {df['user_id'].nunique()}")
print(f"Unique products: {df['product_id'].nunique()}")
