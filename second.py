import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_json("amazon_dataset.json", lines=True)

# Keep only needed columns
df = df[['reviewerID', 'asin', 'overall', 'unixReviewTime']].rename(
    columns={'reviewerID':'user_id', 'asin':'product_id', 'overall':'rating', 'unixReviewTime':'timestamp'}
)

# Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

# Encode user/product IDs as integers
user_enc = LabelEncoder()
prod_enc = LabelEncoder()
df['user_id'] = user_enc.fit_transform(df['user_id'])
df['product_id'] = prod_enc.fit_transform(df['product_id'])

print(df.head())
import torch
from torch.utils.data import Dataset, DataLoader

class ReviewDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(df['product_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float)
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

dataset = ReviewDataset(df)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

import torch.nn as nn

class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=8):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.item_emb = nn.Embedding(num_items, embedding_size)
    
    def forward(self, user, item):
        user_vec = self.user_emb(user)
        item_vec = self.item_emb(item)
        return (user_vec * item_vec).sum(1)

num_users = df['user_id'].nunique()
num_items = df['product_id'].nunique()

model = RecommenderNet(num_users, num_items)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(10):
    total_loss = 0
    for u, i, r in loader:
        optimizer.zero_grad()
        preds = model(u, i)
        loss = criterion(preds, r)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

import torch

def recommend_top_n(model, user_id, n=5):
    # Predict for all items
    user_tensor = torch.tensor([user_id]*num_items)
    item_tensor = torch.arange(num_items)
    scores = model(user_tensor, item_tensor)
    
    top_items = torch.topk(scores, n).indices.tolist()
    return top_items

print("Top 5 recommendations for user 0:", recommend_top_n(model, 0, n=5))
top_n_asins = prod_enc.inverse_transform([24, 15, 14, 27, 6])
print(top_n_asins)
