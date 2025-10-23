import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

file_path = "Grocery_and_Gourmet_Food.json"
chunk_size = 100000  # process 100k rows at a time
chunks = []

for chunk in pd.read_json(file_path, lines=True, chunksize=chunk_size):
    # Keep only relevant columns
    chunk = chunk[['reviewerID','asin','overall']]
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
print(df.shape)

user_enc = LabelEncoder()
prod_enc = LabelEncoder()

df['user_id'] = user_enc.fit_transform(df['reviewerID'])
df['product_id'] = prod_enc.fit_transform(df['asin'])
df['rating'] = df['overall'].astype(float)

# Keep only minimal columns
df = df[['user_id','product_id','rating']]

# Save for later use
df.to_parquet("grocery_prepared.parquet", index=False)
print(f"Unique users: {df['user_id'].nunique()}, Unique products: {df['product_id'].nunique()}")



class GroceryDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(df['product_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float)
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

dataset = GroceryDataset(df)
loader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=0)  # num_workers=0 avoids multiprocessing memory issues



num_users = df['user_id'].nunique()
num_items = df['product_id'].nunique()

class DeepRecommender(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=64, hidden_size=128):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.fc = nn.Sequential(
            nn.Linear(embedding_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, user, item):
        user_vec = self.user_emb(user)
        item_vec = self.item_emb(item)
        x = torch.cat([user_vec, item_vec], dim=1)
        return self.fc(x).squeeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepRecommender(num_users, num_items).to(device)
