import torch
import numpy as np
import pandas as pd
import umap
import plotly.express as px
from sklearn.cluster import DBSCAN
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import SmartBirdDataset
from model import BirdEmbedder
import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("🕵️  STARTING FORENSIC ANALYSIS OF EMBEDDINGS...")
    
    # 1. Load Metadata & Dataset
    # We need the raw dataframe to get Scientific Names (Genus)
    ds = SmartBirdDataset(mode='train', task='species')
    df = ds.df.copy()
    
    # Subset for speed/clarity (2500 points)
    indices = np.random.choice(len(ds), 2500, replace=False)
    subset_ds = Subset(ds, indices)
    
    # Get Genus for coloring (Scientific name is "Genus species")
    # This cleans up the visual noise significantly
    subset_df = df.iloc[indices].copy()
    subset_df['Genus'] = subset_df['scientific'].apply(lambda x: x.split()[0] if isinstance(x, str) else "Unknown")
    
    # 2. Extract Embeddings
    loader = DataLoader(subset_ds, batch_size=32, shuffle=False)
    model = BirdEmbedder(embedding_size=512, pretrained=False)
    model.load_state_dict(torch.load("data/models/bird_embedder.pth"))
    model.to(DEVICE).eval()
    
    print("   Extracting vectors...")
    vecs = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader):
            vecs.append(model(imgs.to(DEVICE)).cpu().numpy())
    X = np.vstack(vecs)
    
    # 3. Run UMAP (2D Projection)
    print("   Projecting to 2D (UMAP)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    projections = reducer.fit_transform(X)
    
    subset_df['x'] = projections[:, 0]
    subset_df['y'] = projections[:, 1]
    
    # 4. AUTOMATIC CLUSTER DETECTION (DBSCAN)
    # This finds dense blobs automatically
    print("   Running DBSCAN to identify isolated clusters...")
    # eps is the distance; min_samples is how many dots make a cluster
    clustering = DBSCAN(eps=0.5, min_samples=20).fit(projections)
    subset_df['cluster'] = clustering.labels_
    
    # Analyze the clusters
    print("\n" + "="*40)
    print("🔍 MYSTERY CLUSTER ANALYSIS")
    print("="*40)
    
    # Group by cluster ID
    unique_clusters = sorted(subset_df['cluster'].unique())
    for cluster_id in unique_clusters:
        if cluster_id == -1: continue # -1 is "Noise" / "Main Cloud"
        
        cluster_data = subset_df[subset_df['cluster'] == cluster_id]
        # Get top 3 species in this cluster
        top_species = cluster_data['species'].value_counts().head(3)
        top_genus = cluster_data['Genus'].value_counts().head(1)
        
        print(f"📍 Cluster {cluster_id}: Found {len(cluster_data)} points.")
        print(f"   Dominant Genus: {top_genus.index[0]} ({top_genus.values[0]} count)")
        print(f"   Top Species: {', '.join(top_species.index)}")
        print("-" * 20)

    # 5. Generate Interactive Plot
    print("\n🎨 Generating Interactive HTML Map...")
    fig = px.scatter(
        subset_df, 
        x='x', 
        y='y', 
        color='Genus', # Coloring by Family is much cleaner than Species
        hover_data=['species', 'scientific'],
        title='Interactive Biodiversity Map (Color by Genus)',
        width=1000, 
        height=800
    )
    fig.update_traces(marker=dict(size=5))
    
    out_path = "data/visuals/interactive_map.html"
    fig.write_html(out_path)
    print(f"✅ Saved to {out_path}")
    print("   -> Download this file and open it in Chrome/Edge to explore!")

if __name__ == "__main__":
    main()
