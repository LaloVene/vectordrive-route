import numpy as np
from sklearn.cluster import KMeans

def compute_kmeans_anchors(dataset, num_anchors=16, random_state=42):
    """
    Extracts all future trajectories from the training dataset
    and clusters them into K static trajectory anchors.
    """
    print(f"Collecting trajectories for K-Means clustering (K={num_anchors})...")
    trajectories = []
    
    # We can read future_traj from dataset's internal samples list to avoid loading heavy images/sensors
    for meta in dataset.samples_list:
        trajectories.append(meta['future_traj'])
        
    trajectories = np.array(trajectories) # Shape: (N, 8, 2)
    N = trajectories.shape[0]
    print(f"Total trajectories collected: {N}")
    
    # Flatten trajectory waypoints for clustering: shape (N, 16)
    flat_trajectories = trajectories.reshape(N, -1)
    
    kmeans = KMeans(n_clusters=num_anchors, random_state=random_state, n_init=10)
    kmeans.fit(flat_trajectories)
    
    # Cluster centers shape: (K, 16) -> reshape to (K, 8, 2)
    anchors = kmeans.cluster_centers_.reshape(num_anchors, 8, 2)
    
    # Sort anchors based on final waypoint's lateral position x (from left to right) 
    # to maintain a consistent ordering in the model logits
    final_x = anchors[:, -1, 0]
    sort_indices = np.argsort(final_x)
    anchors = anchors[sort_indices]
    
    print("K-Means anchors computed and sorted successfully.")
    return anchors
