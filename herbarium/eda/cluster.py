import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from distance import max_haversine_distance, generate_radius, max_distance
import random
from shapely.geometry import box

# Function to cluster points using DBSCAN
def perform_dbscan(gdf, max_distance, eps):
    """Perform DBSCAN clustering on the given GeoDataFrame."""
    coords = np.array(list(zip(gdf['decimalLongitude'], gdf['decimalLatitude'])))
    db = DBSCAN(eps=eps, min_samples=1).fit(coords)
    return db.labels_

# Function to check for clusters with max distance greater than threshold
def check_max_haversine_distances(gdf):
    """Calculate the max distances for each cluster."""
    max_distances = gdf.reset_index(drop=True).groupby('cluster').apply(max_haversine_distance).reset_index()
    max_distances.columns = ['cluster', 'max_distance_meters']
    return max_distances

def check_max_distances(gdf):
    """Calculate the max distances for each cluster."""
    max_distances = gdf.reset_index(drop=True).groupby('cluster').apply(max_distance).reset_index()
    max_distances.columns = ['cluster', 'max_distance_meters']
    return max_distances

# Function to refine clusters based on max distance
def refine_clusters(input_gdf, max_distance, num_clusters, state):
    """
    TODO: separate functions
    Refine clusters that exceed the max distance threshold. Sample randomly within cluster, create bbox and repeat until no longer necessary
    """
    while True:
        input_gdf = input_gdf.set_index('occurrenceID')
        max_distances = check_max_distances(input_gdf)

        clusters_too_large = max_distances[max_distances['max_distance_meters'] > max_distance]['cluster'].values
        if len(clusters_too_large) == 0:
            break  # Exit loop if no clusters exceed the max distance

        gdf_too_large = input_gdf[input_gdf['cluster'].isin(clusters_too_large)].copy()

        while len(clusters_too_large) > 0:
            for cluster in clusters_too_large:
                # _get_point
                # _create_bbox
                # _assign_new_cluster
                # Get points from the cluster that is too large
                points = gdf_too_large[gdf_too_large['cluster'] == cluster]

                # Randomly sample 1 point as center
                sampled_point = points.sample(n=1).iloc[0]

                # Define bounding box around the sampled point
                buffer_size = (max_distance / 111320) / 2  # Approx. degrees -- both ways
                lon, lat = sampled_point['decimalLongitude'], sampled_point['decimalLatitude']

                # Create bounding box
                bbox = box(lon - buffer_size, lat - buffer_size, lon + buffer_size, lat + buffer_size)

                # Assign points within the bounding box to the same cluster
                mask = (gdf_too_large['decimalLongitude'].between(lon - buffer_size, lon + buffer_size) &
                        gdf_too_large['decimalLatitude'].between(lat - buffer_size, lat + buffer_size))
                
                # If the sampled point's cluster has points nearby, assign them to a new cluster
                gdf_too_large.loc[mask, 'cluster'] = num_clusters + 1
                num_clusters += 1

            # Update the list of clusters that are too large based on the refined clusters
            max_distances = check_max_distances(gdf_too_large)
            clusters_too_large = max_distances[max_distances['max_distance_meters'] > max_distance]['cluster'].values
            print(num_clusters, len(clusters_too_large), gdf_too_large[gdf_too_large['cluster'].isin(clusters_too_large)].shape[0])

        # Update the original GeoDataFrame with refined clusters
        input_gdf.loc[input_gdf.index.isin(gdf_too_large.index), 'cluster'] = gdf_too_large['cluster'].values
        
        # Renumber clusters and update the cluster count
        input_gdf = renumber_clusters(input_gdf, state)
        num_clusters = len(input_gdf['cluster'].unique())

    return input_gdf


# Function to renumber clusters
def renumber_clusters(gdf, state):
    """Renumber clusters and create a state-specific cluster label."""
    gdf['state_cluster'] = [f"{state}_{cluster}" for cluster in pd.factorize(gdf['cluster'])[0]]
    return gdf

# Main function to run the clustering process
def geocell_clustering(gdf, state, max_distance):
    """Cluster points within a specified state and max distance."""
    
    # Perform initial DBSCAN clustering
    radius_deg = generate_radius(max_distance)

    labels = perform_dbscan(gdf, max_distance, eps = radius_deg)
    gdf['cluster'] = labels
    num_clusters = labels.max()

    print('Performed initial DBSCAN')

    # Refine clusters until all meet the max distance threshold
    gdf = refine_clusters(gdf, radius_deg, num_clusters, state)

    print('Refined clusters')

    # Renumber clusters and create state-specific labels
    gdf = renumber_clusters(gdf, state)

    print('Renumbered clusters')
    
    return gdf
