## Inefficient method - e,n,u is faster
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box, Point
from joblib import Parallel, delayed
from tqdm import tqdm
from gbif import build_geodataframe
from helper import _create_directories
import os

def _build_grid(cell_size_m, bbox):
    # Approximate conversion factors
    deg_per_meter_lat = 1 / 111000  # degrees per meter for latitude
    deg_per_meter_lon = 1 / (111000 * np.cos(np.radians((bbox[1] + bbox[3]) / 2)))  # degrees per meter for longitude

    # Calculate the number of rows and columns
    lat_range = bbox[3] - bbox[1]  # degrees of latitude
    lon_range = bbox[2] - bbox[0]  # degrees of longitude

    nrows = int(np.ceil(lat_range / (cell_size_m * deg_per_meter_lat)))
    ncols = int(np.ceil(lon_range / (cell_size_m * deg_per_meter_lon)))

    # Use tqdm to create a progress bar
    tasks = [(i, j, cell_size_m, deg_per_meter_lat, deg_per_meter_lon, bbox) for i in range(nrows) for j in range(ncols)]
    with Parallel(n_jobs=10) as parallel:
        grid_polygons = list(Parallel(n_jobs=-1)(delayed(_create_cell)(*task) for task in tasks))

    return grid_polygons

def _create_cell(i, j, cell_size_m, deg_per_meter_lat, deg_per_meter_lon, bbox):
    # Calculate the corners of the cell
    min_lat = bbox[1] + i * cell_size_m * deg_per_meter_lat
    max_lat = bbox[1] + (i + 1) * cell_size_m * deg_per_meter_lat
    min_lon = bbox[0] + j * cell_size_m * deg_per_meter_lon
    max_lon = bbox[0] + (j + 1) * cell_size_m * deg_per_meter_lon

    # Create a polygon for the cell
    return box(min_lon, min_lat, max_lon, max_lat)

def _create_state_grid(usa_polygon, state, cell_size_m = 512):
    # Take bounds -> Make grid -> Overlay with multi-polygon boundary & prune.
    state_polygon = usa_polygon[usa_polygon['name'] == state]

    state_grid = _build_grid(cell_size_m, bbox = state_polygon.total_bounds)
    state_grid_gdf = gpd.GeoDataFrame(geometry=state_grid)
    state_grid_gdf.set_crs(usa_polygon.crs, inplace=True)

    grid_within_state = gpd.sjoin(state_grid_gdf, state_polygon, how='inner', predicate='intersects')
    
    # Create unique key for each grid cell
    grid_within_state = grid_within_state.reset_index(drop = True)
    grid_within_state['key'] = grid_within_state['name'].astype(str) + '_' + grid_within_state.index.astype(str)
    grid_within_state = grid_within_state[['key', 'geometry']]

    return grid_within_state[['key', 'geometry']]

def _prune_empty_cells(occurrence_gdf, grid_within_state, state):
    state_occurrence_gdf = occurrence_gdf[occurrence_gdf['stateProvince'] == state].copy()
    state_occurrence_gdf.set_crs(grid_within_state.crs, inplace=True)
    observations_in_cells = gpd.sjoin(state_occurrence_gdf, grid_within_state, how='inner', predicate='within')

    return observations_in_cells, state_occurrence_gdf


if __name__ == '__main__':

    dimension = 512

    output_geojson=f'/data/cher/universe7/herbarium/data/geocell/clusters_{dimension}m.geojson'
    output_csv=f'/data/cher/universe7/herbarium/data/geocell/clusters_key_{dimension}m.csv'

    output_geojson='/data/cher/universe7/herbarium/data/geocell/grid.geojson'
    output_csv='/data/cher/universe7/herbarium/data/geocell/grid_key.csv'

    _create_directories(output_geojson,
                        output_csv)

    usa_polygon = gpd.read_file('/data/cher/universe7/herbarium/data/us-state-boundaries.geojson')[['name', 'geometry']]

    occurrence_gdf = build_geodataframe(gbif_path = "/data/cher/universe7/herbarium/data/MO-herbarium/occurrence.txt")
    occurrence_gdf.set_crs(usa_polygon.crs, inplace=True)

    for state in tqdm(occurrence_gdf['stateProvince'].unique(), desc="Processing States"):

        # Create grid for state within state polygon boundaries.
        grid_within_state = _create_state_grid(usa_polygon, state, cell_size_m = dimension)

        # Find occurrences within cells. Prune empty grid cells.
        observations_in_cells, state_occurrence_gdf = _prune_empty_cells(occurrence_gdf, grid_within_state, state)

        # Save {occurrenceID, grid key}
        observations_in_cells[['occurrenceID', 'key']].to_csv(output_csv, mode='a', header=False, index=False)

        if not os.path.exists(output_geojson):
            grid_within_state.to_file(output_geojson, driver="GeoJSON")
        else:
            grid_within_state.to_file(output_geojson, driver="GeoJSON", mode='a')

        # Calculate the number of cells with at least one observation
        num_cells_with_observations = observations_in_cells['index_right'].nunique()
        observations_per_cell = observations_in_cells.groupby('index_right').size().reset_index(name='observations_count').sort_values('observations_count', ascending = False)

        print(f"Occurrences: {state_occurrence_gdf.shape[0]}\nSatellite images: {num_cells_with_observations}\nDescriptions per image {np.mean(observations_per_cell.observations_count)}")

    print(f"Saved all grids to {output_geojson}")
    print(f"Saved occurrenceID and grid mapping to {output_csv}")