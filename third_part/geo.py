#!/usr/bin/python3.10
# coding=utf-8

import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily
import sklearn.cluster
from shapely.geometry import Point

#%%
def make_geo(df_accidents: pd.DataFrame, df_locations: pd.DataFrame) -> geopandas.GeoDataFrame:
    """
    Creates a GeoDataFrame by merging and cleaning accident and location data.

    Args:
        df_accidents (pd.DataFrame): DataFrame of accidents with 'p1' column.
        df_locations (pd.DataFrame): DataFrame of locations with 'p1', 'd', and 'e' columns.

    Returns:
        gpd.GeoDataFrame: Cleaned GeoDataFrame with valid coordinates in CRS EPSG:4326.
    """
    # Merge the accidents and locations dataframes based on column 'p1'
    merged_df = pd.merge(df_accidents, df_locations, on="p1")
    
    # Remove rows with missing or zero coordinates
    merged_df = merged_df.dropna(subset=["d", "e"])
    merged_df = merged_df[(merged_df["d"] != 0) & (merged_df["e"] != 0)]
    
    # Swap coordinates if 'd' is less than 'e' to maintain consistency
    swapped = merged_df["d"] < merged_df["e"]
    merged_df.loc[swapped, ["d", "e"]] = merged_df.loc[swapped, ["e", "d"]].values
    
    # Convert coordinates into geometry column with Point objects
    merged_df["geometry"] = merged_df.apply(lambda row: Point(row["d"], row["e"]), axis=1)
    
    # Create GeoDataFrame with CRS EPSG:5514 (S-JTSK for Czech Republic)
    gdf = geopandas.GeoDataFrame(merged_df, geometry="geometry", crs="EPSG:5514")
    
    # Filter GeoDataFrame to include only points within Czech Republic's bounding box
    czech_bbox = {"min_x": -950000, "max_x": -400000, "min_y": -1250000, "max_y": -900000}
    gdf = gdf.cx[czech_bbox["min_x"]:czech_bbox["max_x"], 
                 czech_bbox["min_y"]:czech_bbox["max_y"]]

    return gdf

#%%
def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None,
             show_figure: bool = False):
    """
    Plots two maps showing accidents involving alcohol in April and September for Prague region.

    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame with accident data.
        fig_location (str): File path to save the figure.
        show_figure (bool): Whether to display the figure.
    """
    # Filter data for Prague region (bounding box)
    praha_gdf = gdf.cx[-810000: -660000, -1100000: -100000].copy()
    
    # Extract month information from the 'date' column
    praha_gdf['month'] = praha_gdf['date'].dt.month
    
    # Filter accidents where alcohol influence is significant (p11 >= 4)
    alcohol_incidents = praha_gdf[praha_gdf['p11'] >= 4]
    
    # Separate data for April and September
    april, september = 4, 9
    april_accidents = alcohol_incidents[alcohol_incidents['month'] == april]
    september_accidents = alcohol_incidents[alcohol_incidents['month'] == september]
    
    # Create subplots for April and September
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    axes = axes.ravel()
    
    # Plot accidents for April
    april_accidents.plot(ax=axes[0], color='red', markersize=5, alpha=0.6)
    contextily.add_basemap(axes[0], crs=praha_gdf.crs.to_string(), source=contextily.providers.OpenStreetMap.Mapnik)
    axes[0].set_title('Accidents due to alcohol in April in Prague')
    
    # Plot accidents for September
    september_accidents.plot(ax=axes[1], color='blue', markersize=5, alpha=0.6)
    contextily.add_basemap(axes[1], crs=praha_gdf.crs.to_string(), source=contextily.providers.OpenStreetMap.Mapnik)
    axes[1].set_title('Accidents due to alcohol in September in Prague')
    
    # Hide axis details for cleaner visuals
    for ax in axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_xlim(praha_gdf.total_bounds[0], praha_gdf.total_bounds[2])
        ax.set_ylim(praha_gdf.total_bounds[1], praha_gdf.total_bounds[3])
    
    plt.tight_layout()
    
    # Save the figure if a file path is provided
    if fig_location:
        plt.savefig(fig_location, dpi=300)
    
    # Display the figure if requested
    if show_figure:
        plt.show()

#%%
def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """
    Plots accidents caused by animals in the South Moravian region, highlighting clusters.

    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame with accident data.
        fig_location (str): File path to save the figure.
        show_figure (bool): Whether to display the figure.
    """
    # Filter accidents for South Moravian region (JHM) caused by animals (p10 == 4)
    brno_gdf = gdf.cx[-660000: -520000, -1500000: -1100000].copy()
    animal_accidents = brno_gdf[(brno_gdf['p10'] == 4) & (brno_gdf['region'] == 'JHM')].copy()
    
    # Extract coordinates for clustering
    coords = animal_accidents.geometry.apply(lambda geom: (geom.x, geom.y)).to_list()
    coords = pd.DataFrame(coords, columns=["x", "y"])
    
    # Apply KMeans clustering on coordinates
    kmeans = sklearn.cluster.KMeans().fit(coords)
    animal_accidents['cluster'] = kmeans.labels_
    
    # Count the number of points in each cluster
    cluster_counts = animal_accidents['cluster'].value_counts()
    
    # Create polygons for each cluster (convex hulls)
    clusters = animal_accidents.dissolve(by='cluster')
    clusters['geometry'] = clusters.convex_hull
    cluster_polygons = geopandas.GeoDataFrame(clusters, geometry='geometry', crs=animal_accidents.crs)
    cluster_polygons['density'] = cluster_polygons.index.map(cluster_counts)
    
    # Normalize densities for color mapping
    norm = plt.Normalize(vmin=cluster_counts.min(), vmax=cluster_counts.max())
    cmap = plt.cm.Blues
    
    # Plot the clusters and accidents
    fig, ax = plt.subplots(figsize=(14, 7))
    cluster_polygons.plot(ax=ax, color=[cmap(norm(d)) for d in cluster_polygons['density']],
                          edgecolor='black', alpha=0.6)
    animal_accidents.plot(ax=ax, color='red', markersize=5, alpha=0.6)
    contextily.add_basemap(ax, crs=animal_accidents.crs.to_string(), source=contextily.providers.OpenStreetMap.Mapnik)
    
    # Customize plot
    ax.set_title('Accidents Due to Animals in JHM (Clusters Highlighted)')
    ax.axis('off')
    plt.tight_layout()

    # Add color bar for cluster density
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Number of Accidents in Cluster')

    # Save or display the figure
    if fig_location:
        plt.savefig(fig_location, dpi=300)
    if show_figure:
        plt.show()
    
    plt.close(fig)

#%%
if __name__ == "__main__":
    # Load data
    df_accidents = pd.read_pickle("accidents.pkl.gz")
    df_locations = pd.read_pickle("locations.pkl.gz")
    
    # Create GeoDataFrame
    gdf = make_geo(df_accidents, df_locations)
    
    # Generate plots
    plot_geo(gdf, "geo1.png", True)
    plot_cluster(gdf, "geo2.png", True)
