# geo.py
This script processes and visualizes geographic accident data using `GeoPandas`, `matplotlib`, and `contextily`. 

### Key Functions:
1. **make_geo(df_accidents, df_locations)**:
   - Merges accident and location datasets based on a shared identifier (`p1`).
   - Cleans invalid coordinates and converts them into a `GeoDataFrame` with CRS `EPSG:5514`.
   - Filters points to ensure they are within the Czech Republic's bounding box.

2. **plot_geo(gdf, fig_location=None, show_figure=False)**:
   - Filters data for Prague and selects accidents where alcohol was a factor.
   - Plots accidents occurring in April and September using red/blue markers on a map.

3. **plot_cluster(gdf, fig_location=None, show_figure=False)**:
   - Focuses on accidents caused by animals in the South Moravian region.
   - Uses KMeans clustering to identify accident hotspots.
   - Visualizes accident clusters with convex hulls and color-coded densities.

4. **Main Execution**:
   - Loads accident and location data from compressed pickle files.
   - Calls the `make_geo` function to create a `GeoDataFrame`.
   - Generates and saves maps for alcohol-related accidents and accident clusters.

---

# doc.py
This script analyzes traffic accident data and generates tables, visualizations, and LaTeX reports.

### Key Functions:
1. **find_most_common_number_by_category(df, categories)**:
   - Identifies the most frequent cause of accidents in each category.
   - Computes the percentage of each leading cause.

2. **create_table(df)**:
   - Summarizes accident counts by reason for 2023 and 2024.

3. **table_to_tex(df, stream=sys.stdout)**:
   - Converts the accident summary table into a LaTeX-formatted table with dividers.

4. **map_reason(value)**:
   - Maps accident cause codes (`p12`) to descriptive reason labels.

5. **get_dataframe()**:
   - Loads accident data from a compressed pickle file and maps reasons.

6. **plot_graphs(dataframe, fig_location, show_figure)**:
   - Creates bar plots and pie charts showing accident distribution by region and reason.
   - Uses `seaborn` for improved visualization.

7. **Main Execution**:
   - Loads data, processes accident reasons, and generates plots.
   - Saves plots to `fig.pdf`.
   - Converts and prints accident summary tables in LaTeX format.

---

# doc.pdf
A report summarizing accident statistics in the Czech Republic.

# stat.ipynb
This Jupyter Notebook explores traffic accident data using Python's data science ecosystem, including `pandas`, `seaborn`, and `matplotlib`.
