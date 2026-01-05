"""Spatial weighting calculations for risk-informed sensor placement."""

import numpy as np
import geopandas as gpd
from typing import Union, Optional
from pathlib import Path


def calculate_spatial_weights(
    locations_gdf: gpd.GeoDataFrame,
    weight_source: Union[str, Path, gpd.GeoDataFrame, np.ndarray, dict],
    weight_column: Optional[str] = None,
    aggregation: str = "mean",
    normalize: bool = True
) -> np.ndarray:
    """Calculate spatial weights for sensor placement."""
    # Handle array input
    if isinstance(weight_source, np.ndarray):
        if len(weight_source) != len(locations_gdf):
            raise ValueError(
                f"Weight array length ({len(weight_source)}) must match "
                f"locations ({len(locations_gdf)})"
            )
        weights = weight_source.copy()

    # Handle dict input
    elif isinstance(weight_source, dict):
        # Assume locations_gdf has an ID column
        if "gauge_id" in locations_gdf.columns:
            id_values = locations_gdf["gauge_id"]
        else:
            id_values = locations_gdf.index
        weights = np.array([weight_source.get(loc_id, 0.0) for loc_id in id_values])

    # Handle GeoDataFrame or file input
    else:
        # Load if it's a file path
        if isinstance(weight_source, (str, Path)):
            weight_gdf = gpd.read_file(weight_source)
        elif isinstance(weight_gdf, gpd.GeoDataFrame):
            weight_gdf = weight_source
        else:
            raise TypeError(
                f"weight_source must be str, Path, GeoDataFrame, array, or dict, "
                f"got {type(weight_source)}"
            )

        if weight_column is None:
            raise ValueError("weight_column must be specified for GeoDataFrame input")

        if weight_column not in weight_gdf.columns:
            raise ValueError(
                f"Column '{weight_column}' not found in weight data. "
                f"Available: {list(weight_gdf.columns)}"
            )

        # Spatial join
        joined = gpd.sjoin(
            locations_gdf,
            weight_gdf[[weight_column, "geometry"]],
            how="left",
            predicate="intersects"
        )

        # Aggregate if multiple intersections
        if aggregation == "mean":
            weights = joined.groupby(joined.index)[weight_column].mean()
        elif aggregation == "max":
            weights = joined.groupby(joined.index)[weight_column].max()
        elif aggregation == "sum":
            weights = joined.groupby(joined.index)[weight_column].sum()
        elif aggregation == "min":
            weights = joined.groupby(joined.index)[weight_column].min()
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        # Fill missing values
        weights = weights.reindex(locations_gdf.index, fill_value=0.0).values

    # Handle NaN values
    weights = np.where(np.isnan(weights), 0.0, weights)

    # Normalize if requested
    if normalize:
        min_val = weights.min()
        max_val = weights.max()
        if max_val > min_val:
            weights = (weights - min_val) / (max_val - min_val)
        else:
            weights = np.ones_like(weights)

    return weights


def load_risk_data(
    risk_file: Union[str, Path],
    layer: Optional[str] = None,
    risk_column: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    Load risk/priority data from file.

    Parameters
    ----------
    risk_file : str or Path
        Path to shapefile, GeoPackage, or other vector file.
    layer : str, optional
        Layer name (for multi-layer files like GeoPackage).
    risk_column : str, optional
        Column name with risk values (auto-detected if None).

    Returns
    -------
    risk_gdf : gpd.GeoDataFrame
        GeoDataFrame with risk data.

    Examples
    --------
    >>> # FEMA flood risk data
    >>> risk = load_risk_data("NRI_CensusTracts.gdb", layer="NRI_CensusTracts")

    >>> # Custom shapefile
    >>> risk = load_risk_data("my_risk_data.shp", risk_column="priority")
    """
    if layer:
        risk_gdf = gpd.read_file(risk_file, layer=layer)
    else:
        risk_gdf = gpd.read_file(risk_file)

    # Auto-detect risk column if not specified
    if risk_column is None:
        # Look for common risk column names
        risk_candidates = [
            'risk', 'priority', 'score', 'value', 'weight',
            'flood_risk', 'RFLD_RISKS', 'hazard'
        ]
        for candidate in risk_candidates:
            matching = [col for col in risk_gdf.columns if candidate.lower() in col.lower()]
            if matching:
                risk_column = matching[0]
                print(f"Auto-detected risk column: {risk_column}")
                break

    return risk_gdf
