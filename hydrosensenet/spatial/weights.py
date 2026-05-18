"""Spatial weighting for risk-informed sensor placement."""

from pathlib import Path
from typing import Iterable, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd


def _location_ids(locations_gdf, id_column):
    if id_column is not None:
        if id_column not in locations_gdf.columns:
            raise ValueError(
                f"id_column {id_column!r} not in locations_gdf. "
                f"Available: {list(locations_gdf.columns)}"
            )
        return pd.Index(locations_gdf[id_column])
    if "gauge_id" in locations_gdf.columns:
        return pd.Index(locations_gdf["gauge_id"])
    return pd.Index(locations_gdf.index)


def calculate_spatial_weights(
    locations_gdf: gpd.GeoDataFrame,
    weight_source: Union[str, Path, gpd.GeoDataFrame, np.ndarray, dict],
    weight_column: Optional[str] = None,
    id_column: Optional[str] = None,
    align_to: Optional[Iterable] = None,
    fill_value: float = 0.0,
    aggregation: str = "mean",
    normalize: bool = False,
) -> np.ndarray:
    """Compute a per-location weight vector for ``sensor_placement_qr``.

    Output is aligned to ``align_to`` (e.g. the streamflow matrix's
    column order) when given, else to ``locations_gdf`` row order.

    For the FEMA flood-risk workflow set ``fill_value=1e-10`` so
    uncovered locations stay selectable by QR pivoting (``0.0`` removes
    them entirely). ``normalize`` is off because min-max scaling would
    map the minimum to ``0`` and have the same effect.
    """
    if isinstance(weight_source, np.ndarray):
        if len(weight_source) != len(locations_gdf):
            raise ValueError(
                f"Weight array length ({len(weight_source)}) must match "
                f"locations ({len(locations_gdf)})"
            )
        row_weights = weight_source.astype(float).copy()

    elif isinstance(weight_source, dict):
        ids = _location_ids(locations_gdf, id_column)
        row_weights = np.array(
            [weight_source.get(k, fill_value) for k in ids], dtype=float
        )

    else:
        if isinstance(weight_source, (str, Path)):
            weight_gdf = gpd.read_file(weight_source)
        elif isinstance(weight_source, gpd.GeoDataFrame):
            weight_gdf = weight_source
        else:
            raise TypeError(
                "weight_source must be str, Path, GeoDataFrame, ndarray, "
                f"or dict; got {type(weight_source).__name__}"
            )
        if weight_column is None:
            raise ValueError("weight_column must be specified for GeoDataFrame input")
        if weight_column not in weight_gdf.columns:
            raise ValueError(
                f"Column {weight_column!r} not in weight data. "
                f"Available: {list(weight_gdf.columns)}"
            )

        weight_gdf = weight_gdf.to_crs(locations_gdf.crs)
        joined = gpd.sjoin(
            locations_gdf[["geometry"]],
            weight_gdf[[weight_column, "geometry"]],
            how="left",
            predicate="intersects",
        )
        grouped = joined.groupby(joined.index)[weight_column]
        aggregators = {
            "mean": grouped.mean,
            "max": grouped.max,
            "sum": grouped.sum,
            "min": grouped.min,
        }
        if aggregation not in aggregators:
            raise ValueError(
                f"Unknown aggregation {aggregation!r}. "
                f"Choose from {list(aggregators)}"
            )
        row_weights = (
            aggregators[aggregation]()
            .reindex(locations_gdf.index)
            .to_numpy(dtype=float)
        )

    if align_to is not None:
        ids = _location_ids(locations_gdf, id_column)
        by_id = pd.Series(row_weights, index=ids).groupby(level=0).agg(aggregation)
        weights = by_id.reindex(list(align_to)).to_numpy(dtype=float)
    else:
        weights = row_weights

    weights = np.where(np.isnan(weights), fill_value, weights)

    if normalize:
        lo, hi = weights.min(), weights.max()
        weights = (weights - lo) / (hi - lo) if hi > lo else np.ones_like(weights)

    return weights
