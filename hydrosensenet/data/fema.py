"""FEMA National Risk Index (NRI) loader.

Download the GeoDatabase from
https://www.fema.gov/about/openfema/data-sets/national-risk-index-data
and pass the path to :func:`load_nri`.
"""

from pathlib import Path
from typing import Iterable, Optional, Union

import geopandas as gpd


NRI_LAYERS = {
    "tract": "NRI_CensusTracts",
    "county": "NRI_Counties",
}


def load_nri(
    path: Union[str, Path],
    scale: str = "tract",
    columns: Optional[Iterable[str]] = ("RFLD_RISKS",),
    mask: Optional[gpd.GeoDataFrame] = None,
) -> gpd.GeoDataFrame:
    """Load a FEMA NRI layer, optionally clipped to ``mask``.

    ``scale`` is ``"tract"`` or ``"county"``. ``columns`` selects which
    hazard columns to keep (e.g. ``RFLD_RISKS``, ``CFLD_RISKS``,
    ``HRCN_RISKS``); pass ``None`` to keep all.
    """
    if scale not in NRI_LAYERS:
        raise ValueError(f"scale must be one of {list(NRI_LAYERS)}, got {scale!r}")

    gdf = gpd.read_file(path, layer=NRI_LAYERS[scale])

    if columns is not None:
        cols = list(columns)
        missing = [c for c in cols if c not in gdf.columns]
        if missing:
            raise ValueError(
                f"Columns not in NRI layer: {missing}. "
                f"Available: {sorted(gdf.columns)}"
            )
        gdf = gdf[cols + ["geometry"]]

    if mask is not None:
        gdf = gpd.clip(gdf, mask.to_crs(gdf.crs))

    return gdf
