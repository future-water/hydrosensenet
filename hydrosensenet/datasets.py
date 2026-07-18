"""Bundled example datasets.

Small NWM-derived samples committed to the GitHub repository, so
tutorials and quick experiments run without the full NWM download
stack. In a repository checkout (or editable install) the files are
read directly; installed packages download them once from GitHub and
cache them under ``~/.cache/hydrosensenet``.
"""
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Union

import geopandas as gpd
import pandas as pd

_BASE_URL = (
    "https://raw.githubusercontent.com/future-water/hydrosensenet/main/example_data"
)

# filename -> sha256 of the committed file
_FILES = {
    "streamflow": (
        "texas_gulf_sample_streamflow.parquet",
        "977eaaac26995d8a6c3bfe881f9a8e3cbaeb974062f92b33408b493ffac9cdba",
    ),
    "locations": (
        "texas_gulf_sample_locations.geojson",
        "91873bb65c286cff1a940e6007f52bec74f544991f994ce97264f651e4286b5d",
    ),
}


def _repo_example_dir() -> Optional[Path]:
    """Return the checkout's example_data directory, if present."""
    candidate = Path(__file__).resolve().parent.parent / "example_data"
    return candidate if candidate.is_dir() else None


def _default_cache_dir() -> Path:
    d = Path.home() / ".cache" / "hydrosensenet"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fetch(name: str, cache_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolve a bundled data file locally, downloading it if needed."""
    filename, expected_sha = _FILES[name]

    repo_dir = _repo_example_dir()
    if repo_dir is not None and (repo_dir / filename).exists():
        return repo_dir / filename

    target_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename

    if not target.exists():
        import requests

        url = f"{_BASE_URL}/{filename}"
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        target.write_bytes(response.content)

    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != expected_sha:
        target.unlink()
        raise ValueError(
            f"Checksum mismatch for {filename} (got {digest[:12]}..., "
            f"expected {expected_sha[:12]}...). The cached file was removed; "
            f"try again, and if the error persists file an issue."
        )
    return target


def load_example_basin(
    cache_dir: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Load the bundled Texas-Gulf NWM sample basin.

    A small subset of the National Water Model v3.0 retrospective run:
    two years (2020-2021) of daily mean streamflow for 368 reaches of
    the lower Colorado River, Texas (HUC8 12090302, between Columbus
    and the Gulf coast), with reach centroid locations. Suitable for
    tutorials and quick experiments; use
    :class:`~hydrosensenet.data.nwm.NWMDataLoader` for full-scale
    studies.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory for the downloaded copy when running outside a
        repository checkout. Defaults to ``~/.cache/hydrosensenet``.

    Returns
    -------
    streamflow : pd.DataFrame
        Daily streamflow (m^3/s), one column per reach COMID, indexed
        by time.
    locations : gpd.GeoDataFrame
        Reach centroid points (EPSG:4326) with a ``comid`` column, one
        row per streamflow column, in the same order.
    """
    streamflow = pd.read_parquet(_fetch("streamflow", cache_dir))
    locations = gpd.read_file(_fetch("locations", cache_dir))

    # Align defensively: locations rows follow streamflow column order
    order = {str(c): i for i, c in enumerate(streamflow.columns)}
    locations = (
        locations.assign(_pos=locations["comid"].astype(str).map(order))
        .sort_values("_pos")
        .drop(columns="_pos")
        .reset_index(drop=True)
    )
    return streamflow, locations
