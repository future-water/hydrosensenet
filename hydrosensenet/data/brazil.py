"""Region-specific downloaders for gauge networks, watersheds, and country boundaries."""

import shutil
import zipfile
from pathlib import Path
from typing import Union, Optional

import requests
import geopandas as gpd


# =========================================================================
# Generic helpers
# =========================================================================

def _download_zip(url: str, zip_path: Path) -> Path:
    """Stream a zip file from `url` to disk. Raises on HTTP failure."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return zip_path


def _already_exists(path: Path, verbose: bool) -> bool:
    """Report whether the final file already exists."""
    if path.exists():
        if verbose:
            print(f"Already exists: {path}")
        return True
    return False


# =========================================================================
# Brazil gauges (CAMELS-BR)
# =========================================================================

BRAZIL_GAUGES_URL = (
    "https://zenodo.org/records/15025488/files/"
    "13_CAMELS_BR_gauge_location.zip?download=1"
)
BRAZIL_GAUGES_INNER_PATH = "13_CAMELS_BR_gauge_location/location_gauges_streamflow.gpkg"


def download_brazil_gauges(
    output_dir: Union[str, Path],
    zip_file: str = "brazil_gauges.zip",
    gpkg_file: str = "brazil_gauges.gpkg",
    force_download: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Download CAMELS-BR gauge locations from Zenodo.

    Source: https://zenodo.org/records/15025488
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gpkg_path = output_dir / gpkg_file

    if _already_exists(gpkg_path, verbose) and not force_download:
        return gpkg_path

    if verbose:
        print("Downloading Brazil stream gauges from Zenodo ...")
    zip_path = _download_zip(BRAZIL_GAUGES_URL, output_dir / zip_file)

    # Extract one specific file from inside the zip
    with zipfile.ZipFile(zip_path, "r") as z, \
         z.open(BRAZIL_GAUGES_INNER_PATH) as src, \
         open(gpkg_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

    if verbose:
        print(f"Extracted: {gpkg_path}")
    return gpkg_path


# =========================================================================
# Brazil watersheds (ANA)
# =========================================================================

BRAZIL_WATERSHEDS_URL = (
    "https://metadados.snirh.gov.br/geonetwork/srv/api/records/"
    "0574947a-2c5b-48d2-96a4-b07c4702bbab/attachments/"
    "SNIRH_RegioesHidrograficas_2020.zip"
)


def download_brazil_watersheds(
    output_dir: Union[str, Path],
    zip_file: str = "brazil_watersheds.zip",
    shp_file: str = "brazil_watersheds.shp",
    force_download: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Download Brazil hydrographic regions from the National Water Agency (ANA).

    Source: https://metadados.snirh.gov.br/geonetwork/srv/api/records/
            0574947a-2c5b-48d2-96a4-b07c4702bbab
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shp_path = output_dir / shp_file

    if _already_exists(shp_path, verbose) and not force_download:
        return shp_path

    if verbose:
        print("Downloading Brazil watersheds from ANA ...")
    zip_path = _download_zip(BRAZIL_WATERSHEDS_URL, output_dir / zip_file)

    # Source CRS is EPSG:4674 (SIRGAS 2000); convert to WGS84 for consistency
    gdf = gpd.read_file(f"zip://{zip_path}").set_crs("EPSG:4674").to_crs("EPSG:4326")
    gdf.to_file(shp_path, driver="ESRI Shapefile")
    if verbose:
        print(f"Saved: {shp_path}")
    return shp_path


# =========================================================================
# Country boundary (generic, via GADM)
# =========================================================================

GADM_URL_TEMPLATE = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_{iso3}_shp.zip"


def download_country_boundary(
    output_dir: Union[str, Path],
    iso3: str,
    zip_file: Optional[str] = None,
    shp_file: Optional[str] = None,
    force_download: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Download a country boundary from GADM, given an ISO 3166-1 alpha-3 code.

    Examples
    --------
    >>> download_country_boundary("./data", iso3="BRA")
    >>> download_country_boundary("./data", iso3="USA")

    Source: https://gadm.org/download_country.html
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_file = zip_file or f"gadm41_{iso3}.zip"
    shp_file = shp_file or f"gadm41_{iso3}.shp"
    shp_path = output_dir / shp_file

    if _already_exists(shp_path, verbose) and not force_download:
        return shp_path

    url = GADM_URL_TEMPLATE.format(iso3=iso3)
    if verbose:
        print(f"Downloading {iso3} boundary from GADM ...")
    zip_path = _download_zip(url, output_dir / zip_file)

    gdf = gpd.read_file(f"zip://{zip_path}", layer=0)
    gdf.to_file(shp_path, driver="ESRI Shapefile")
    if verbose:
        print(f"Saved: {shp_path}")
    return shp_path