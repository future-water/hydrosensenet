"""Brazil gauge network data utilities."""

import requests
from pathlib import Path
from typing import Union
import geopandas as gpd
import pandas as pd
import os

def _download_zip(
    url: str,
    zip_path: Path
) -> Path:
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()

            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        print(f"An error occurred. Try downloading the website manually")

def download_brazil_gauges(
    output_dir: Union[str, Path],
    zip_file: str,
    gpkg_file: str,
    force_download: bool = False,
    verbose: bool = True
) -> Path:
    """
    Download stream gauge locations in Brazil
    
    Parameters
    ----------
    output_dir : str or Path
        Directory to save the downloaded data
    zip_file : str
        Name for downloaded zipfile
    gpkg_file : str
        Name for geoPackage file
    force_download : bool, default=False
        If True, download even if file exists
    verbose : bool, default=True
        Print progress messages
    Notes
    -----
    Website link: https://zenodo.org/records/3964745
    
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output filenames
    zip_path = output_dir / zip_file 
    gpkg_path = os.path.join(output_dir, gpkg_file)

    url = "https://zenodo.org/records/3964745/files/15_CAMELS_BR_gauges_location_shapefile.zip?download=1"
    
    # Check if files already exist
    if os.path.exists(gpkg_path) and not force_download:
        if verbose:
            print(f"✓ GeoPackage already exists:")
            print(f"  Locations: {gpkg_path}")
        return gpkg_path
    if verbose:
        print(f"Downloading Brazil Stream Gauges")
        print(f"Zenodo")

    # Download zip
    _download_zip(url,zip_path)
    if verbose:
        print(f"Downloaded Zipfile")

    # Unzip
    try:
        bnd_shp = gpd.read_file(f"zip://{zip_path}/15_CAMELS_BR_gauges_location_shapefile").to_crs("EPSG:4326")
        if verbose:
            print("Shapefile successfully loaded into a GeoDataFrame")
        bnd_shp.to_file(gpkg_path, driver="GPKG")
        if verbose:
            print(f"✓ GeoPackage saved locally as {gpkg_path}")
        return gpkg_path
    except Exception as e:
        if verbose:
            print(f"Error reading GeoPackage with GeoPandas: {e}")


def prepare_gauge_geodataframe() -> any:
    return False

def download_brazil_watersheds(
    output_dir: Union[str, Path],
    zip_file: str,
    shp_file: str,
    force_download: bool = False,
    verbose: bool = True
) -> Path:
    """
    Download Brazil watersheds
    
    Parameters
    ----------
    output_dir : str or Path
        Directory to save the downloaded data
    zip_file : str
        Name for downloaded zipfile
    shp_file : str
        Name for shapefile
    force_download : bool, default=False
        If True, download even if file exists
    verbose : bool, default=True
        Print progress messages

    Notes
    -----
    Website link: https://metadados.snirh.gov.br/geonetwork/srv/api/records/0574947a-2c5b-48d2-96a4-b07c4702bbab
    
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output filenames
    zip_path = os.path.join(output_dir, zip_file)
    shp_path = os.path.join(output_dir, shp_file)

    url = "https://metadados.snirh.gov.br/geonetwork/srv/api/records/0574947a-2c5b-48d2-96a4-b07c4702bbab/attachments/SNIRH_RegioesHidrograficas_2020.zip"
    
    # Check if files already exist
    if os.path.exists(shp_path) and not force_download:
        if verbose:
            print(f"✓ Shapefile already exists:")
            print(f"  Locations: {shp_path}")
        return shp_path
    if verbose:
        print(f"Downloading Brazil Watershed shapefile")
        print(f"Source: National Water Agency (ANA)")

    # Download zip
    _download_zip(url,zip_path)
    if verbose:
        print(f"Downloaded Zipfile")

    # Unzip
    try:
        bnd_shp = gpd.read_file(f"zip://{zip_path}").set_crs("EPSG:4674").to_crs("EPSG:4326")
        if verbose:
            print("Shapefile successfully loaded into a GeoDataFrame")
        bnd_shp.to_file(shp_path, driver="ESRI Shapefile")
        if verbose:
            print(f"✓ Shapefile saved locally as {shp_path}")
        return shp_path
    except Exception as e:
        if verbose:
            print(f"Error reading shapefile with GeoPandas: {e}")

def download_country_boundary(
    output_dir: Union[str, Path],
    zip_file: str,
    shp_file: str,
    iso3: str,
    force_download: bool = False,
    verbose: bool = True
) -> Path:
    """
    Download any country boundary using ISO3
    
    Parameters
    ----------
    output_dir : str or Path
        Directory to save the downloaded data
    zip_file : str
        Name for downloaded zipfile
    shp_file : str
        Name for shapefile
    iso3 : str
        ISO 3166-1 alpha-3 code for desired country
    force_download : bool, default=False
        If True, download even if file exists
    verbose : bool, default=True
        Print progress messages

    Notes
    -----
    Must have a iso3 code. A more straightforward alternative is to directly download from the website.
    
    Website Link:
    https://gadm.org/download_country.html
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_{iso3}_shp.zip"
    
    # Define output filenames
    zip_path = os.path.join(output_dir, zip_file)
    shp_path = os.path.join(output_dir, shp_file)

    # Check if files already exist
    if os.path.exists(shp_path) and not force_download:
        if verbose:
            print(f"✓ Shapefile already exists:")
            print(f"  Locations: {shp_path}")
        return shp_path
    if verbose:
        print(f"Downloading {iso3} boundary shapefile")
        print(f"Source: GADM")

    # Download zip
    _download_zip(url,zip_path)
    if verbose:
        print(f"Downloaded Zipfile")

    # Unzip
    try:
        bnd_shp = gpd.read_file(f"zip://{zip_path}",layer=0)
        if verbose:
            print("Shapefile successfully loaded into a GeoDataFrame")
        bnd_shp.to_file(shp_path, driver="ESRI Shapefile")
        if verbose:
            print(f"✓ Shapefile saved locally as {shp_path}")
        return shp_path
    except Exception as e:
        if verbose:
            print(f"Error reading shapefile with GeoPandas: {e}")