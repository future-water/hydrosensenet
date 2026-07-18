"""USGS gauge network data utilities."""

import requests
from pathlib import Path
from typing import Union, Optional, Tuple
import geopandas as gpd
import pandas as pd
import zipfile
import io
import warnings


def download_usgs_gauges(
    output_dir: Union[str, Path],
    huc2: Optional[str] = None,
    force_download: bool = False,
    verbose: bool = True
) -> Tuple[Path, Path]:
    """
    Download USGS gauge network data with NHDPlus COMID linkage.

    Parameters
    ----------
    output_dir : str or Path
        Directory to save the downloaded data
    huc2 : str, optional
        HUC2 region code to filter gauges (e.g., '12' for Texas-Gulf)
    force_download : bool, default=False
        If True, download even if file exists
    verbose : bool, default=True
        Print progress messages

    Returns
    -------
    tuple of Path
        (shapefile_path, comid_linkage_path)
        - shapefile_path: Path to the gauge locations GeoJSON
        - comid_linkage_path: Path to the USGS-COMID linkage CSV

    Notes
    -----
    Downloads USGS Streamgage NHDPlus Basins dataset from ScienceBase:
    - 12,422 streamgages across conterminous US
    - Pre-computed NHDPlusV2 COMID linkages
    - Station locations and attributes

    Reference:
    Wieczorek, M.E., et al., 2018, USGS Streamgage NHDPlus Version 1 Basins 2011
    https://www.sciencebase.gov/catalog/item/5ebe92af82ce476925e44b8f
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output filenames
    if huc2:
        geojson_file = output_dir / f"usgs_gauges_huc{huc2}.geojson"
        comid_file = output_dir / f"usgs_comid_linkage_huc{huc2}.csv"
    else:
        geojson_file = output_dir / "usgs_gauges_all.geojson"
        comid_file = output_dir / "usgs_comid_linkage_all.csv"

    # Check if files already exist
    if geojson_file.exists() and comid_file.exists() and not force_download:
        if verbose:
            print(f"✓ USGS gauge data already exists:")
            print(f"  Locations: {geojson_file}")
            print(f"  COMID linkage: {comid_file}")
            gdf = gpd.read_file(geojson_file)
            print(f"  {len(gdf):,} gauges loaded")
        return geojson_file, comid_file

    if verbose:
        print(f"Downloading USGS Streamgage NHDPlus Basins dataset...")
        print(f"  Source: USGS ScienceBase")

    try:
        # Download USGS Streamgage NHDPlus Basins shapefile (includes COMID!)
        # Use ScienceBase REST API to get the file
        item_id = "5ebe92af82ce476925e44b8f"

        if verbose:
            print(f"  Fetching file information from ScienceBase API...")

        # Get item metadata to find download URL
        api_url = f"https://www.sciencebase.gov/catalog/item/{item_id}?format=json"
        metadata_response = requests.get(api_url, timeout=60)
        metadata_response.raise_for_status()
        metadata = metadata_response.json()

        # Files are nested in 'facets' array in ScienceBase
        # Find shapefile components in facets
        shapefile_urls = {}
        files_found = []

        for facet in metadata.get('facets', []):
            for file_item in facet.get('files', []):
                file_name = file_item.get('name', '')
                files_found.append(file_name)

                # Collect all shapefile components
                if file_name.startswith('SWIM_gage_loc'):
                    file_url = file_item.get('url') or file_item.get('downloadUri')
                    if file_url:
                        shapefile_urls[file_name] = file_url
                        if verbose:
                            print(f"    Found: {file_name}")

        if not shapefile_urls:
            if verbose:
                print(f"  Available files: {files_found}")
            raise ValueError(f"Could not find SWIM shapefile components. Available files: {files_found}")

        if verbose:
            print(f"  Downloading USGS-NHDPlus linkage shapefile...")
            print(f"  This includes 12,422 gauges with COMIDs already linked")

        # Download all shapefile components to temp directory
        temp_dir = output_dir / "temp_swim"
        temp_dir.mkdir(exist_ok=True)

        if verbose:
            print(f"  Downloading {len(shapefile_urls)} shapefile components...")

        for file_name, file_url in shapefile_urls.items():
            if verbose:
                print(f"    {file_name}...")
            response = requests.get(file_url, timeout=300)
            response.raise_for_status()

            # Save to temp directory
            file_path = temp_dir / file_name
            file_path.write_bytes(response.content)

        # Load shapefile
        shp_file = temp_dir / "SWIM_gage_loc.shp"
        if not shp_file.exists():
            raise ValueError(f"Shapefile not found after download: {shp_file}")

        if verbose:
            print(f"  Loading shapefile...")
        gdf = gpd.read_file(shp_file)

        # Clean up temp directory (will do this after processing)
        # Moved to after saving output files

        if verbose:
            print(f"  ✓ Loaded {len(gdf):,} USGS gauges with COMID linkage")
            # Check what columns are available
            if 'COMID' in gdf.columns or 'FLComID' in gdf.columns:
                comid_col = 'COMID' if 'COMID' in gdf.columns else 'FLComID'
                print(f"    COMID column found: '{comid_col}'")

        # No need to download separate basin characteristics - COMIDs are in the shapefile!
        df_basins = None

        # Filter by HUC2 if specified
        if huc2:
            # Look for HUC columns (HUC02, HUC_02, HUC2, etc.)
            huc_cols = [col for col in gdf.columns if 'huc' in col.lower()]

            if huc_cols:
                huc_col = huc_cols[0]
                gdf[huc_col] = gdf[huc_col].astype(str).str.zfill(2)
                gdf_filtered = gdf[gdf[huc_col] == huc2].copy()

                if verbose:
                    print(f"  ✓ Filtered to HUC2 {huc2}: {len(gdf_filtered):,} gauges")

                # Filter basins too if available
                if df_basins is not None and 'STAID' in df_basins.columns:
                    station_col = 'STAID'
                    if station_col in gdf_filtered.columns:
                        valid_stations = gdf_filtered[station_col].values
                        df_basins = df_basins[df_basins[station_col].isin(valid_stations)].copy()

                gdf = gdf_filtered.reset_index(drop=True)  # Reset index for proper iteration
            else:
                if verbose:
                    print(f"  Warning: No HUC column found, cannot filter by HUC2")

        # Extract COMID linkage from shapefile attributes
        comid_linkage = None

        # Find COMID column in the shapefile
        comid_cols = [col for col in gdf.columns if 'comid' in col.lower() and col.lower() != 'reachcode']

        # Find station ID column
        station_col = None
        for col in ['SITE_NO', 'STAID', 'GAGE_ID', 'SOURCE_FEA', 'SITENO', 'Gage_no']:
            if col in gdf.columns:
                station_col = col
                break

        if comid_cols and station_col:
            comid_col = comid_cols[0]

            # Create linkage table directly from shapefile
            comid_linkage = gdf[[station_col, comid_col]].copy()
            comid_linkage.columns = ['STAID', 'COMID']

            # Clean up COMID values (convert to int)
            comid_linkage['COMID'] = pd.to_numeric(comid_linkage['COMID'], errors='coerce')
            comid_linkage = comid_linkage.dropna(subset=['COMID'])
            comid_linkage['COMID'] = comid_linkage['COMID'].astype(int)

            # Remove invalid COMIDs (e.g., -9999 which indicates no linkage)
            comid_linkage = comid_linkage[comid_linkage['COMID'] > 0].copy()

            # Keep STAID as string to preserve leading zeros
            comid_linkage['STAID'] = comid_linkage['STAID'].astype(str)

            if verbose:
                print(f"  ✓ Extracted {len(comid_linkage):,} USGS-COMID linkages from shapefile")
        else:
            if verbose:
                print(f"  Warning: Could not find COMID or station ID columns")
                print(f"  Available columns: {list(gdf.columns[:20])}...")


        # Save files
        gdf.to_file(geojson_file, driver="GeoJSON")

        if comid_linkage is not None and len(comid_linkage) > 0:
            comid_linkage.to_csv(comid_file, index=False)
            if verbose:
                print(f"✓ Saved USGS gauge data:")
                print(f"  Locations: {geojson_file} ({len(gdf):,} gauges)")
                print(f"  COMID linkage: {comid_file} ({len(comid_linkage):,} linked)")
        else:
            # Create empty linkage file
            pd.DataFrame(columns=['STAID', 'COMID']).to_csv(comid_file, index=False)
            if verbose:
                print(f"✓ Saved USGS gauge data:")
                print(f"  Locations: {geojson_file} ({len(gdf):,} gauges)")
                print(f"  Warning: No COMID linkage available")
                print(f"  You may need to manually create COMID linkage")

        # Clean up temp directory
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        return geojson_file, comid_file

    except Exception as e:
        if verbose:
            print(f"Error downloading USGS gauge data: {e}")
            print(f"\nAlternative: Manually download from:")
            print(f"  https://www.sciencebase.gov/catalog/item/5ebe92af82ce476925e44b8f")
        raise


def load_usgs_gauges(
    shapefile_path: Union[str, Path],
    comid_linkage_path: Optional[Union[str, Path]] = None,
    huc2: Optional[str] = None,
    comids: Optional[list] = None
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Load USGS gauge network data with COMID linkage.

    Parameters
    ----------
    shapefile_path : str or Path
        Path to USGS gauge GeoJSON/Shapefile
    comid_linkage_path : str or Path, optional
        Path to USGS-COMID linkage CSV file
    huc2 : str, optional
        Filter to specific HUC2 region
    comids : list, optional
        Filter to specific NHDPlus COMIDs

    Returns
    -------
    tuple
        (gauges_gdf, comid_linkage_df)
        - gauges_gdf: GeoDataFrame of gauge locations
        - comid_linkage_df: DataFrame with STAID-COMID pairs
    """
    # Load gauge locations
    gdf = gpd.read_file(shapefile_path)

    # Load COMID linkage if provided
    if comid_linkage_path and Path(comid_linkage_path).exists():
        # Preserve STAID as string to keep leading zeros
        comid_df = pd.read_csv(comid_linkage_path, dtype={'STAID': str, 'COMID': int})
    else:
        comid_df = pd.DataFrame(columns=['STAID', 'COMID'])

    # Filter by HUC2 if specified
    if huc2:
        huc_cols = [col for col in gdf.columns if 'huc' in col.lower()]
        if huc_cols:
            huc_col = huc_cols[0]
            gdf[huc_col] = gdf[huc_col].astype(str)
            gdf = gdf[gdf[huc_col].str.startswith(huc2)].copy()

    # Filter by COMIDs if specified
    if comids and len(comid_df) > 0:
        matched_stations = comid_df[comid_df['COMID'].isin(comids)]['STAID'].astype(str).values
        # Find station column in the GeoDataFrame
        station_col = None
        for col in ['STAID', 'SITE_NO', 'GAGE_ID', 'Gage_no']:
            if col in gdf.columns:
                station_col = col
                break
        if station_col:
            # Ensure string comparison for station IDs
            gdf = gdf[gdf[station_col].astype(str).isin(matched_stations)].copy()
        comid_df = comid_df[comid_df['COMID'].isin(comids)].copy()

    return gdf, comid_df


def match_usgs_to_nwm(
    usgs_gdf: gpd.GeoDataFrame,
    comid_linkage: pd.DataFrame,
    available_comids: list,
    verbose: bool = True
) -> Tuple[list, gpd.GeoDataFrame]:
    """
    Match USGS gauges to NWM COMIDs and return indices.

    Parameters
    ----------
    usgs_gdf : GeoDataFrame
        USGS gauge locations
    comid_linkage : DataFrame
        USGS-COMID linkage with columns ['STAID', 'COMID']
    available_comids : list
        List of COMIDs available in NWM dataset
    verbose : bool
        Print matching statistics

    Returns
    -------
    usgs_comids : list of int
        COMIDs aligned to ``matched_usgs_gdf`` rows: ``usgs_comids[i]``
        is the COMID of ``matched_usgs_gdf.iloc[i]``.
    matched_usgs_gdf : GeoDataFrame
        Matched USGS gauges with a ``comid`` column (one row per
        station; duplicate stations dropped with a warning).
    """
    if len(comid_linkage) == 0:
        if verbose:
            print("Warning: No COMID linkage available")
        return [], gpd.GeoDataFrame()

    # Filter to COMIDs that exist in both datasets
    available_comids_set = set([int(c) for c in available_comids])
    comid_linkage = comid_linkage[comid_linkage['COMID'].isin(available_comids_set)].copy()

    if len(comid_linkage) == 0:
        if verbose:
            print("Warning: No USGS gauges match available COMIDs")
        return [], gpd.GeoDataFrame()

    # Find station column in the GeoDataFrame
    station_col = None
    for col in ['STAID', 'SITE_NO', 'GAGE_ID', 'Gage_no']:
        if col in usgs_gdf.columns:
            station_col = col
            break
    if not station_col:
        if verbose:
            print(f"Warning: Could not find station ID column in GeoDataFrame")
            print(f"  Available columns: {list(usgs_gdf.columns[:10])}...")
        return [], gpd.GeoDataFrame()

    # Merge the linkage into the gauge GeoDataFrame on the station id so
    # COMIDs stay aligned with gauge rows. Compare station ids as strings
    # on both sides (shapefiles may store them as int, dropping zeros).
    gdf = usgs_gdf.copy()
    gdf['_station_key'] = gdf[station_col].astype(str)
    linkage = comid_linkage[['STAID', 'COMID']].copy()
    linkage['_station_key'] = linkage['STAID'].astype(str)

    merged = gdf.merge(
        linkage[['_station_key', 'COMID']].rename(columns={'COMID': '_matched_comid'}),
        on='_station_key',
        how='inner',
    )

    if len(merged) == 0:
        if verbose:
            print("Warning: No USGS gauges match available COMIDs")
        return [], gpd.GeoDataFrame()

    # Drop duplicate stations (e.g. one station linked to several COMIDs,
    # or repeated gauge rows) so each returned row is a unique station
    n_before = len(merged)
    merged = merged.drop_duplicates(subset='_station_key', keep='first')
    n_duplicates = n_before - len(merged)
    if n_duplicates > 0:
        warnings.warn(
            f"Dropped {n_duplicates} duplicate station match(es) when "
            f"matching USGS gauges to NWM COMIDs; kept the first COMID "
            f"per station"
        )

    merged['comid'] = merged['_matched_comid'].astype(int)
    matched_usgs_gdf = merged.drop(
        columns=['_station_key', '_matched_comid']
    ).reset_index(drop=True)
    usgs_comids = matched_usgs_gdf['comid'].tolist()

    if verbose:
        print(f"✓ Matched {len(usgs_comids)} USGS gauges to NWM COMIDs")
        print(f"  Example COMIDs: {usgs_comids[:5]}...")

    return usgs_comids, matched_usgs_gdf
