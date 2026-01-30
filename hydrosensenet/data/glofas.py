"""Global Flood Awareness System (GloFAS) data utilities.

This module provides functions to download, process, and filter ?? data
for sensor network optimization workflows.
"""

import pandas as pd
import cdsapi
from typing import Union, List, Optional, Dict
from pathlib import Path
import time

class GLOFASDataLoader:
    """
    Load and process GloFAS historical data from Copernicus CDS

    GloFAS provides global river discharge data on a 0.05° (~5km) grid.
    
    Examples
    --------
    >>> # Download data for a region
    >>> loader = GloFASDataLoader()
    >>> loader.set_region(
    ...     bounds=[5.5, -74, -34, -34],  # Brazil: [North, West, South, East]
    ...     name="brazil"
    ... )
    >>> df = loader.download_streamflow(
    ...     start_date="2020-01-01",
    ...     end_date="2022-12-31",
    ...     output_dir="./data/brazil"
    ... )
    """
    
    # GloFAS dataset configuration
    DATASET = "cems-glofas-historical"

    # Available options
    DEFAULT_CONFIG = {
        "system_version": ["version_4_0"], # ← Can use "version_3_1", "version_2_1"
        "hydrological_model": ["lisflood"], # ← Can use "htessel_lisflood"
        "product_type": ["consolidated"], # ← Can use "intermediate"
        "variable": ["river_discharge_in_the_last_24_hours"], # ← Can use "runoff_water_equivalent", "snow_depth_water_equivalent", "soil_wetness_index"
        "data_format": "grib2", # ← Can use "netcdf"
        "download_format": "unarchived" # ← Can use "zip" 
    }
    def __init__(self, cds_quiet: bool = True):
        """
        Initialize GloFAS data loader.

        Parameters
        ----------
        cds_quiet : bool, default=True
            Suppress CDS API output.
        """
        self.cds_quiet = cds_quiet
        
        # Region configuration
        self.bounds = None
        self.region_name = None
        
        # Data storage
        self.downloaded_files = []

        # Initialize CDS client
        self._setup_cds_client()

    def _setup_cds_client(self):
        """Initialize Copernicus CDS API client."""
        try:
            self.cds_client = cdsapi.Client(quiet=self.cds_quiet)
            if not self.cds_quiet:
                print("✓ CDS API client initialized")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize CDS client: {e}\n"
                "Make sure you have:\n"
                "  1. Registered at https://cds.climate.copernicus.eu\n"
                "  2. Created ~/.cdsapirc with your API key\n"
                "  3. Installed cdsapi: pip install cdsapi"
            )
    
    def set_region(
        self,
        bounds: List[float],
        name: Optional[str] = None
    ) -> "GloFASDataLoader":
        """
        Set geographic region for data download.

        Parameters
        ----------
        bounds : list of float
            Geographic bounds as [North, West, South, East] in decimal degrees.
            Example: [5.5, -74, -34, -34] for Brazil
        name : str, optional
            Name identifier for the region (used in file naming).

        Returns
        -------
        self
            Returns self for method chaining.
        """
        if len(bounds) != 4:
            raise ValueError("Bounds must be [North, West, South, East]")
        
        north, west, south, east = bounds
        
        # Validate bounds
        if not (-90 <= south < north <= 90):
            raise ValueError(f"Invalid latitude bounds: South={south}, North={north}")
        if not (-180 <= west <= 180 and -180 <= east <= 180):
            raise ValueError(f"Invalid longitude bounds: West={west}, East={east}")
        
        self.bounds = bounds
        self.region_name = name or f"region_{north}_{west}_{south}_{east}"
        
        print(f"✓ Region set: {self.region_name}")
        print(f"  Bounds: North={north}°, West={west}°, South={south}°, East={east}°")
        area_approx = abs(north - south) * abs(east - west)
        print(f"  Approximate area: {area_approx:.1f} degree²")
        
        return self
    
    def download_streamflow(
        self,
        start_date: str,
        end_date: str,
        output_dir: Union[str, Path],
        download_by: str = "year",
        custom_config: Optional[Dict] = None
    ) -> List[Path]:
        """
        Download GloFAS streamflow data for the specified region and time period.

        Parameters
        ----------
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        end_date : str
            End date in 'YYYY-MM-DD' format.
        output_dir : str or Path
            Directory to save downloaded files.
        download_by : str, default='year'
            Download granularity: 'year', 'month', or 'all'.
            - 'year': Download each year separately
            - 'month': Download each month separately
            - 'all': Download entire date range in one file
        custom_config : dict, optional
            Override DEFAULT_CONFIG parameters. Use this to customize:
            - system_version: e.g., ["version_4_0", "version_3_1"]
            - hydrological_model: e.g., ["lisflood", "htessel_lisflood"]
            - product_type: e.g., ["consolidated", "intermediate"]
            - variable: e.g., ["river_discharge_in_the_last_24_hours", "snow_depth_water_equivalent"]
            - data_format: "grib2" or "netcdf"
            - download_format: "unarchived" or "zip"

        Returns
        -------
        list of Path
            Paths to downloaded GRIB2 files.
        """
        if self.bounds is None:
            raise ValueError("Region not set. Use set_region() first.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Merge custom config with defaults
        config = self.DEFAULT_CONFIG.copy()
        if custom_config:
            config.update(custom_config)
            print(f"✓ Using custom configuration:")
            for key, val in custom_config.items():
                print(f"  {key}: {val}")

        # Generate download periods
        periods = self._generate_download_periods(start, end, download_by)
        
        print(f"\n{'='*60}")
        print(f"Downloading GloFAS data for {self.region_name}")
        print(f"{'='*60}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Download strategy: by {download_by} ({len(periods)} request(s))")
        print(f"Output directory: {output_dir}")
        
        downloaded_files = []
        
        for i, (period_start, period_end, label) in enumerate(periods, 1):
            print(f"\n{'-'*60}")
            print(f"Period {i}/{len(periods)}: {label}")
            print(f"{'-'*60}")
            
            file_path = self._download_period(
                period_start, 
                period_end, 
                output_dir, 
                label,
                config
            )
            downloaded_files.append(file_path)
        
        self.downloaded_files = downloaded_files
        
        print(f"\n{'='*60}")
        print(f"✓ All downloads complete!")
        print(f"{'='*60}")
        print(f"Files saved to: {output_dir}")
        print(f"Total files: {len(downloaded_files)}")
        
        return downloaded_files
    
    def _generate_download_periods(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        download_by: str
    ) -> List[tuple]:
        """Generate list of download periods."""
        periods = []
        
        if download_by == "year":
            for year in range(start.year, end.year + 1):
                period_start = max(start, pd.Timestamp(f"{year}-01-01"))
                period_end = min(end, pd.Timestamp(f"{year}-12-31"))
                periods.append((period_start, period_end, str(year)))
        
        elif download_by == "month":
            current = start
            while current <= end:
                # End of month or end date, whichever is earlier
                month_end = current + pd.offsets.MonthEnd(0)
                period_end = min(end, month_end)
                label = current.strftime("%Y-%m")
                periods.append((current, period_end, label))
                # Move to first day of next month
                current = month_end + pd.Timedelta(days=1)
        
        elif download_by == "all":
            label = f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
            periods.append((start, end, label))
        
        else:
            raise ValueError(f"Invalid download_by: {download_by}. Use 'year', 'month', or 'all'")
        
        return periods
    
    def _download_period(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        output_dir: Path,
        label: str,
        config: Dict
    ) -> Path:
        """Download data for a single period."""
        # Generate all dates in the period
        date_range = pd.date_range(start=start, end=end, freq='D')
        
        # Extract unique years, months, and days
        years = [str(y) for y in sorted(set(date_range.year))]
        months = [f"{m:02d}" for m in sorted(set(date_range.month))]
        days = [f"{d:02d}" for d in sorted(set(date_range.day))]
        
        # Build API request
        request = {
            **config,
            "hyear": years,
            "hmonth": months,
            "hday": days,
            "area": self.bounds
        }
        
        # Determine file extension based on data format
        file_ext = ".nc" if config.get("data_format") == "netcdf" else ".grib"
        output_file = output_dir / f"{self.region_name}_streamflow_{label}{file_ext}"
        
        print(f"  Requesting data from CDS...")
        print(f"  Date range: {start.date()} to {end.date()} ({len(date_range)} days)")
        print(f"  Years: {years}")
        print(f"  Months: {months}")
        print(f"  Days: {days}")
        print(f"  Area: {self.bounds}")

        t_start = time.perf_counter()
        
        try:
            self.cds_client.retrieve(
                self.DATASET,
                request
            ).download(str(output_file))
            
            t_end = time.perf_counter()
            
            # Get file size
            file_size_mb = output_file.stat().st_size / 1e6
            
            print(f"  ✓ Download complete ({t_end - t_start:.2f}s)")
            print(f"  File: {output_file.name}")
            print(f"  Size: {file_size_mb:.1f} MB")
            
        except Exception as e:
            print(f"  ✗ Download failed: {e}")
            raise
        
        return output_file

    def download_multiple_years(
        self,
        years: List[int],
        output_dir: Union[str, Path] = ".",
        custom_config: Optional[Dict] = None
    ) -> List[Path]:
        """
        Download multiple years of data, saving each year separately.

        Parameters
        ----------
        years : list of int
            List of years to download.
        output_dir : str or Path, default='.'
            Directory to save downloaded files.
        custom_config : dict, optional
            Custom configuration to override defaults.

        Returns
        -------
        list of Path
            Paths to downloaded files.
        """
        if self.bounds is None:
            raise ValueError("Region not set. Use set_region() first.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []

        for i, year in enumerate(years, 1):
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"

            print(f"\n{'='*60}")
            print(f"Year {i}/{len(years)}: {year}")
            print(f"{'='*60}")

            files = self.download_streamflow(
                start_date=start_date,
                end_date=end_date,
                output_dir=output_dir,
                download_by="year",
                custom_config=custom_config
            )

            saved_files.extend(files)

        print(f"\n{'='*60}")
        print(f"✓ All {len(years)} years downloaded!")
        print(f"{'='*60}")
        print(f"Files saved to: {output_dir}")

        return saved_files


def get_glofas_info() -> dict:
    """
    Get information about the GloFAS dataset.

    Returns
    -------
    dict
        Dataset information including resolution, coverage, and variables.
    """
    return {
        'name': 'Global Flood Awareness System (GloFAS)',
        'provider': 'Copernicus Emergency Management Service',
        'spatial_resolution': '0.05° (~5km)',
        'temporal_resolution': 'Daily',
        'coverage': 'Global',
        'time_period': '1979-present',
        'primary_variable': 'River discharge (m³/s)',
        'hydrological_models': ['LISFLOOD', 'HTESSEL'],
        'versions': ['4.0', '3.1'],
        'data_format': ['GRIB2', 'NetCDF'],
        'cds_dataset': 'cems-glofas-historical',
        'documentation': 'https://confluence.ecmwf.int/display/CEMS/GloFAS'
    }