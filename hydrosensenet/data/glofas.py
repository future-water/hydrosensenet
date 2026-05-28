"""Global Flood Awareness System (GloFAS) downloader.

Wraps the Copernicus Climate Data Store (CDS) API for the
`cems-glofas-historical` dataset.
"""

import time
from pathlib import Path
from typing import Union, List, Optional, Dict

import pandas as pd


class GloFASDataLoader:
    """
    Download GloFAS historical data from Copernicus CDS.

    GloFAS provides global river discharge on a 0.05 deg (~5 km) grid.

    Examples
    --------
    >>> loader = GloFASDataLoader()
    >>> loader.set_region(bounds=[5.5, -74, -34, -34], name="brazil")
    >>> files = loader.download_streamflow(
    ...     start_date="2020-01-01",
    ...     end_date="2022-12-31",
    ...     output_dir="./data/brazil",
    ...     download_by="year",
    ... )
    """

    DATASET = "cems-glofas-historical"

    DEFAULT_CONFIG = {
        "system_version": ["version_4_0"],     # also: "version_3_1", "version_2_1"
        "hydrological_model": ["lisflood"],    # also: "htessel_lisflood"
        "product_type": ["consolidated"],      # also: "intermediate"
        "variable": ["river_discharge_in_the_last_24_hours"],
        "data_format": "grib2",                # also: "netcdf"
        "download_format": "unarchived",       # also: "zip"
    }

    def __init__(self, cds_quiet: bool = True, verbose: bool = True):
        """
        Parameters
        ----------
        cds_quiet : suppress the cdsapi client's own output
        verbose : print this class's progress messages
        """
        self.cds_quiet = cds_quiet
        self.verbose = verbose
        self.bounds = None
        self.region_name = None
        self.downloaded_files: List[Path] = []
        self._setup_cds_client()

    def _setup_cds_client(self):
        # Imported here (not at module top) so that `import hydrosensenet.data`
        # does not require cdsapi for users who only use NWM/USGS/GloFAS-free paths.
        try:
            import cdsapi
        except ImportError as e:
            raise ImportError(
                "cdsapi is required for GloFAS downloads. Install it with "
                "`pip install cdsapi`."
            ) from e
        try:
            self.cds_client = cdsapi.Client(quiet=self.cds_quiet)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize CDS client: {e}\n"
                "Make sure you have:\n"
                "  1. Registered at https://cds.climate.copernicus.eu\n"
                "  2. Created ~/.cdsapirc with your API key\n"
                "  3. Installed cdsapi (pip install cdsapi)"
            )

    # ---------- region ----------

    def set_region(self, bounds: List[float], name: Optional[str] = None):
        """
        Set the region of interest.

        bounds : [North, West, South, East] in decimal degrees
        name : identifier used for output filenames
        """
        if len(bounds) != 4:
            raise ValueError("bounds must be [North, West, South, East]")
        north, west, south, east = bounds
        if not (-90 <= south < north <= 90):
            raise ValueError(f"invalid latitude bounds: south={south}, north={north}")
        if not (-180 <= west <= 180 and -180 <= east <= 180):
            raise ValueError(f"invalid longitude bounds: west={west}, east={east}")

        self.bounds = bounds
        self.region_name = name or f"region_{north}_{west}_{south}_{east}"

        if self.verbose:
            area = abs(north - south) * abs(east - west)
            print(f"Region '{self.region_name}': "
                  f"N={north}, W={west}, S={south}, E={east} (~{area:.1f} deg^2)")

    # ---------- download ----------

    def download_streamflow(
        self,
        start_date: str,
        end_date: str,
        output_dir: Union[str, Path],
        download_by: str = "year",
        custom_config: Optional[Dict] = None,
    ) -> List[Path]:
        """
        Download GloFAS streamflow for the configured region and time period.

        download_by : 'year', 'month', or 'all' — splits the request into
            that many files. CDS request-size limits favour 'year' for
            ranges longer than a few months.
        custom_config : override any DEFAULT_CONFIG keys.
        """
        if self.bounds is None:
            raise ValueError("Region not set. Call set_region() first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        config = {**self.DEFAULT_CONFIG, **(custom_config or {})}
        periods = self._generate_periods(start_date, end_date, download_by)

        if self.verbose:
            print(f"Downloading {self.region_name}: "
                  f"{start_date} to {end_date} as {len(periods)} {download_by}-file(s)")

        files = []
        for i, (p_start, p_end, label) in enumerate(periods, 1):
            if self.verbose:
                print(f"\n[{i}/{len(periods)}] {label} "
                      f"({p_start.date()} to {p_end.date()})")
            files.append(self._download_period(p_start, p_end, output_dir, label, config))

        self.downloaded_files = files
        if self.verbose:
            print(f"\nDone: {len(files)} file(s) in {output_dir}")
        return files

    @staticmethod
    def _generate_periods(start_date: str, end_date: str, download_by: str):
        """Split a date range into download periods."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        if download_by == "year":
            return [
                (max(start, pd.Timestamp(f"{y}-01-01")),
                 min(end,  pd.Timestamp(f"{y}-12-31")),
                 str(y))
                for y in range(start.year, end.year + 1)
            ]
        if download_by == "month":
            periods, current = [], start
            while current <= end:
                month_end = current + pd.offsets.MonthEnd(0)
                periods.append((current, min(end, month_end), current.strftime("%Y-%m")))
                current = month_end + pd.Timedelta(days=1)
            return periods
        if download_by == "all":
            label = f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
            return [(start, end, label)]
        raise ValueError(f"download_by must be 'year', 'month', or 'all'; got '{download_by}'")

    def _download_period(self, start, end, output_dir, label, config) -> Path:
        """Make one CDS request covering [start, end]."""
        dates = pd.date_range(start, end, freq="D")
        request = {
            **config,
            "hyear":  [str(y) for y in sorted(set(dates.year))],
            "hmonth": [f"{m:02d}" for m in sorted(set(dates.month))],
            "hday":   [f"{d:02d}" for d in sorted(set(dates.day))],
            "area": self.bounds,
        }

        ext = ".nc" if config.get("data_format") == "netcdf" else ".grib"
        output_file = output_dir / f"{self.region_name}_streamflow_{label}{ext}"

        t0 = time.perf_counter()
        try:
            self.cds_client.retrieve(self.DATASET, request).download(str(output_file))
        except Exception as e:
            print(f"  Download failed for {label}: {e}")
            raise

        if self.verbose:
            size_mb = output_file.stat().st_size / 1e6
            print(f"  -> {output_file.name} "
                  f"({size_mb:.1f} MB, {time.perf_counter() - t0:.1f}s)")
        return output_file

    # ---------- metadata ----------

    @classmethod
    def info(cls) -> Dict:
        """Reference info about the GloFAS dataset."""
        return {
            "name": "Global Flood Awareness System (GloFAS)",
            "provider": "Copernicus Emergency Management Service",
            "spatial_resolution": "0.05 deg (~5 km)",
            "temporal_resolution": "Daily",
            "coverage": "Global",
            "time_period": "1979-present",
            "primary_variable": "River discharge (m^3/s)",
            "hydrological_models": ["LISFLOOD", "HTESSEL"],
            "versions": ["4.0", "3.1", "2.1"],
            "data_format": ["GRIB2", "NetCDF"],
            "cds_dataset": cls.DATASET,
            "documentation": "https://confluence.ecmwf.int/display/CEMS/GloFAS",
        }