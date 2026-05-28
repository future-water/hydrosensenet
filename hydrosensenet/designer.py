"""High-level sensor network design API.

This file supports BOTH workflows from one class:

  - The original Texas-Gulf notebooks (baseline / flexible / risk), which build
    the designer from an UNSPLIT streamflow matrix:
        SensorNetworkDesigner(streamflow_data=..., locations=..., location_labels=...)
    and read result.performance_metrics and call result.plot_comparison(flowlines_gdf=...).

  - The GloFAS notebook (design_glofas_brazil_v2), which builds it from data that
    is ALREADY split, uses regional placement, reads result.eval_results, and lets
    plot_comparison auto-detect a gridded background.

Blocks added purely for backward compatibility with the original notebooks are
marked with "# --- compat ---" so they are easy to find.
"""

import time
from pathlib import Path
from typing import Union, List, Optional, Dict

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from hydrosensenet.data.loaders import load_streamflow_data, prepare_gauge_locations
from hydrosensenet.data.preprocessors import split_timeseries, prepare_matrix
from hydrosensenet.core.algorithms import sensor_placement_qr, qr_pivot_selection
from hydrosensenet.core.metrics import reconstruction_evaluation


class SensorNetworkDesigner:
    """High-level API for designing a sensor network from streamflow data."""

    def __init__(
        self,
        streamflow_data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        locations: gpd.GeoDataFrame = None,
        location_labels: Optional[List[str]] = None,
        train_frac: float = 0.7,
        filter_invalid: bool = True,
        *,
        X_train: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
    ):
        """
        Two ways to provide data:

          - streamflow_data : an UNSPLIT matrix; it is split here using
            train_frac (this is what the original Texas-Gulf notebooks use).
          - X_train, X_test : data that is ALREADY split (the GloFAS workflow).

        `locations` has one row per data column, in the same order as the
        columns of X_train / X_test.
        """
        if X_train is not None and X_test is not None:
            self.streamflow_data = None                       # already split
        elif streamflow_data is not None:
            # --- compat --- split here so the original notebooks can pass an
            # unsplit matrix exactly as before.
            X_train, X_test = split_timeseries(
                streamflow_data, train_frac=train_frac, filter_invalid=filter_invalid
            )
            if hasattr(X_train, "values"):
                X_train, X_test = X_train.values, X_test.values
            self.streamflow_data = streamflow_data            # baseline reads .shape on this
        else:
            raise ValueError("provide either streamflow_data or both X_train and X_test")

        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError(
                f"X_train has {X_train.shape[1]} columns, X_test has {X_test.shape[1]}"
            )
        if len(locations) != X_train.shape[1]:
            raise ValueError(
                f"locations has {len(locations)} rows but data has {X_train.shape[1]} columns"
            )

        self.X_train = X_train
        self.X_test = X_test
        self.locations = locations
        self.location_labels = location_labels or [str(i) for i in range(X_train.shape[1])]

    # ----- convenience constructors -----

    @classmethod
    def from_split(
        cls,
        X_train: np.ndarray,
        X_test: np.ndarray,
        locations: gpd.GeoDataFrame,
        location_labels: Optional[List[str]] = None,
    ) -> "SensorNetworkDesigner":
        """Build from data that is already split (explicit alias for the GloFAS path)."""
        return cls(X_train=X_train, X_test=X_test,
                   locations=locations, location_labels=location_labels)

    @classmethod
    def from_streamflow(
        cls,
        streamflow_data: Union[np.ndarray, pd.DataFrame],
        locations: gpd.GeoDataFrame,
        train_frac: float = 0.7,
        filter_invalid: bool = True,
        location_labels: Optional[List[str]] = None,
    ) -> "SensorNetworkDesigner":
        """Build a designer from an unsplit streamflow matrix."""
        return cls(streamflow_data=streamflow_data, locations=locations,
                   location_labels=location_labels,
                   train_frac=train_frac, filter_invalid=filter_invalid)

    @classmethod
    def from_files(
        cls,
        streamflow_file: Union[str, Path, List],
        locations_file: Union[str, Path],
        time_col: Optional[str] = None,
        train_frac: float = 0.7,
        **kwargs,
    ) -> "SensorNetworkDesigner":
        """Build a designer from streamflow + gauge-location files."""
        streamflow = load_streamflow_data(streamflow_file, time_col=time_col, **kwargs)
        locations = prepare_gauge_locations(locations_file)

        if isinstance(streamflow, pd.DataFrame):
            matrix, labels = streamflow.values, list(streamflow.columns)
        else:
            matrix, labels = prepare_matrix(streamflow)

        return cls(streamflow_data=matrix, locations=locations,
                   location_labels=labels, train_frac=train_frac)

    @classmethod
    def from_nwm_download(
        cls,
        streamflow_file: Union[str, Path],
        locations_geojson: Union[str, Path],
        id_col: str = "comid",
        train_frac: float = 0.7,
    ) -> "SensorNetworkDesigner":
        """
        Build from an NWM-style streamflow DataFrame (COMID columns) plus a
        GeoJSON with flowline geometries.

        Original LineString geometries are preserved on `self.locations`
        (used by plot_comparison for reach-based visuals). Centroid
        latitude/longitude columns are added for convenience.
        """
        streamflow = load_streamflow_data(streamflow_file)
        if not isinstance(streamflow, pd.DataFrame):
            raise ValueError("expected a DataFrame from streamflow_file")

        streamflow_comids = {int(c) for c in streamflow.columns}

        # Keep LineString geometry, add centroid columns alongside
        locations = gpd.read_file(locations_geojson)
        locations = locations[locations[id_col].isin(streamflow_comids)].copy()
        locations = locations.sort_values(id_col).reset_index(drop=True)

        centroids = locations.to_crs("EPSG:5070").geometry.centroid.to_crs("EPSG:4326")
        locations["latitude"] = centroids.y.values
        locations["longitude"] = centroids.x.values

        # Align streamflow columns to (now sorted) location row order
        comid_order = [str(c) for c in locations[id_col].values]
        streamflow.columns = streamflow.columns.astype(str)
        streamflow = streamflow[comid_order]

        return cls(streamflow_data=streamflow.values, locations=locations,
                   location_labels=comid_order, train_frac=train_frac)

    # ----- main API -----

    def design_network(
        self,
        n_sensors: Optional[int] = None,
        *,
        weights: Optional[Union[np.ndarray, str, Path]] = None,
        weight_column: Optional[str] = None,        # --- compat --- used only with file-path weights
        existing_sensors: Optional[List[int]] = None,
        region_column: Optional[str] = None,
        sensors_per_region: Optional[Dict[str, int]] = None,
        evaluate: bool = True,
        verbose: bool = True,
    ) -> "NetworkDesignResult":
        """
        Design an optimal sensor network.

        Three modes (mutually exclusive):
          - existing_sensors with len == n_sensors : skip QR, just evaluate.
          - region_column + sensors_per_region     : regional QR.
          - n_sensors                              : global QR.

        weights : per-location importance weights (length n_loc), OR a file
            path / column name pair (handled via calculate_spatial_weights).
        existing_sensors with len < n_sensors are treated as fixed in QR.
        """
        use_regional = region_column is not None
        if use_regional:
            if sensors_per_region is None:
                raise ValueError("sensors_per_region required with region_column")
            if region_column not in self.locations.columns:
                raise ValueError(f"'{region_column}' not in self.locations.columns")
            if existing_sensors is not None:
                raise ValueError("existing_sensors not supported with regional selection")
            n_sensors = sum(sensors_per_region.values())
        elif n_sensors is None:
            raise ValueError("n_sensors required when region_column is not given")

        # --- compat --- allow weights to be a file path + column (original feature).
        if isinstance(weights, (str, Path)):
            from hydrosensenet.spatial import calculate_spatial_weights
            weights = calculate_spatial_weights(
                self.locations, weights, weight_column=weight_column
            )
        if weights is not None:
            weights = np.asarray(weights)
            if len(weights) != self.X_train.shape[1]:
                raise ValueError(
                    f"weights length {len(weights)} != n_locations {self.X_train.shape[1]}"
                )

        if verbose:
            mode = "regional" if use_regional else "global"
            print(f"\nDesigning sensor network: {n_sensors} sensors, {mode} selection")
            print(f"  Data: train {self.X_train.shape}, test {self.X_test.shape}")

        # ----- select -----
        t0 = time.perf_counter()
        if existing_sensors is not None and len(existing_sensors) == n_sensors:
            if verbose:
                print(f"  Using provided {n_sensors} sensors (skipping QR)")
            selected_indices = np.asarray(existing_sensors)

        elif use_regional:
            region_assignments = pd.DataFrame({
                region_column: self.locations[region_column].values,
                "col_pos": np.arange(len(self.locations)),
            })
            _, selected_indices = qr_pivot_selection(
                self.X_train, region_assignments, sensors_per_region,
                region_column=region_column,
            )

        else:
            selected_indices = sensor_placement_qr(
                self.X_train, n_sensors=n_sensors,
                weights=weights, fixed_indices=existing_sensors,
                verbose=verbose,
            )

        if verbose:
            print(f"  Selection done in {time.perf_counter() - t0:.1f}s "
                  f"({len(selected_indices)} sensors)")

        # ----- evaluate -----
        eval_results = None
        if evaluate:
            t0 = time.perf_counter()
            eval_results = reconstruction_evaluation(
                self.X_train, self.X_test, selected_indices, n_sensors,
                verbose=verbose,
            )
            if verbose:
                print(f"  Evaluation done in {time.perf_counter() - t0:.1f}s "
                      f"(relative error: {eval_results['relative_error']:.4f})")

        return NetworkDesignResult(
            selected_indices=selected_indices,
            n_sensors=n_sensors,
            locations=self.locations.iloc[selected_indices].copy(),
            location_labels=[self.location_labels[i] for i in selected_indices],
            eval_results=eval_results,
            weights=weights,
            region_column=region_column if use_regional else None,
            sensors_per_region=sensors_per_region if use_regional else None,
            all_locations=self.locations,
        )


class NetworkDesignResult:
    """Results from a sensor network design run."""

    def __init__(
        self,
        selected_indices: np.ndarray,
        n_sensors: int,
        locations: gpd.GeoDataFrame,
        location_labels: List[str],
        eval_results: Optional[Dict],
        weights: Optional[np.ndarray] = None,
        region_column: Optional[str] = None,
        sensors_per_region: Optional[Dict[str, int]] = None,
        all_locations: Optional[gpd.GeoDataFrame] = None,
    ):
        self.selected_indices = selected_indices
        self.n_sensors = n_sensors
        self.locations = locations
        self.location_labels = location_labels
        self.eval_results = eval_results
        self.weights = weights
        self.region_column = region_column
        self.sensors_per_region = sensors_per_region
        self.all_locations = all_locations

        # --- compat --- the original notebooks read result.performance_metrics['nnse'].
        # Expose the same dict shape the original produced (r2 is None, as it was).
        self.performance_metrics = (
            {
                "r2": None,
                "nse": eval_results["nse"],
                "nnse": eval_results["nnse"],
                "eval_results": eval_results,
            }
            if eval_results is not None else None
        )

    # ----- summary -----

    def print_summary(self):
        """Print a short text summary."""
        print("\nSensor network design")
        print(f"  Sensors selected: {self.n_sensors}")

        if self.region_column and self.sensors_per_region:
            print(f"  Regional allocation ({self.region_column}):")
            for region, count in self.sensors_per_region.items():
                print(f"    {region}: {count}")

        preview = list(self.selected_indices[:10])
        suffix = " ..." if len(self.selected_indices) > 10 else ""
        print(f"  Indices: {preview}{suffix}")

        if self.eval_results is not None:
            nse  = self.eval_results["nse"]
            nnse = self.eval_results["nnse"]
            print("\n  Performance (median across locations):")
            print(f"    NSE:  {np.nanmedian(nse):.3f}")
            print(f"    NNSE: {np.nanmedian(nnse):.3f}")
            print(f"    Relative error: {self.eval_results['relative_error']:.4f}")

    # ----- export -----

    def get_dataframe(self) -> pd.DataFrame:
        """Return a flat DataFrame with selected sensors + metrics."""
        df = self.locations.drop(columns="geometry").reset_index(drop=True)

        # Ensure latitude/longitude columns (derive from geometry if missing)
        if "latitude" not in df.columns or "longitude" not in df.columns:
            lats, lons = self._point_coords(self.locations)
            df["latitude"], df["longitude"] = lats, lons

        df.insert(0, "sensor_rank", np.arange(1, len(df) + 1))
        df["location_index"] = self.selected_indices
        df["location_label"] = self.location_labels

        if self.eval_results is not None:
            df["nse"]  = self.eval_results["nse"][self.selected_indices]
            df["nnse"] = self.eval_results["nnse"][self.selected_indices]
            df["rmse"] = self.eval_results["rmse"][self.selected_indices]
        return df

    def export(
        self,
        output_path: Union[str, Path],
        format: str = "auto",
        include_metrics: bool = True,
    ):
        """Save results to CSV / shapefile / GeoJSON / GeoPackage / Excel."""
        output_path = Path(output_path)
        if format == "auto":
            format = {
                ".csv": "csv", ".shp": "shp", ".geojson": "geojson",
                ".gpkg": "gpkg", ".xlsx": "xlsx",
            }.get(output_path.suffix.lower(), "csv")

        if format in ("csv", "xlsx"):
            df = self.get_dataframe()
            (df.to_excel if format == "xlsx" else df.to_csv)(output_path, index=False)
        else:
            # Geometry-aware formats: keep the GeoDataFrame, attach metrics inline
            gdf = self.locations.copy()
            gdf["sensor_rank"] = np.arange(1, len(gdf) + 1)
            if include_metrics and self.eval_results is not None:
                gdf["nse"]  = self.eval_results["nse"][self.selected_indices]
                gdf["nnse"] = self.eval_results["nnse"][self.selected_indices]
                gdf["rmse"] = self.eval_results["rmse"][self.selected_indices]
            driver = {"shp": "ESRI Shapefile", "geojson": "GeoJSON", "gpkg": "GPKG"}[format]
            gdf.to_file(output_path, driver=driver)

        print(f"Results exported to: {output_path}")

    # ----- helpers (also used by plot_*) -----

    @staticmethod
    def _point_coords(gdf: gpd.GeoDataFrame):
        """Return (lats, lons). Uses centroids for non-Point geometries."""
        if gdf.geometry.geom_type.iloc[0] == "Point":
            return gdf.geometry.y.to_numpy(), gdf.geometry.x.to_numpy()
        # Centroids in an equal-area projection for accuracy, then back
        centroids = gdf.to_crs("EPSG:5070").geometry.centroid.to_crs(gdf.crs)
        return centroids.y.to_numpy(), centroids.x.to_numpy()

    # ----- plotting -----

    def plot_sensors(
        self,
        save_path: Optional[str] = None,
        figsize: tuple = (8, 6),
        dpi: int = 200,
        color: str = "green",
        label: Optional[str] = None,
    ):
        """Simple map showing selected sensor locations."""
        fig, ax = self._setup_axes(figsize, dpi)
        lats, lons = self._point_coords(self.locations)
        ax.set_extent(self._calculate_extent(lons, lats), crs=ccrs.PlateCarree())

        ax.scatter(
            lons, lats,
            color=color, edgecolors="white", linewidth=0.5, s=15,
            label=label or f"Selected sensors (n={len(lats)})",
            transform=ccrs.PlateCarree(),
        )
        self._add_map_features(ax)
        ax.legend(loc="upper right", frameon=True, fontsize=9, framealpha=0.9)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=dpi, transparent=True)
            print(f"Figure saved to: {save_path}")
        return fig, ax

    def plot_comparison(
        self,
        baseline_result: "NetworkDesignResult",
        metric: str = "nnse",
        network_type: str = "auto",
        save_path: Optional[str] = None,
        figsize: tuple = (8, 6),
        dpi: int = 200,
        metric_range: tuple = (-0.5, 0.5),
        # --- compat --- accepted so the original notebooks' calls don't error.
        # flowlines_gdf can also supply the reach geometry if it isn't already
        # carried on self.all_locations.
        flowlines_gdf: Optional[gpd.GeoDataFrame] = None,
        location_labels: Optional[List[str]] = None,
        comparison_labels: Optional[Dict[str, str]] = None,
        comid_column: Optional[str] = None,
    ):
        """
        Plot this design against a baseline, with a metric-difference background.

        network_type
        ------------
        'grid'    : pcolormesh of (this - baseline) — for gridded data (GloFAS)
        'reaches' : colored flowlines of the same — for stream-reach data (USGS/NWM)
        'points'  : no background, just sensor markers — for scattered points
        'auto'    : LineString geometries -> reaches;
                    Point on a uniform lat/lon grid -> grid;
                    otherwise -> points
        """
        if self.eval_results is None or baseline_result.eval_results is None:
            raise ValueError("both designs must have eval_results (evaluate=True)")

        # --- compat --- fall back to a passed-in flowlines layer if needed.
        background_gdf = self.all_locations if self.all_locations is not None else flowlines_gdf
        if background_gdf is None:
            raise ValueError("need self.all_locations or flowlines_gdf for the background")

        diff = self.eval_results[metric] - baseline_result.eval_results[metric]

        if network_type == "auto":
            network_type = self._detect_network_type(background_gdf)

        fig, ax = self._setup_axes(figsize, dpi)

        # Extent from both sensor sets
        opt_lats,  opt_lons  = self._point_coords(self.locations)
        base_lats, base_lons = self._point_coords(baseline_result.locations)
        ax.set_extent(
            self._calculate_extent(
                np.concatenate([opt_lons, base_lons]),
                np.concatenate([opt_lats, base_lats]),
            ),
            crs=ccrs.PlateCarree(),
        )

        # Background (or none)
        artist = None
        if network_type == "grid":
            artist = self._draw_grid_background(ax, background_gdf, diff, metric_range)
        elif network_type == "reaches":
            artist = self._draw_reach_background(ax, background_gdf, diff, metric_range)

        if artist is not None:
            cbar = plt.colorbar(artist, ax=ax, fraction=0.04, pad=0.02)
            cbar.set_label(f"Δ {metric.upper()}")

        # Sensor overlays (honor comparison_labels if the original notebooks passed them)
        labels = comparison_labels or {}
        opt_label  = labels.get("optimal", "Reconfigured sensors")
        base_label = labels.get("baseline", f"Baseline (n={len(base_lats)})")
        ax.scatter(opt_lons,  opt_lats,
                   color="green", edgecolors="white", linewidth=0.1,
                   alpha=0.8, s=7, label=opt_label,
                   transform=ccrs.PlateCarree())
        ax.scatter(base_lons, base_lats,
                   color="k", edgecolors="white", linewidth=0.1,
                   alpha=0.8, s=7, label=base_label,
                   transform=ccrs.PlateCarree())

        self._add_map_features(ax)
        ax.legend(loc="upper right", frameon=True, fontsize=9, framealpha=0.9)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=dpi, transparent=True)
            print(f"Figure saved to: {save_path}")
        return fig, ax

    # ----- plotting helpers -----

    def _detect_network_type(self, gdf: gpd.GeoDataFrame) -> str:
        """Pick between 'grid', 'reaches', 'points' from geometry."""
        geom_type = gdf.geometry.geom_type.iloc[0]
        if geom_type in ("LineString", "MultiLineString"):
            return "reaches"
        if geom_type == "Point":
            return "grid" if self._is_regular_grid(gdf) else "points"
        return "points"

    @staticmethod
    def _is_regular_grid(gdf: gpd.GeoDataFrame, tol: float = 0.1) -> bool:
        """Are these Points on a roughly uniform lat/lon grid?"""
        if gdf.geometry.geom_type.iloc[0] != "Point":
            return False
        u_lats = np.unique(gdf.geometry.y.to_numpy())
        u_lons = np.unique(gdf.geometry.x.to_numpy())
        if len(u_lats) < 2 or len(u_lons) < 2:
            return False
        lat_d, lon_d = np.diff(u_lats), np.diff(u_lons)
        cv_lat = lat_d.std() / lat_d.mean() if lat_d.mean() > 0 else np.inf
        cv_lon = lon_d.std() / lon_d.mean() if lon_d.mean() > 0 else np.inf
        return cv_lat < tol and cv_lon < tol

    def _draw_grid_background(self, ax, gdf: gpd.GeoDataFrame, diff: np.ndarray, metric_range: tuple):
        """pcolormesh for gridded data. Missing cells stay transparent (NaN)."""
        lons = gdf.geometry.x.to_numpy()
        lats = gdf.geometry.y.to_numpy()
        u_lons = np.sort(np.unique(lons))
        u_lats = np.sort(np.unique(lats))

        Z = np.full((len(u_lats), len(u_lons)), np.nan)
        Z[np.searchsorted(u_lats, lats), np.searchsorted(u_lons, lons)] = diff

        norm = TwoSlopeNorm(vmin=metric_range[0], vcenter=0, vmax=metric_range[1])
        return ax.pcolormesh(
            u_lons, u_lats, Z,
            cmap="bwr_r", norm=norm, shading="auto", alpha=0.8,
            transform=ccrs.PlateCarree(),
        )

    def _draw_reach_background(self, ax, gdf: gpd.GeoDataFrame, diff: np.ndarray, metric_range: tuple):
        """LineCollection for reach-based data."""
        segments, values = [], []
        for geom, d in zip(gdf.geometry, diff):
            if geom.geom_type == "LineString":
                segments.append(np.array(geom.coords))
                values.append(d)
            elif geom.geom_type == "MultiLineString":
                # Split into individual parts, replicating the metric value
                for part in geom.geoms:
                    segments.append(np.array(part.coords))
                    values.append(d)

        lines = LineCollection(segments, linewidths=1.0, alpha=0.8, zorder=1,
                               transform=ccrs.PlateCarree())
        lines.set_array(np.asarray(values))
        lines.set_cmap("bwr_r")
        lines.set_norm(Normalize(vmin=metric_range[0], vmax=metric_range[1]))
        ax.add_collection(lines)
        return lines

    @staticmethod
    def _setup_axes(figsize, dpi):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.axes(projection=ccrs.PlateCarree())
        return fig, ax

    @staticmethod
    def _add_map_features(ax):
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                          alpha=0.3, linestyle="--")
        ax.add_feature(cfeature.STATES, linewidth=0.4, linestyle=':', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

    @staticmethod
    def _calculate_extent(lons, lats, padding=0.05):
        lon_range = lons.max() - lons.min()
        lat_range = lats.max() - lats.min()
        return [
            lons.min() - lon_range * padding,
            lons.max() + lon_range * padding,
            lats.min() - lat_range * padding,
            lats.max() + lat_range * padding,
        ]