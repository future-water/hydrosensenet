"""High-level API for sensor network design - Simple interface for water managers."""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Union, List, Optional, Dict
import warnings

from .data import load_streamflow_data, prepare_gauge_locations, split_timeseries, prepare_matrix
from .core import sensor_placement_qr, calculate_performance_metrics, reconstruction_evaluation
from .spatial import calculate_spatial_weights


class SensorNetworkDesigner:
    """Easy-to-use interface for designing optimal sensor networks."""

    def __init__(
        self,
        streamflow_data: np.ndarray,
        locations: gpd.GeoDataFrame,
        location_labels: Optional[List[str]] = None
    ):
        """Initialize sensor network designer.

        Parameters
        ----------
        streamflow_data : np.ndarray
            Matrix of shape ``(n_timesteps, n_locations)``; each column
            is one candidate location's time series.
        locations : gpd.GeoDataFrame
            One row per column of ``streamflow_data``, in the same
            order.
        location_labels : list of str, optional
            Label per location (e.g. gauge IDs or COMIDs). Defaults to
            stringified column positions (or column names for DataFrame
            input).
        """
        if hasattr(streamflow_data, "values"):  # DataFrame input
            if location_labels is None:
                location_labels = [str(c) for c in streamflow_data.columns]
            streamflow_data = streamflow_data.values
        streamflow_data = np.asarray(streamflow_data)

        self.streamflow_data = streamflow_data
        self.locations = locations
        self.location_labels = location_labels or [str(i) for i in range(streamflow_data.shape[1])]

        # Validate dimensions
        if streamflow_data.shape[1] != len(locations):
            raise ValueError(
                f"Streamflow data has {streamflow_data.shape[1]} locations but "
                f"locations GeoDataFrame has {len(locations)} rows"
            )
        if len(self.location_labels) != streamflow_data.shape[1]:
            raise ValueError(
                f"Got {len(self.location_labels)} location_labels for "
                f"{streamflow_data.shape[1]} locations"
            )

    @classmethod
    def from_files(
        cls,
        streamflow_file: Union[str, Path, List[Union[str, Path]]],
        locations_file: Optional[Union[str, Path]] = None,
        time_col: Optional[str] = None,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        **streamflow_kwargs
    ):
        """Create designer from data files (CSV, Parquet, Excel, NetCDF, etc.)."""
        # Load streamflow data
        streamflow = load_streamflow_data(
            streamflow_file,
            time_col=time_col,
            **streamflow_kwargs
        )

        # Load locations
        locations = prepare_gauge_locations(
            locations_file,
            lat_col=lat_col,
            lon_col=lon_col
        )

        # Convert to matrix if needed
        if isinstance(streamflow, pd.DataFrame):
            matrix = streamflow.values
            labels = list(streamflow.columns)
        else:
            matrix, labels = prepare_matrix(streamflow)

        return cls(matrix, locations, labels)

    @classmethod
    def from_csv(
        cls,
        streamflow_file: Union[str, Path, List[Union[str, Path]]],
        locations_file: Optional[Union[str, Path]] = None,
        time_col: Optional[str] = None,
        lat_col: str = "latitude",
        lon_col: str = "longitude"
    ):
        """Load data from files (works with all formats despite name)."""
        return cls.from_files(
            streamflow_file=streamflow_file,
            locations_file=locations_file,
            time_col=time_col,
            lat_col=lat_col,
            lon_col=lon_col
        )

    @classmethod
    def from_nwm_download(
        cls,
        streamflow_file: Union[str, Path],
        locations_geojson: Union[str, Path],
        id_col: str = "comid"
    ):
        """Convenience method for NWM downloads with auto-alignment."""
        import geopandas as gpd

        # Load streamflow data (auto-detects 'time' column now)
        streamflow = load_streamflow_data(streamflow_file)

        # Get COMID columns from streamflow
        if isinstance(streamflow, pd.DataFrame):
            comids_streamflow = set([int(col) for col in streamflow.columns])
        else:
            raise ValueError("Expected DataFrame from streamflow file")

        # Load locations GeoJSON
        locations_gdf = gpd.read_file(locations_geojson)

        # Filter to only COMIDs that exist in streamflow data
        locations_filtered = locations_gdf[
            locations_gdf[id_col].isin(comids_streamflow)
        ].copy()

        # Calculate centroids for point locations
        # Reproject to projected CRS for accurate centroids
        locations_proj = locations_filtered.to_crs("EPSG:5070")  # USA Contiguous Albers Equal Area
        centroids = locations_proj.geometry.centroid.to_crs("EPSG:4326")

        locations_filtered['longitude'] = centroids.x
        locations_filtered['latitude'] = centroids.y
        locations_filtered.geometry = centroids

        # Align streamflow columns with locations
        # Sort both by COMID to ensure alignment
        locations_filtered = locations_filtered.sort_values(id_col).reset_index(drop=True)
        comid_order = [str(c) for c in locations_filtered[id_col].values]
        streamflow_aligned = streamflow[comid_order]

        # Create designer
        matrix = streamflow_aligned.values
        labels = comid_order

        return cls(matrix, locations_filtered, labels)

    def design_network(
        self,
        n_sensors: int,
        weights: Optional[Union[np.ndarray, str, Path]] = None,
        weight_column: Optional[str] = None,
        weight_fill_value: float = 0.0,
        existing_sensors: Optional[List[int]] = None,
        train_frac: float = 0.7,
        evaluate: bool = True,
        verbose: bool = True
    ) -> 'NetworkDesignResult':
        """Design an optimal sensor network.

        Splits the record in time (when ``evaluate=True``), selects
        sensor locations by weighted pivoted QR on the training block,
        and scores reconstruction of the full field on the held-out
        block.

        Parameters
        ----------
        n_sensors : int
            Number of sensors to place (including ``existing_sensors``).
        weights : np.ndarray or str or Path, optional
            Per-location weights aligned to the data columns, or a path
            to a spatial weight file resolved via
            :func:`~hydrosensenet.calculate_spatial_weights`.
        weight_column : str, optional
            Column to read when ``weights`` is a file path.
        weight_fill_value : float, default=0.0
            Fill for locations not covered by the weight file. The
            default ``0.0`` removes uncovered locations from
            consideration; use a tiny positive value (e.g. ``1e-10``)
            to keep them selectable as a last resort.
        existing_sensors : list of int, optional
            Column indices of gauges to keep (indices into the original
            column order). They appear first in the result; if
            ``len(existing_sensors) == n_sensors`` the QR step is
            skipped and the existing network is just evaluated.
        train_frac : float, default=0.7
            Fraction of timesteps used for design when evaluating.
        evaluate : bool, default=True
            Hold out ``1 - train_frac`` of the record and compute
            reconstruction metrics (NSE, NNSE, relative error).
        verbose : bool, default=True
            Print progress information.

        Returns
        -------
        NetworkDesignResult
            Selected indices, locations, labels, and (if evaluated)
            per-location performance metrics.

        Notes
        -----
        Columns containing NaN/inf in the training block are dropped
        before selection (a warning reports how many). All indices in
        the returned result refer to the **original** column order, and
        per-location metric arrays have the original length with NaN at
        dropped locations. An ``existing_sensors`` entry pointing at a
        dropped column raises a ``ValueError``.
        """
        import time

        if verbose:
            print(f"\n{'='*70}")
            print(f"SENSOR NETWORK DESIGN")
            print(f"{'='*70}")
            print(f"Input data: {self.streamflow_data.shape[0]:,} timesteps × {self.streamflow_data.shape[1]:,} locations")
            print(f"Target sensors: {n_sensors}")
            print(f"Evaluation: {'Enabled' if evaluate else 'Disabled'}")

        n_locations = self.streamflow_data.shape[1]

        # Process weights (built against the original, unfiltered columns)
        if weights is not None:
            if verbose:
                print(f"\n[1/4] Processing weights...")
            if isinstance(weights, (str, Path)):
                weight_array = calculate_spatial_weights(
                    self.locations,
                    weights,
                    weight_column=weight_column,
                    fill_value=weight_fill_value
                )
            elif isinstance(weights, np.ndarray):
                weight_array = weights
            else:
                raise TypeError("weights must be array or file path")
            if len(weight_array) != n_locations:
                raise ValueError(
                    f"Weights length ({len(weight_array)}) must match number "
                    f"of locations ({n_locations})"
                )
            if verbose:
                print(f"      ✓ Weights applied")
        else:
            weight_array = None

        # Split data if evaluation requested. Columns with NaN/inf in the
        # training block are dropped in BOTH paths; good_cols maps the
        # filtered column space back to original column indices.
        if evaluate:
            if verbose:
                print(f"\n[2/4] Splitting data (train: {train_frac:.0%}, test: {1-train_frac:.0%})...")
            start_time = time.time()
            X_train, X_test, mapping = split_timeseries(
                self.streamflow_data,
                train_frac=train_frac,
                filter_invalid=True,
                return_mapping=True
            )
            good_cols = np.asarray(mapping["good_cols"])
            if verbose:
                print(f"      ✓ Train: {X_train.shape[0]:,} × {X_train.shape[1]:,}")
                print(f"      ✓ Test:  {X_test.shape[0]:,} × {X_test.shape[1]:,}")
                print(f"      Completed in {time.time() - start_time:.2f}s")
        else:
            finite_cols = np.isfinite(self.streamflow_data).all(axis=0)
            good_cols = np.where(finite_cols)[0]
            X_train = (
                self.streamflow_data[:, good_cols]
                if len(good_cols) < n_locations else self.streamflow_data
            )
            X_test = None

        n_dropped = n_locations - len(good_cols)
        if n_dropped:
            warnings.warn(
                f"{n_dropped} of {n_locations} locations contain NaN/inf in "
                f"the training data and were excluded from selection.",
                UserWarning,
                stacklevel=2,
            )
            if verbose:
                print(f"      ⚠ Excluded {n_dropped} locations with NaN/inf values")
        if n_sensors > len(good_cols):
            raise ValueError(
                f"Requested {n_sensors} sensors but only {len(good_cols)} "
                f"locations have valid (finite) training data"
            )

        # Weights and existing sensors are given in original column space;
        # translate them into the filtered space used for selection.
        weight_array_filtered = (
            weight_array[good_cols] if (weight_array is not None and n_dropped) else weight_array
        )
        if existing_sensors is not None:
            existing_sensors = [int(i) for i in existing_sensors]
            out_of_range = [i for i in existing_sensors if not 0 <= i < n_locations]
            if out_of_range:
                raise ValueError(f"existing_sensors indices out of range: {out_of_range}")
            if n_dropped:
                orig_to_filtered = {int(orig): k for k, orig in enumerate(good_cols)}
                dropped_fixed = [i for i in existing_sensors if i not in orig_to_filtered]
                if dropped_fixed:
                    raise ValueError(
                        f"existing_sensors columns {dropped_fixed} contain NaN/inf "
                        f"in the training data and cannot be evaluated"
                    )
                fixed_filtered = [orig_to_filtered[i] for i in existing_sensors]
            else:
                fixed_filtered = existing_sensors
        else:
            fixed_filtered = None

        # Perform sensor placement (skip if all sensors are already specified)
        if fixed_filtered is not None and len(fixed_filtered) == n_sensors:
            # Skip QR - we're just evaluating existing sensors
            if verbose:
                print(f"\n[3/4] Using {n_sensors} existing sensor locations (skipping QR)...")
            selected_filtered = np.array(fixed_filtered)
        else:
            # Run QR decomposition to find optimal locations
            if verbose:
                print(f"\n[3/4] Running QR decomposition for sensor placement...")
                print(f"      Matrix size: {X_train.shape[0]:,} × {X_train.shape[1]:,}")
                print(f"      This may take several minutes for large datasets...")
                start_time = time.time()

            selected_filtered = sensor_placement_qr(
                X_train,
                n_sensors=n_sensors,
                weights=weight_array_filtered,
                fixed_indices=fixed_filtered,
                verbose=verbose
            )

            if verbose:
                print(f"      ✓ QR decomposition completed in {time.time() - start_time:.2f}s")
                print(f"      ✓ Selected {len(selected_filtered)} sensor locations")

        # Map selection back to original column indices
        selected_indices = (
            good_cols[selected_filtered] if n_dropped else np.asarray(selected_filtered)
        )

        # Evaluate if requested
        if evaluate and X_test is not None:
            if verbose:
                print(f"\n[4/4] Evaluating reconstruction performance...")
                start_time = time.time()

            eval_results = reconstruction_evaluation(
                X_train, X_test, np.asarray(selected_filtered), n_sensors, verbose=verbose
            )

            # Per-location metric arrays are computed in filtered space;
            # expand them to original length (NaN at dropped columns) so
            # they can be indexed with original column indices.
            def _expand(values):
                full = np.full(n_locations, np.nan)
                full[good_cols] = values
                return full

            nse = _expand(eval_results['nse'])
            nnse = _expand(eval_results['nnse'])
            eval_results['good_cols'] = good_cols
            r2 = None  # Skip R² calculation (same as NSE)

            if verbose:
                print(f"      ✓ Evaluation completed in {time.time() - start_time:.2f}s")
                print(f"      Relative error: {eval_results['relative_error']:.4f}")
        else:
            eval_results = None
            r2 = nse = nnse = None

        # Create result object
        result = NetworkDesignResult(
            selected_indices=selected_indices,
            locations=self.locations.iloc[selected_indices],
            location_labels=[self.location_labels[i] for i in selected_indices],
            performance_metrics={
                'r2': r2,
                'nse': nse,
                'nnse': nnse,
                'eval_results': eval_results
            } if evaluate else None,
            n_sensors=n_sensors,
            weights=weight_array
        )

        if verbose:
            print(f"\n{'='*70}")
            print(f"DESIGN COMPLETE")
            print(f"{'='*70}\n")

        return result


class NetworkDesignResult:
    """Results from sensor network design.

    Attributes
    ----------
    selected_indices : np.ndarray
        Selected column indices in priority order (rank 1 first).
    locations : gpd.GeoDataFrame
        Rows of the input locations for the selected sensors.
    location_labels : list of str
        Labels of the selected sensors, in the same order.
    performance_metrics : dict or None
        When designed with ``evaluate=True``: per-location ``nse`` and
        ``nnse`` arrays plus the raw ``eval_results`` dict from
        :func:`~hydrosensenet.reconstruction_evaluation`.
    n_sensors : int
        Number of sensors requested.
    weights : np.ndarray or None
        Weight vector used for the design, if any.
    """

    def __init__(
        self,
        selected_indices: np.ndarray,
        locations: gpd.GeoDataFrame,
        location_labels: List[str],
        performance_metrics: Optional[Dict],
        n_sensors: int,
        weights: Optional[np.ndarray]
    ):
        self.selected_indices = selected_indices
        self.locations = locations
        self.location_labels = location_labels
        self.performance_metrics = performance_metrics
        self.n_sensors = n_sensors
        self.weights = weights

    def print_summary(self):
        """Print summary of results."""
        print(f"\n{'='*60}")
        print(f"SENSOR NETWORK DESIGN RESULTS")
        print(f"{'='*60}")
        print(f"Number of sensors selected: {self.n_sensors}")
        print(f"Selected locations: {self.selected_indices[:10]}..." if len(self.selected_indices) > 10 else f"Selected locations: {self.selected_indices}")

        if self.performance_metrics:
            print(f"\nPERFORMANCE METRICS:")
            r2 = self.performance_metrics.get('r2')
            nse = self.performance_metrics.get('nse')
            nnse = self.performance_metrics.get('nnse')

            if r2 is not None:
                print(f"  Median R²: {np.nanmedian(r2):.3f}")
            if nse is not None:
                print(f"  Median NSE: {np.nanmedian(nse):.3f}")
            if nnse is not None:
                print(f"  Median NNSE: {np.nanmedian(nnse):.3f}")

            eval_res = self.performance_metrics.get('eval_results')
            if eval_res:
                print(f"  Relative Error: {eval_res['relative_error']:.4f}")

        print(f"{'='*60}\n")

    def export(
        self,
        output_path: Union[str, Path],
        format: str = "auto",
        include_metrics: bool = True
    ):
        """Export results to file."""
        output_path = Path(output_path)

        # Auto-detect format
        if format == "auto":
            suffix = output_path.suffix.lower()
            format_map = {
                '.csv': 'csv',
                '.shp': 'shapefile',
                '.geojson': 'geojson',
                '.gpkg': 'geopackage',
                '.xlsx': 'excel'
            }
            format = format_map.get(suffix, 'csv')

        # Prepare output data
        output_data = self.locations.copy()
        output_data['sensor_rank'] = range(1, len(output_data) + 1)

        if include_metrics and self.performance_metrics:
            r2 = self.performance_metrics.get('r2')
            nse = self.performance_metrics.get('nse')
            nnse = self.performance_metrics.get('nnse')
            if r2 is not None:
                output_data['r2'] = r2[self.selected_indices]
            if nse is not None:
                output_data['nse'] = nse[self.selected_indices]
            if nnse is not None:
                output_data['nnse'] = nnse[self.selected_indices]

        # Export based on format
        if format == "csv":
            # Convert to regular DataFrame for CSV
            df = pd.DataFrame(output_data.drop(columns='geometry'))
            # Convert to centroids if geometries are not Points
            if output_data.geometry.geom_type.iloc[0] != 'Point':
                output_proj = output_data.to_crs("EPSG:5070")
                centroids = output_proj.geometry.centroid.to_crs(output_data.crs)
                df['longitude'] = centroids.x
                df['latitude'] = centroids.y
            else:
                df['longitude'] = output_data.geometry.x
                df['latitude'] = output_data.geometry.y
            df.to_csv(output_path, index=False)
        elif format in ["shapefile", "shp"]:
            output_data.to_file(output_path, driver="ESRI Shapefile")
        elif format == "geojson":
            output_data.to_file(output_path, driver="GeoJSON")
        elif format in ["geopackage", "gpkg"]:
            output_data.to_file(output_path, driver="GPKG")
        elif format == "excel":
            df = pd.DataFrame(output_data.drop(columns='geometry'))
            # Convert to centroids if geometries are not Points
            if output_data.geometry.geom_type.iloc[0] != 'Point':
                output_proj = output_data.to_crs("EPSG:5070")
                centroids = output_proj.geometry.centroid.to_crs(output_data.crs)
                df['longitude'] = centroids.x
                df['latitude'] = centroids.y
            else:
                df['longitude'] = output_data.geometry.x
                df['latitude'] = output_data.geometry.y
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Results exported to: {output_path}")

    def get_dataframe(self) -> pd.DataFrame:
        """
        Get results as DataFrame.

        Returns
        -------
        pd.DataFrame
            Selected sensors with coordinates and metrics.
        """
        # Convert to centroids if geometries are not Points
        if self.locations.geometry.geom_type.iloc[0] != 'Point':
            locations_proj = self.locations.to_crs("EPSG:5070")
            centroids = locations_proj.geometry.centroid.to_crs(self.locations.crs)
            lons = centroids.x
            lats = centroids.y
        else:
            lons = self.locations.geometry.x
            lats = self.locations.geometry.y

        df = pd.DataFrame({
            'sensor_rank': range(1, len(self.selected_indices) + 1),
            'location_index': self.selected_indices,
            'location_label': self.location_labels,
            'longitude': lons,
            'latitude': lats
        })

        if self.performance_metrics:
            r2 = self.performance_metrics.get('r2')
            nse = self.performance_metrics.get('nse')
            nnse = self.performance_metrics.get('nnse')
            if r2 is not None:
                df['r2'] = r2[self.selected_indices]
            if nse is not None:
                df['nse'] = nse[self.selected_indices]
            if nnse is not None:
                df['nnse'] = nnse[self.selected_indices]

        return df

    def plot(
        self,
        figsize=(12, 8),
        show_rank=True,
        basemap=True,
        title="Optimal Sensor Network Design",
        save_path=None,
        flowlines_gdf=None,
        rank_column='Median Rank'
    ):
        """Create map visualization of selected sensors."""
        try:
            import matplotlib.pyplot as plt
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            from matplotlib.collections import LineCollection
            from matplotlib.colors import Normalize
        except ImportError as e:
            raise ImportError(
                "Plotting requires the optional visualization dependencies "
                "(matplotlib, cartopy). Install them with:\n"
                "  pip install 'hydrosensenet[viz]'"
            ) from e

        # Use Lambert Conformal projection if flowlines provided, otherwise PlateCarree
        if flowlines_gdf is not None:
            proj = ccrs.LambertConformal(
                central_latitude=33,
                central_longitude=-96,
                standard_parallels=(33.0, 45.0)
            )
        else:
            proj = ccrs.PlateCarree()

        # Create figure with geographic projection
        fig, ax = plt.subplots(
            figsize=figsize,
            dpi=600,
            subplot_kw={'projection': proj}
        )

        # Configure axis
        if flowlines_gdf is not None:
            ax.set_extent([-106.65, -93.0, 25.0, 36.5], crs=ccrs.PlateCarree())
            ax.spines['geo'].set_visible(False)
        else:
            # Set extent based on sensor locations
            # Convert to centroids if geometries are not Points
            if self.locations.geometry.geom_type.iloc[0] != 'Point':
                locations_proj = self.locations.to_crs("EPSG:5070")
                centroids = locations_proj.geometry.centroid.to_crs(self.locations.crs)
                lons = centroids.x
                lats = centroids.y
            else:
                lons = self.locations.geometry.x
                lats = self.locations.geometry.y
            margin = 0.5
            ax.set_extent([
                lons.min() - margin, lons.max() + margin,
                lats.min() - margin, lats.max() + margin
            ])

        # Plot flowlines if provided
        if flowlines_gdf is not None:
            # Project flowlines to map projection
            flowlines_proj = flowlines_gdf.to_crs(proj.proj4_params)

            # First layer: thin black outline for contrast
            lines_outline = LineCollection(
                [np.array(geometry.xy).T for geometry in flowlines_proj.geometry],
                linewidths=0.025,
                alpha=0.8,
                color='black',
                zorder=1
            )
            ax.add_collection(lines_outline)

            # Second layer: colored lines by rank
            lines = LineCollection(
                [np.array(geometry.xy).T for geometry in flowlines_proj.geometry],
                linewidths=1,
                alpha=1,
                zorder=1
            )
            lines.set_array(flowlines_proj[rank_column])
            lines.set_cmap('viridis_r')
            ax.add_collection(lines)

            # Add colorbar for flowlines
            cb_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6])
            cb = fig.colorbar(lines, cax=cb_ax, orientation='vertical', label='Sensor Rank')

            # Add legend (matches original even though no labeled elements)
            ax.legend(frameon=False, loc='best')

            plt.subplots_adjust(left=0.05, right=0.8, top=0.95, bottom=0.1)
        else:
            # Original scatter plot for sensors
            # Convert to centroids if geometries are not Points
            if self.locations.geometry.geom_type.iloc[0] != 'Point':
                locations_proj = self.locations.to_crs("EPSG:5070")
                centroids = locations_proj.geometry.centroid.to_crs(self.locations.crs)
                lons = centroids.x
                lats = centroids.y
            else:
                lons = self.locations.geometry.x
                lats = self.locations.geometry.y

            # Color by rank (lower rank = more important)
            scatter = ax.scatter(
                lons, lats,
                c=range(1, len(lons) + 1),
                cmap='RdYlGn_r',
                s=100,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.8,
                transform=ccrs.PlateCarree(),
                zorder=5
            )

            # Add rank numbers if requested
            if show_rank and len(lons) <= 50:
                for i, (lon, lat) in enumerate(zip(lons, lats)):
                    ax.text(
                        lon, lat, str(i+1),
                        fontsize=6,
                        ha='center', va='center',
                        color='white',
                        weight='bold',
                        transform=ccrs.PlateCarree(),
                        zorder=6
                    )

            # Add colorbar for sensors
            cbar = plt.colorbar(scatter, ax=ax, pad=0.02, shrink=0.8)
            cbar.set_label('Sensor Rank (1 = highest priority)', rotation=270, labelpad=15)

            # Add gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3)
            gl.top_labels = False
            gl.right_labels = False

            plt.tight_layout()

        # Add basemap features
        if basemap:
            ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=.2)
            ax.add_feature(cfeature.STATES, linestyle=':', alpha=.2)

        # Add title
        if title:
            ax.set_title(title, fontsize=14, weight='bold', pad=10)

        # Add performance metrics if available (only for scatter plot mode)
        if flowlines_gdf is None and self.performance_metrics:
            r2 = self.performance_metrics.get('r2')
            nse = self.performance_metrics.get('nse')
            text_parts = []
            if r2 is not None:
                text_parts.append(f"Mean R²: {np.nanmean(r2):.3f}")
            if nse is not None:
                text_parts.append(f"Mean NSE: {np.nanmean(nse):.3f}")
            if text_parts:
                ax.text(
                    0.02, 0.98, '\n'.join(text_parts),
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                )

        # Save if requested
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=600)
            print(f"Figure saved to: {save_path}")

        return fig, ax

    def plot_comparison(
        self,
        baseline_result: 'NetworkDesignResult',
        flowlines_gdf: gpd.GeoDataFrame,
        location_labels: List[str],
        comparison_labels: Optional[Dict[str, str]] = None,
        metric: str = 'nnse',
        figsize=(7, 5),
        save_path=None,
        comid_column: Optional[str] = None
    ):
        """
        Plot comparison of NNSE between this network and a baseline network.

        Parameters
        ----------
        baseline_result : NetworkDesignResult
            Baseline network to compare against (e.g., USGS network)
        flowlines_gdf : GeoDataFrame
            GeoDataFrame containing flowline geometries with COMID column
        location_labels : List[str]
            Labels for all locations (COMIDs) matching the order in performance metrics
        comparison_labels : dict, optional
            Dictionary with keys 'optimal' and 'baseline' for scatter plot labels
        metric : str
            Performance metric to compare ('nnse', 'nse', or 'r2')
        figsize : tuple
            Figure size (width, height)
        save_path : str, optional
            Path to save the figure
        comid_column : str, optional
            Name of the COMID column in flowlines_gdf. If None, will auto-detect
            by trying 'COMID', 'comid', 'feature_id', 'FEATUREID'

        Returns
        -------
        tuple
            (figure, axis)
        """
        try:
            import matplotlib.pyplot as plt
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            from matplotlib.collections import LineCollection
            from matplotlib.colors import Normalize
        except ImportError as e:
            raise ImportError(
                "Plotting requires the optional visualization dependencies "
                "(matplotlib, cartopy). Install them with:\n"
                "  pip install 'hydrosensenet[viz]'"
            ) from e

        # Default labels
        if comparison_labels is None:
            comparison_labels = {
                'optimal': 'Optimal sensors',
                'baseline': 'Baseline sensors'
            }

        # Get performance metrics
        if not self.performance_metrics or not baseline_result.performance_metrics:
            raise ValueError("Both results must have performance metrics for comparison")

        optimal_metric = self.performance_metrics.get(metric)
        baseline_metric = baseline_result.performance_metrics.get(metric)

        if optimal_metric is None or baseline_metric is None:
            raise ValueError(f"Metric '{metric}' not available in both results")

        # Calculate difference (optimal - baseline)
        diff_metric = optimal_metric - baseline_metric

        # Auto-detect COMID column if not specified
        flowlines_gdf = flowlines_gdf.copy()
        if comid_column is None:
            # Try common COMID column names
            for candidate in ['COMID', 'comid', 'feature_id', 'FEATUREID']:
                if candidate in flowlines_gdf.columns:
                    comid_column = candidate
                    break
            if comid_column is None:
                raise ValueError(
                    f"Could not find COMID column in flowlines_gdf. "
                    f"Available columns: {list(flowlines_gdf.columns)}. "
                    f"Please specify comid_column parameter."
                )

        # Create DataFrame with COMID and difference
        diff_df = pd.DataFrame({
            'COMID': location_labels,
            'diff': diff_metric
        })

        # Merge with flowlines
        flowlines_gdf['COMID'] = flowlines_gdf[comid_column].astype(str)
        diff_df['COMID'] = diff_df['COMID'].astype(str)
        flowlines_with_diff = flowlines_gdf.merge(diff_df, on='COMID', how='left')

        # Setup projection
        proj = ccrs.LambertConformal(
            central_latitude=33,
            central_longitude=-96,
            standard_parallels=(33.0, 45.0)
        )

        # Project flowlines
        flowlines_proj = flowlines_with_diff.to_crs(proj.proj4_params)

        # Create figure
        fig, ax = plt.subplots(
            figsize=figsize,
            dpi=600,
            subplot_kw={'projection': proj}
        )
        ax.set_extent([-106.65, -93.0, 25.0, 36.5], crs=ccrs.PlateCarree())
        ax.spines['geo'].set_visible(False)

        # Plot flowlines with difference colors
        lines = LineCollection(
            [np.array(geometry.xy).T for geometry in flowlines_proj.geometry],
            linewidths=1,
            alpha=1,
            zorder=1
        )
        norm = Normalize(vmin=-1, vmax=1)
        lines.set_array(flowlines_proj['diff'])
        lines.set_cmap('bwr_r')
        lines.set_norm(norm)
        ax.add_collection(lines)

        # Add colorbar
        cb_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6])
        metric_label = metric.upper() if metric != 'r2' else 'R²'
        cb = fig.colorbar(
            lines,
            cax=cb_ax,
            orientation='vertical',
            label=f'Δ{metric_label}'
        )

        # Prepare sensor centroids for both networks
        def get_centroids(result):
            if result.locations.geometry.geom_type.iloc[0] != 'Point':
                locs_proj = result.locations.to_crs("EPSG:5070")
                centroids = locs_proj.geometry.centroid.to_crs("EPSG:4326")
                centroids_gdf = gpd.GeoDataFrame(geometry=centroids, crs="EPSG:4326")
            else:
                centroids_gdf = result.locations.to_crs("EPSG:4326")
            return centroids_gdf.to_crs(proj.proj4_params)

        optimal_centroids = get_centroids(self)
        baseline_centroids = get_centroids(baseline_result)

        # Plot sensors
        ax.scatter(
            baseline_centroids.geometry.x,
            baseline_centroids.geometry.y,
            color='k',
            edgecolor='white',
            linewidths=0.6,
            alpha=0.8,
            s=7,
            label=comparison_labels['baseline'],
            zorder=5
        )

        ax.scatter(
            optimal_centroids.geometry.x,
            optimal_centroids.geometry.y,
            color='green',
            edgecolor='white',
            linewidths=0.6,
            alpha=0.8,
            s=7,
            label=comparison_labels['optimal'],
            zorder=5
        )

        # Add map features and legend
        ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=.2)
        ax.add_feature(cfeature.STATES, linestyle=':', alpha=.2)
        ax.legend(frameon=False, loc='best')

        plt.subplots_adjust(left=0.05, right=0.8, top=0.95, bottom=0.1)

        # Save if requested
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=600)
            print(f"Figure saved to: {save_path}")

        return fig, ax
