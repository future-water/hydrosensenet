"""High-level API for sensor network design - Simple interface for water managers."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import griddata
from typing import Union, Tuple, List, Optional, Dict
import warnings
import time

from .data import load_streamflow_data, prepare_gauge_locations, split_timeseries, prepare_matrix
from .core import sensor_placement_qr, qr_pivot_selection, calculate_performance_metrics, reconstruction_evaluation
from .spatial import calculate_spatial_weights


class SensorNetworkDesigner:
    """Easy-to-use interface for designing optimal sensor networks."""

    def __init__(
        self,
        streamflow_data: np.ndarray,
        locations: gpd.GeoDataFrame,
        location_labels: Optional[List[str]] = None
    ):
        """Initialize sensor network designer."""
        self.streamflow_data = streamflow_data
        self.locations = locations
        self.location_labels = location_labels or [str(i) for i in range(streamflow_data.shape[1])]

        # Validate dimensions
        if streamflow_data.shape[1] != len(locations):
            raise ValueError(
                f"Streamflow data has {streamflow_data.shape[1]} locations but "
                f"locations GeoDataFrame has {len(locations)} rows"
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
        existing_sensors: Optional[List[int]] = None,
        train_frac: float = 0.7,
        evaluate: bool = True,
        verbose: bool = True,
        # New regional parameters
        region_column: Optional[str] = None,
        sensors_per_region: Optional[Dict[str, int]] = None
    ) -> 'NetworkDesignResult':
        """
        Design optimal sensor network.
        
        Parameters
        ----------
        n_sensors : int
            Total number of sensors to select (ignored if sensors_per_region is provided)
        weights : array-like or Path, optional
            Importance weights for locations
        weight_column : str, optional
            Column name for weights if loading from file
        existing_sensors : list of int, optional
            Indices of existing sensors to keep
        train_frac : float
            Fraction of data to use for training (default: 0.7)
        evaluate : bool
            Whether to evaluate reconstruction performance (default: True)
        verbose : bool
            Print progress messages (default: True)
        region_column : str, optional
            Column name in self.locations containing region assignments.
            If provided, performs regional QR selection instead of global.
        sensors_per_region : dict, optional
            Dictionary mapping region names to number of sensors per region.
            Required if region_column is provided.
            
        Returns
        -------
        NetworkDesignResult
            Results object with selected sensors and performance metrics
            
        Examples
        --------
        # Global selection (original behavior)
        result = designer.design_network(n_sensors=50)
        
        # Regional selection
        result = designer.design_network(
            n_sensors=None,  # ignored when using regional
            region_column='basin_name',
            sensors_per_region={'Basin_A': 10, 'Basin_B': 15, 'Basin_C': 5}
        )
        """
        # Validate regional parameters
        use_regional = region_column is not None
        if use_regional:
            if sensors_per_region is None:
                raise ValueError("sensors_per_region must be provided when using region_column")
            if region_column not in self.locations.columns:
                raise ValueError(f"Column '{region_column}' not found in locations GeoDataFrame")
            # Calculate total sensors from regional allocation
            total_sensors = sum(sensors_per_region.values())
        else:
            total_sensors = n_sensors

        if verbose:
            print(f"\n{'='*70}")
            print(f"SENSOR NETWORK DESIGN")
            print(f"{'='*70}")
            print(f"Input data: {self.streamflow_data.shape[0]:,} timesteps × {self.streamflow_data.shape[1]:,} locations")
            if use_regional:
                print(f"Mode: Regional selection")
                print(f"Region column: {region_column}")
                print(f"Total sensors: {total_sensors} across {len(sensors_per_region)} regions")
                for region, count in sensors_per_region.items():
                    print(f"  - {region}: {count} sensors")
            else:
                print(f"Mode: Global selection")
                print(f"Target sensors: {n_sensors}")
            print(f"Evaluation: {'Enabled' if evaluate else 'Disabled'}")

        # Process weights
        if weights is not None:
            if verbose:
                print(f"\n[1/4] Processing weights...")
            if isinstance(weights, (str, Path)):
                weight_array = calculate_spatial_weights(
                    self.locations,
                    weights,
                    weight_column=weight_column
                )
            elif isinstance(weights, np.ndarray):
                weight_array = weights
            else:
                raise TypeError("weights must be array or file path")
            if verbose:
                print(f"      ✓ Weights applied")
        else:
            weight_array = None

        # Split data if evaluation requested
        if evaluate:
            if verbose:
                print(f"\n[2/4] Splitting data (train: {train_frac:.0%}, test: {1-train_frac:.0%})...")
            start_time = time.time()
            X_train, X_test = split_timeseries(
                self.streamflow_data,
                train_frac=train_frac,
                filter_invalid=True
            )
            if verbose:
                print(f"      ✓ Train: {X_train.shape[0]:,} × {X_train.shape[1]:,}")
                print(f"      ✓ Test:  {X_test.shape[0]:,} × {X_test.shape[1]:,}")
                print(f"      Completed in {time.time() - start_time:.2f}s")
        else:
            X_train = self.streamflow_data
            X_test = None

        # Perform sensor placement (skip if all sensors are already specified)
        if existing_sensors is not None and len(existing_sensors) == n_sensors:
            # Skip QR - we're just evaluating existing sensors
            if verbose:
                print(f"\n[3/4] Using {n_sensors} existing sensor locations (skipping QR)...")
            selected_indices = np.array(existing_sensors)
        
        elif use_regional:
            # Regional QR selection
            if verbose:
                print(f"\n[3/4] Running regional QR decomposition...")
                print(f"      Matrix size: {X_train.shape[0]:,} × {X_train.shape[1]:,}")
                start_time = time.time()
            
            # Prepare region assignments DataFrame
            region_assignments = pd.DataFrame({
                region_column: self.locations[region_column],
                'col_pos': np.arange(len(self.locations))
            })
            
            # Apply weights if provided (note: qr_pivot_selection doesn't support weights yet)
            if weight_array is not None and verbose:
                warnings.warn(
                    "Weights are not currently supported with regional selection. "
                    "Consider implementing weighted regional QR if needed."
                )
            
            # Call qr_pivot_selection
            selected_locations_df, selected_indices = qr_pivot_selection(
                X_train,
                region_assignments,
                sensors_per_region,
                region_column=region_column
            )
            
            selected_indices = np.array(selected_indices)
            
            if verbose:
                print(f"      ✓ Regional QR completed in {time.time() - start_time:.2f}s")
                print(f"      ✓ Selected {len(selected_indices)} sensor locations")
        
        else:
            # Run QR decomposition to find optimal locations
            if verbose:
                print(f"\n[3/4] Running QR decomposition for sensor placement...")
                print(f"      Matrix size: {X_train.shape[0]:,} × {X_train.shape[1]:,}")
                print(f"      This may take several minutes for large datasets...")
                start_time = time.time()

            selected_indices = sensor_placement_qr(
                X_train,
                n_sensors=n_sensors,
                weights=weight_array,
                fixed_indices=existing_sensors,
                verbose=verbose
            )

            if verbose:
                print(f"      ✓ QR decomposition completed in {time.time() - start_time:.2f}s")
                print(f"      ✓ Selected {len(selected_indices)} sensor locations")

        # Evaluate if requested
        if evaluate and X_test is not None:
            if verbose:
                print(f"\n[4/4] Evaluating reconstruction performance...")
                start_time = time.time()

            eval_results = reconstruction_evaluation(
                X_train, X_test, selected_indices, n_sensors, verbose=verbose
            )

            # Use pre-calculated NSE/NNSE from eval_results (avoids redundant computation)
            nse = eval_results['nse']
            nnse = eval_results['nnse']
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
            weights=weight_array,
            region_column=region_column if use_regional else None,
            sensors_per_region=sensors_per_region if use_regional else None,
            all_locations=self.locations
        )

        if verbose:
            print(f"\n{'='*70}")
            print(f"DESIGN COMPLETE")
            print(f"{'='*70}\n")

        return result


class NetworkDesignResult:
    """Results from sensor network design."""

    def __init__(
        self,
        selected_indices: np.ndarray,
        locations: gpd.GeoDataFrame,
        location_labels: List[str],
        performance_metrics: Optional[Dict],
        n_sensors: int,
        weights: Optional[np.ndarray],
        region_column: Optional[str] = None,
        sensors_per_region: Optional[Dict[str, int]] = None,
        all_locations: Optional[gpd.GeoDataFrame] = None
    ):
        self.selected_indices = selected_indices
        self.locations = locations
        self.location_labels = location_labels
        self.performance_metrics = performance_metrics
        self.n_sensors = n_sensors
        self.weights = weights
        self.region_column = region_column
        self.sensors_per_region = sensors_per_region
        self.all_locations = all_locations

    def print_summary(self):
        """Print summary of results."""
        print(f"\n{'='*60}")
        print(f"SENSOR NETWORK DESIGN RESULTS")
        print(f"{'='*60}")
        print(f"Number of sensors selected: {self.n_sensors}")

        if self.region_column and self.sensors_per_region:
            print(f"\nREGIONAL ALLOCATION:")
            for region, count in self.sensors_per_region.items():
                print(f"  {region}: {count} sensors")
        
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

        # Add region information if available
        if self.region_column:
            df['region'] = self.locations[self.region_column].values

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
    
    def plot_sensors(
        self,
        save_path: Optional[str] = None,
        figsize: tuple = (8, 6),
        dpi: int = 600,
        projection_epsg: str = 'EPSG:3857'
    ):
        """
        Simple plot showing selected sensor locations.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save figure
        figsize : tuple
            Figure size (width, height)
        dpi : int
            Resolution
        projection_epsg : str
            EPSG code for centroid calculation if geometries are not Points
            Default: 'EPSG:3857' (Web Mercator)
            Alternatives: 'EPSG:5070' (NAD83 Albers for North America), 'EPSG:6933' (Equal Earth)
            
        Returns
        -------
        fig, ax : matplotlib objects
        
        Example
        -------
        result = designer.design_network(n_sensors=50)
        result.plot_sensors(save_path='sensors.png')
        """
        # Get coordinates
        lats, lons = self._get_coordinates(projection_epsg)
        
        # Create figure
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Auto extent
        extent = self._calculate_extent(lons, lats)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        # Plot sensors
        ax.scatter(
            lons, lats,
            marker='o',
            color='green',
            s=15,
            edgecolors='white',
            linewidth=0.5,
            transform=ccrs.PlateCarree(),
            label=f'Selected sensors (n={len(lats)})',
            zorder=5
        )
        
        # Map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
        
        # Gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.3, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Legend
        ax.legend(loc='upper right', frameon=True, fontsize=9, framealpha=0.9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi, transparent=True)
            print(f"Figure saved to: {save_path}")
        
        return fig, ax
    
    def plot_comparison(
        self,
        baseline_result,
        performance_metric: str = 'nnse',
        flowline_gdf: Optional = None,
        save_path: Optional[str] = None,
        figsize: tuple = (8, 6),
        dpi: int = 600,
        projection_epsg: str = 'EPSG:3857'
    ):
        """
        Plot comparison showing improvement over baseline.
        
        Parameters
        ----------
        baseline_result : NetworkDesignResult
            Baseline network to compare against
        performance_metric : str
            Performance metric to compare ('nnse', 'nse', 'r2')
        flowline_gdf : GeoDataFrame, optional
            Flowlines to plot with performance colors. If None, plots gridded background.
        save_path : str, optional
            Path to save figure
        figsize : tuple
            Figure size
        dpi : int
            Resolution
        projection_epsg : str
            EPSG code for centroid calculation if geometries are not Points
            Default: 'EPSG:3857': Web Mercator (global)
            Common alternatives:
            - 'EPSG:5070': NAD83 Albers Equal Area (North America)
            - 'EPSG:3857': Web Mercator (global)
            - 'EPSG:6933': Equal Earth (global)
            - 'EPSG:32633': UTM Zone 33N (Europe/Africa)
            - 'EPSG:32718': UTM Zone 18S (South America)
            
        Returns
        -------
        fig, ax : matplotlib objects
        
        Examples
        --------
        # Gridded background (no flowlines)
        optimal.plot_comparison(baseline, performance_metric='nnse')
        
        # With flowlines
        optimal.plot_comparison(baseline, flowline_gdf=flowlines, performance_metric='nnse')
        """
        # Check if metrics exist
        if not self.performance_metrics or not baseline_result.performance_metrics:
            raise ValueError("Both results must have performance metrics. Run design_network() with evaluate=True")
        
        # Get metrics
        optimal_metric = self.performance_metrics.get(performance_metric)
        baseline_metric = baseline_result.performance_metrics.get(performance_metric)
        
        if optimal_metric is None or baseline_metric is None:
            raise ValueError(f"Metric '{performance_metric}' not found in results")
        
        # Calculate difference
        diff_metric = optimal_metric - baseline_metric
        
        # Get coordinates
        opt_lats, opt_lons = self._get_coordinates(projection_epsg)
        base_lats, base_lons = baseline_result._get_coordinates(projection_epsg)
        
        # Create figure
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Set extent
        all_lons = np.concatenate([opt_lons, base_lons])
        all_lats = np.concatenate([opt_lats, base_lats])
        extent = self._calculate_extent(all_lons, all_lats)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        # Plot background metric if flowline_gdf NOT provided
        if flowline_gdf is None:
            # Create gridded background 
            gridded_diff = self._create_grid(diff_metric, performance_metric, projection_epsg=projection_epsg)
            
            if gridded_diff is not None:
                # Check if it's sparse grid or regular grid
                if isinstance(gridded_diff, dict) and gridded_diff.get('type') == 'sparse':
                    # Sparse grid - plot as scatter to avoid interpolation artifacts
                    # Calculate cell size based on grid spacing
                    unique_lons = np.unique(gridded_diff['lons'])
                    unique_lats = np.unique(gridded_diff['lats'])
                    
                    if len(unique_lons) > 1 and len(unique_lats) > 1:
                        # Calculate approximate cell size
                        lon_spacing = np.median(np.diff(np.sort(unique_lons)))
                        lat_spacing = np.median(np.diff(np.sort(unique_lats)))
                        
                        # Convert to figure coordinates for marker size
                        # This ensures cells touch but don't overlap
                        ax_extent = ax.get_extent(crs=ccrs.PlateCarree())
                        ax_width = ax_extent[1] - ax_extent[0]
                        ax_height = ax_extent[3] - ax_extent[2]
                        
                        # Calculate marker size to fill grid cells
                        # Size in points^2, figure dimensions in data coordinates
                        marker_size = min(
                            (figsize[0] * 72 * lon_spacing / ax_width) ** 2,
                            (figsize[1] * 72 * lat_spacing / ax_height) ** 2
                        ) * 1.2  # 1.2 factor to ensure cells touch
                    else:
                        marker_size = 10
                    
                    # Plot as scatter with square markers (grid cells)
                    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
                    im = ax.scatter(
                        gridded_diff['lons'],
                        gridded_diff['lats'],
                        c=gridded_diff['values'],
                        cmap='bwr_r',
                        norm=norm,
                        s=marker_size,
                        marker='s',  # Square markers
                        edgecolors='none',
                        transform=ccrs.PlateCarree(),
                        alpha=0.8,
                        zorder=1
                    )
                else:
                    # Regular dense grid - use pcolormesh
                    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
                    im = ax.pcolormesh(
                        gridded_diff.columns.values,
                        gridded_diff.index.values,
                        gridded_diff.values,
                        cmap='bwr_r',
                        norm=norm,
                        shading='auto',
                        transform=ccrs.PlateCarree(),
                        alpha=0.8
                    )
                
                # Colorbar
                cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
                metric_label = performance_metric.upper() if performance_metric != 'r2' else 'R²'
                cbar.set_label(f"Δ {metric_label}")
        
        else:
            # Plot flowlines with metric colors
            from matplotlib.collections import LineCollection
            from matplotlib.colors import Normalize
            
            # Merge diff_metric with flowlines
            # Assume flowlines has COMID or similar ID matching location_labels
            diff_df = pd.DataFrame({
                'location_id': self.location_labels,
                'diff': diff_metric
            })
            
            # Try common ID columns
            id_col = None
            for col in ['COMID', 'comid', 'id', 'ID', 'feature_id', 'FEATUREID']:
                if col in flowline_gdf.columns:
                    id_col = col
                    break
            
            if id_col is None:
                print("Warning: Could not find ID column in flowline_gdf. Plotting without flowlines.")
            else:
                # Merge
                flowlines_copy = flowline_gdf.copy()
                flowlines_copy['location_id'] = flowlines_copy[id_col].astype(str)
                diff_df['location_id'] = diff_df['location_id'].astype(str)
                flowlines_diff = flowlines_copy.merge(diff_df, on='location_id', how='left')
                
                # Project flowlines
                proj = ccrs.PlateCarree()
                flowlines_proj = flowlines_diff.to_crs(proj.proj4_params)
                
                # Plot as LineCollection
                lines = LineCollection(
                    [np.array(geom.xy).T for geom in flowlines_proj.geometry],
                    linewidths=1.5,
                    alpha=0.8,
                    zorder=1
                )
                norm = Normalize(vmin=-0.5, vmax=0.5)
                lines.set_array(flowlines_proj['diff'])
                lines.set_cmap('bwr_r')
                lines.set_norm(norm)
                ax.add_collection(lines)
                
                # Colorbar
                cbar = plt.colorbar(lines, ax=ax, fraction=0.04, pad=0.02)
                metric_label = performance_metric.upper() if performance_metric != 'r2' else 'R²'
                cbar.set_label(f"Δ {metric_label}")
        
        # Plot baseline sensors
        ax.scatter(
            base_lons, base_lats,
            marker='o',
            color='k',
            edgecolors='none',
            s=5,
            alpha=0.5,
            label=f'Baseline (n={len(base_lats)})',
            transform=ccrs.PlateCarree(),
            zorder=4
        )
        
        # Plot optimal sensors
        ax.scatter(
            opt_lons, opt_lats,
            marker='o',
            color='green',
            s=10,
            edgecolors='white',
            linewidth=0.3,
            label=f'Reconfigured sensors',
            transform=ccrs.PlateCarree(),
            zorder=5
        )
        
        # Map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
        
        # Gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.3, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Legend
        ax.legend(loc='upper right', frameon=True, fontsize=9, framealpha=0.9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi, transparent=True)
            print(f"Figure saved to: {save_path}")
        
        return fig, ax
    
    def _get_coordinates(self, projection_epsg='EPSG:3857'):
        """
        Extract lat/lon coordinates from locations.
        
        Parameters
        ----------
        projection_epsg : str
            EPSG code for equal-area projection (default: 'EPSG:3857' for global)
            Common alternatives:
            - 'EPSG:5070': NAD83 Albers Equal Area (North America)
            - 'EPSG:3857': Web Mercator (global)
            - 'EPSG:6933': Equal Earth (global)
        """
        # Handle both Point and LineString geometries
        if self.locations.geometry.geom_type.iloc[0] != 'Point':
            # Convert to centroids using specified projection
            locations_proj = self.locations.to_crs(projection_epsg)
            centroids = locations_proj.geometry.centroid.to_crs(self.locations.crs)
            lons = centroids.x.values
            lats = centroids.y.values
        else:
            lons = self.locations.geometry.x.values
            lats = self.locations.geometry.y.values
        
        return lats, lons
    
    def _calculate_extent(self, lons, lats, padding=0.05):
        """Calculate map extent with padding."""
        lon_range = lons.max() - lons.min()
        lat_range = lats.max() - lats.min()
        
        extent = [
            lons.min() - lon_range * padding,
            lons.max() + lon_range * padding,
            lats.min() - lat_range * padding,
            lats.max() + lat_range * padding
        ]
        
        return extent
    
    def _create_grid(self, diff_metric, metric_name, grid_resolution=0.1, projection_epsg='EPSG:3857'):
        """
        Create gridded version of metric for background plotting.
        
        Parameters
        ----------
        diff_metric : array
            Metric differences for all locations
        metric_name : str
            Name of the metric
        grid_resolution : float
            Grid spacing in degrees
        projection_epsg : str
            EPSG code for equal-area projection (default: 'EPSG:3857')
        """
        # Need all locations (not just selected) for gridding
        # Check if we have access to all locations
        
        if not hasattr(self, 'all_locations') or self.all_locations is None:
            # Try to get from parent designer if available
            if hasattr(self, '_designer') and self._designer is not None:
                all_locs = self._designer.locations
            else:
                # No all_locations available - skip gridding
                print(f"Warning: Cannot create gridded background. all_locations not found.")
                print(f"         Modify designer.py to pass all_locations=self.locations when creating NetworkDesignResult.")
                return None
        else:
            all_locs = self.all_locations
        
        # Verify we have the right number of metric values
        if len(diff_metric) != len(all_locs):
            print(f"Warning: Metric array length ({len(diff_metric)}) doesn't match locations ({len(all_locs)}). Skipping gridded background.")
            return None
        
        # Get all coordinates
        if all_locs.geometry.geom_type.iloc[0] != 'Point':
            locs_proj = all_locs.to_crs(projection_epsg)
            centroids = locs_proj.geometry.centroid.to_crs(all_locs.crs)
            all_lons = centroids.x.values
            all_lats = centroids.y.values
        else:
            all_lons = all_locs.geometry.x.values
            all_lats = all_locs.geometry.y.values
        
        # Check if data is on a regular grid (even if sparse)
        unique_lats = np.unique(all_lats)
        unique_lons = np.unique(all_lons)
        n_unique_lats = len(unique_lats)
        n_unique_lons = len(unique_lons)
        
        print(f"Grid info: {n_unique_lats} unique lats × {n_unique_lons} unique lons = {n_unique_lats * n_unique_lons} possible cells")
        print(f"           {len(all_lats)} actual locations ({100*len(all_lats)/(n_unique_lats * n_unique_lons):.1f}% coverage)")
        
        # For sparse grids, return coordinates and values directly
        # This will be plotted with scatter() instead of pcolormesh()
        # to avoid interpolation artifacts
        
        return {
            'type': 'sparse',
            'lons': all_lons,
            'lats': all_lats,
            'values': diff_metric
        }