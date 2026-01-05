"""
hydrosensenet - Optimal sensor placement for hydrological monitoring networks

This package provides tools for optimal sensor placement in hydrological monitoring networks
using QR decomposition and risk-informed optimization approaches.

Reference:
Oh, J., Lee, J., Bartos, M. Scalable, adaptive and risk-informed design of hydrological
sensor networks. Nat Water (2025). https://doi.org/10.1038/s44221-025-00496-7

Quick Start
-----------
>>> from hydrosensenet import SensorNetworkDesigner
>>> # Works with any format - CSV, Parquet, Excel, NetCDF, etc.
>>> # Parquet recommended for 5-10x better performance!
>>> designer = SensorNetworkDesigner.from_csv("streamflow.parquet", "gauges.parquet")
>>> result = designer.design_network(n_sensors=50)
>>> result.print_summary()
>>> result.export("sensor_locations.csv")
"""

__version__ = "0.1.1"
__author__ = "Jeil Oh, John Lee, Matthew Bartos"

# ============================================================================
# HIGH-LEVEL API (Recommended for most users)
# ============================================================================
from .designer import SensorNetworkDesigner, NetworkDesignResult

# ============================================================================
# MODULAR API (For advanced users who need fine control)
# ============================================================================

# Core algorithms
from .core import (
    sensor_placement_qr,
    qr_pivot_selection,
    calculate_performance_metrics,
    reconstruction_evaluation,
)

# Data loading and preprocessing
from .data import (
    load_streamflow_data,
    prepare_gauge_locations,
    split_timeseries,
    prepare_matrix,
    filter_valid_data,
)

# Spatial operations
from .spatial import (
    calculate_spatial_weights,
    load_risk_data,
)

# I/O utilities (optimized for Parquet)
from .io import (
    save_streamflow,
    load_streamflow,
    save_locations,
    load_locations,
    migrate_csv_to_parquet,
    migrate_directory,
)

# ============================================================================
# LEGACY API (For backwards compatibility - will be deprecated in v1.0)
# ============================================================================
import warnings

def _deprecated_import(name):
    """Helper to show deprecation warnings."""
    warnings.warn(
        f"{name} is deprecated and will be removed in v1.0. "
        f"Use the new modular API or SensorNetworkDesigner instead.",
        DeprecationWarning,
        stacklevel=3
    )

# Legacy imports from old modules (with deprecation warnings)
try:
    from . import sensor_network_utils as _legacy_snu
    from . import glofas_processing_utils as _legacy_gpu

    # Create wrapper functions that show deprecation warnings
    def load_data(*args, **kwargs):
        _deprecated_import("load_data")
        return _legacy_snu.load_data(*args, **kwargs)

    def prepare_usgs_indices(*args, **kwargs):
        _deprecated_import("prepare_usgs_indices")
        return _legacy_snu.prepare_usgs_indices(*args, **kwargs)

    # Add other legacy functions as needed...

except ImportError:
    # Legacy modules not available - that's fine
    pass

# ============================================================================
# PUBLIC API
# ============================================================================
__all__ = [
    # HIGH-LEVEL API (Recommended)
    "SensorNetworkDesigner",
    "NetworkDesignResult",

    # CORE ALGORITHMS
    "sensor_placement_qr",
    "qr_pivot_selection",
    "calculate_performance_metrics",
    "reconstruction_evaluation",

    # DATA LOADING
    "load_streamflow_data",
    "prepare_gauge_locations",
    "split_timeseries",
    "prepare_matrix",
    "filter_valid_data",

    # SPATIAL OPERATIONS
    "calculate_spatial_weights",
    "load_risk_data",

    # I/O UTILITIES
    "save_streamflow",
    "load_streamflow",
    "save_locations",
    "load_locations",
    "migrate_csv_to_parquet",
    "migrate_directory",
]
