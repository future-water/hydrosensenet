"""
hydrosensenet - Optimal sensor placement for hydrological monitoring networks

This package provides tools for optimal sensor placement in hydrological monitoring networks
using QR decomposition and risk-informed optimization approaches.

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

__version__ = "0.2.0"
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
    load_nri,
)

# Spatial operations
from .spatial import (
    calculate_spatial_weights,
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

# Bundled example data
from .datasets import load_example_basin

# ============================================================================
# LEGACY API (For backwards compatibility - will be removed in v1.0)
# ============================================================================
# The legacy modules pull in heavy optional dependencies (cartopy,
# matplotlib) at import time, so they are loaded lazily via PEP 562
# module __getattr__ instead of eagerly here.
_LEGACY_ATTRS = {
    "sensor_network_utils": "sensor_network_utils",
    "glofas_processing_utils": "glofas_processing_utils",
    "load_data": "sensor_network_utils",
    "prepare_usgs_indices": "sensor_network_utils",
}


def __getattr__(name):
    if name in _LEGACY_ATTRS:
        import importlib
        import warnings

        warnings.warn(
            f"{name} is deprecated and will be removed in v1.0. "
            "Use the new modular API or SensorNetworkDesigner instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        module_name = _LEGACY_ATTRS[name]
        try:
            module = importlib.import_module(f".{module_name}", __name__)
        except ImportError as e:
            raise ImportError(
                f"The legacy module {module_name!r} requires the optional "
                "visualization dependencies (matplotlib, cartopy). "
                "Install them with:\n"
                "  pip install 'hydrosensenet[viz]'"
            ) from e
        if name == module_name:
            return module
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "load_nri",

    # SPATIAL OPERATIONS
    "calculate_spatial_weights",

    # I/O UTILITIES
    "save_streamflow",
    "load_streamflow",
    "save_locations",
    "load_locations",
    "migrate_csv_to_parquet",
    "migrate_directory",

    # EXAMPLE DATA
    "load_example_basin",
]
