"""
Basic tests for SensorNetworkDesigner
"""
import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point


def test_imports():
    """Test that main classes can be imported"""
    from hydrosensenet import SensorNetworkDesigner
    assert SensorNetworkDesigner is not None


def test_designer_initialization():
    """Test basic initialization of SensorNetworkDesigner"""
    from hydrosensenet import SensorNetworkDesigner

    # Create minimal test data
    n_locations = 10
    n_timesteps = 100

    # Random streamflow data
    streamflow_data = np.random.rand(n_timesteps, n_locations)

    # Create simple location data
    locations_gdf = gpd.GeoDataFrame(
        {'id': range(n_locations)},
        geometry=[Point(i, i) for i in range(n_locations)],
        crs="EPSG:4326"
    )

    location_labels = [f"gauge_{i}" for i in range(n_locations)]

    # Initialize designer
    designer = SensorNetworkDesigner(
        streamflow_data=streamflow_data,
        locations=locations_gdf,
        location_labels=location_labels
    )

    assert designer is not None
    assert designer.streamflow_data.shape == (n_timesteps, n_locations)


def test_designer_network_design():
    """Test that network design runs without errors"""
    from hydrosensenet import SensorNetworkDesigner

    n_locations = 10
    n_timesteps = 100
    n_sensors = 5

    streamflow_data = np.random.rand(n_timesteps, n_locations)

    locations_gdf = gpd.GeoDataFrame(
        {'id': range(n_locations)},
        geometry=[Point(i, i) for i in range(n_locations)],
        crs="EPSG:4326"
    )

    location_labels = [f"gauge_{i}" for i in range(n_locations)]

    designer = SensorNetworkDesigner(
        streamflow_data=streamflow_data,
        locations=locations_gdf,
        location_labels=location_labels
    )

    # Design network
    result = designer.design_network(n_sensors=n_sensors, evaluate=False)

    assert result is not None
    assert len(result.selected_indices) == n_sensors
