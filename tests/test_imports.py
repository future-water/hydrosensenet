"""
Test that all major modules can be imported
"""
import pytest


def test_core_imports():
    """Test core module imports"""
    from hydrosensenet.core import algorithms, metrics
    assert algorithms is not None
    assert metrics is not None


def test_data_imports():
    """Test data module imports"""
    from hydrosensenet.data import loaders, preprocessors
    assert loaders is not None
    assert preprocessors is not None


def test_spatial_imports():
    """Test spatial module imports"""
    from hydrosensenet.spatial import weights
    assert weights is not None


def test_io_imports():
    """Test io module imports"""
    from hydrosensenet import io
    assert io is not None
