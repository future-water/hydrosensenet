"""Tests for the bundled example datasets."""
import hashlib

import geopandas as gpd
import pandas as pd
import pytest

from hydrosensenet import datasets
from hydrosensenet.datasets import load_example_basin


def test_load_example_basin_from_checkout():
    """In a repo checkout the bundled files load directly, aligned."""
    streamflow, locations = load_example_basin()

    assert isinstance(streamflow, pd.DataFrame)
    assert isinstance(locations, gpd.GeoDataFrame)
    assert len(locations) == streamflow.shape[1]
    # rows follow column order
    assert [str(c) for c in streamflow.columns] == [
        str(c) for c in locations["comid"]
    ]
    assert streamflow.index.inferred_type in ("datetime64", "datetime")
    assert (streamflow.notna().mean() > 0.9).all()


def test_fetch_checksum_mismatch_removes_file(tmp_path, monkeypatch):
    bad = tmp_path / "sample.bin"
    bad.write_bytes(b"corrupted")
    monkeypatch.setattr(datasets, "_FILES", {"sample": ("sample.bin", "0" * 64)})
    monkeypatch.setattr(datasets, "_repo_example_dir", lambda: None)

    with pytest.raises(ValueError, match="Checksum mismatch"):
        datasets._fetch("sample", cache_dir=tmp_path)
    assert not bad.exists()


def test_fetch_downloads_once_and_caches(tmp_path, monkeypatch):
    payload = b"example-bytes"
    sha = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(datasets, "_FILES", {"sample": ("sample.bin", sha)})
    monkeypatch.setattr(datasets, "_repo_example_dir", lambda: None)

    calls = []

    class FakeResponse:
        content = payload

        @staticmethod
        def raise_for_status():
            pass

    def fake_get(url, timeout):
        calls.append(url)
        return FakeResponse()

    import requests
    monkeypatch.setattr(requests, "get", fake_get)

    first = datasets._fetch("sample", cache_dir=tmp_path)
    second = datasets._fetch("sample", cache_dir=tmp_path)

    assert first == second == tmp_path / "sample.bin"
    assert len(calls) == 1  # second call hit the cache
    assert first.read_bytes() == payload
