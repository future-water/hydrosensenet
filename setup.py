from setuptools import setup, find_packages

setup(
    name="hydrosensenet",
    version="0.1.1",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "xarray>=0.19.0",
        "geopandas>=0.10.0",
        "cartopy>=0.20.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "shapely>=1.8.0",
        "rioxarray>=0.8.0",
        "pyarrow>=10.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "sphinx>=4.0.0",
        ],
        "nwm": [
            "fsspec>=2021.11.0",
            "dask[distributed]>=2021.11.0",
            "pynhd>=0.14.0",
            "s3fs>=2021.11.0",
            "zarr>=2.10.0",
        ],
    },
)
