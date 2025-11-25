import sys

# Assures Python 3.10 or above
assert sys.version_info >= (3, 10)

from packaging.version import Version
import sklearn
# Assures Scikit-Learn version 1.6.1 or above
assert Version(sklearn.__version__) >= Version("1.6.1")

from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

# Function for loading housing data into DataFrame
def load_housing_data() -> pd.DataFrame:
    # Looks for datasets/housing.tgz
    tarball_path = Path("datasets/housing.tgz")
    # If file is not found, creates directory and copies file from GitHub
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets", filter="data")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

# Function to add income categories to DataFrame
def add_income_cat(housing: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with an income_cat column added."""
    housing = housing.copy()
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0., 1.5, 3., 4.5, 6., float("inf")],
        labels=[1, 2, 3, 4, 5],
    )
    return housing
