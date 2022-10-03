import numpy
import pandas
import pyarrow
import pyarrow.csv as csv

_DESTINATION = "./data"


def setup_pandas_examples():
    """Sets up the pandas examples."""

    small = _build_dataset(name="small_dataset", size=1000, n_cols=10)
    medium = _build_dataset(name="medium_dataset", size=100000, n_cols=10)
    large = _build_dataset(name="large_dataset", size=10000000, n_cols=10)

    return (
        pandas.DataFrame.from_dict(small),
        pandas.DataFrame.from_dict(medium),
        pandas.DataFrame.from_dict(large),
    )


def _build_dataset(name: str, size: int = 1000, n_cols: int = 10):
    """Builds the datasets used in the demos.

    Returns
    -------
    datasets : dict
        A dictionary of datasets used in the demos.
    """
    dataset = {f"column_{c}": numpy.random.rand(size) for c in range(1, 10)}

    table = pyarrow.Table.from_pydict(dataset)

    csv.write_csv(table, f"{_DESTINATION}/{name}.csv")

    return dataset
