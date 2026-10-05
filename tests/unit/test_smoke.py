import pandas as pd

import mkopoguard


def test_package_imports():
    assert mkopoguard is not None


def test_pandas_works():
    df = pd.DataFrame({"amount": [1000, 2000]})
    assert df["amount"].sum() == 3000
