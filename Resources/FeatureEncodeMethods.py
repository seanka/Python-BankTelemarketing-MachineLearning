import numpy as np
import pandas as pd


def change_education(x):
    return pd.DataFrame(
        np.where(x.isin(["basic.4y", "basic.6y", "basic.9y"]), "basic", x),
        columns=["education"],
        index=x.index,
    )


def change_job(x):
    return pd.DataFrame(
        np.where(x == "retired", "unemployed", x), columns=["job"], index=x.index
    )


def bin_pdays(x):
    return np.where(x < 30, 1, 0)
