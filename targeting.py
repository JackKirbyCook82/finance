# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 27 2023
@name:   Targeting Objects
@author: Jack Kirby Cook

"""

import os.path
import numpy as np
import xarray as xr
import pandas as pd
import dask.dataframe as dk

from support.pipelines import Processor, Loader, Saver

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TargetingProcessor", "TargetingSaver", "TargetingLoader"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


class TargetingProcessor(Processor):
    def execute(self, contents, *args, apy, **kwargs):
        ticker, expire, strategy, securities, content = contents
        assert isinstance(content, xr.Dataset)
        targets = content.to_dask_dataframe() if bool(content.chunks) else content.to_dataframe()

        print(content)
        print(targets)
        raise Exception()

        targets = targets.where(targets["apy"] >= apy)
        targets = targets.dropna(how="all")
        yield ticker, expire, strategy, securities, targets


class TargetingSaver(Saver):
    def execute(self, contents, *args, **kwargs):
        ticker, expire, strategy, securities, content = contents
        assert isinstance(content, (pd.DataFrame, dk.DataFrame))
        file = str(strategy).replace("|", "_")
        file = os.path.join(self.repository, str(file) + ".csv")
        self.write(content, file=file, mode="a")


class TargetingLoader(Loader):
    def execute(self, strategy, *args, **kwargs):
        file = str(strategy).replace("|", "_")
        file = os.path.join(self.repository, str(file) + ".csv")
        datatypes = {"ticker": str, "tau": np.int16, "spot": np.float32, "value": np.float32, "cost": np.float32, "apy": np.float32}
        dataframe = self.read(pd.DataFrame, file=file, datatypes=datatypes, datetypes=["date", "expire"])
        yield strategy, dataframe



