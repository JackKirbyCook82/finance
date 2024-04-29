# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Strategy Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
import xarray as xr
from itertools import product
from collections import OrderedDict as ODict

from support.calculations import Calculation, Calculator
from support.pipelines import Processor
from support.files import Files

from finance.variables import Securities, Strategies

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StrategyFile", "StrategyCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


strategy_index = {security: str for security in list(map(str, Securities))} | {"strategy": str, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
strategy_columns = {"spot": np.float32, "minimum": np.float32, "maximum": np.float32, "size": np.float32, "underlying": np.float32}


class StrategyFile(Files.Dataframe, variable="strategies", index=strategy_index, columns=strategy_columns):
    pass


class StrategyCalculation(Calculation, fields=["strategy"]):
    def execute(self, options, *args, fees, **kwargs):
        assert all([isinstance(option, xr.Dataset) for option in options])


class VerticalPutCalculation(StrategyCalculation, strategy=Strategies.Vertical.Put, feed=Strategies.Vertical.Put.options): pass
class VerticalCallCalculation(StrategyCalculation, strategy=Strategies.Vertical.Call, feed=Strategies.Vertical.Call.options): pass
class CollarLongCalculation(StrategyCalculation, strategy=Strategies.Collar.Long, feed=Strategies.Collar.Long.options): pass
class CollarShortCalculation(StrategyCalculation, strategy=Strategies.Collar.Short, feed=Strategies.Collar.Short.options): pass


class StrategyCalculator(Calculator, Processor, calculations=ODict(list(StrategyCalculation)), title="Calculated"):
    def execute(self, contents, *args, **kwargs):
        options = contents["options"]
        assert isinstance(options, pd.DataFrame)
        if self.empty(options):
            return
        options = ODict([(security, dataset) for security, dataset in self.options(options) if not self.empty(dataset["size"])])
        if not bool(options):
            return
        strategies = [strategy for strategy in self.calculate(options, *args, **kwargs)]
        strategies = xr.concat(strategies, dim="scenario")
        yield contents | dict(strategies=strategies)

    def calculate(self, options, *args, **kwargs):
        assert isinstance(options, dict)
        assert all([isinstance(option, xr.Dataset) for option in options.values()])
        for strategy, calculation in self.calculations.items():
            if not all([security in options.keys() for security in list(strategy.options)]):
                continue
            variable = str(strategy).lower()
            dataset = calculation(options, *args, **kwargs)
            variables = {"strategy": xr.Variable("strategy", [variable]).squeeze("strategy")}
            dataset = dataset.assign_coords(variables)
            if not self.size(dataset["size"]):
                continue
            yield dataset

    @staticmethod
    def options(dataframe):
        assert isinstance(dataframe, pd.DataFrame)
        if bool(dataframe.empty):
            return
        datasets = xr.Dataset.from_dataframe(dataframe)
        datasets = datasets.squeeze("ticker").squeeze("expire").squeeze("date")
        for instrument, position in product(datasets["instrument"].values, datasets["position"].values):
            option = Securities[f"{instrument}|{position}"]
            dataset = datasets.sel({"instrument": instrument, "position": position})
            dataset = dataset.rename({"strike": str(option)})
            dataset["strike"] = dataset[str(option)]
            yield option, dataset



