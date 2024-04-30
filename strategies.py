# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Strategy Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC
from itertools import product
from collections import OrderedDict as ODict

from support.calculations import Variable, Equation, Calculation, Calculator
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
strategy_domains = [{variable + position: name for variable, name in {"q": "size", "x": "underlying", "y": "price", "k": "strike"}.items()} for position in ("α", "β")]


class StrategyFile(Files.Dataframe, variable="strategies", index=strategy_index, columns=strategy_columns):
    pass


class StrategyEquation(Equation):
    qo = Variable(np.int32, lambda qα, qβ: np.minimum(qα, qβ))
    xo = Variable(np.float32, lambda xα, xβ: (xα + xβ) / 2)
    wo = Variable(np.float32, lambda yo, ε: yo * 100 - ε)
    whτ = Variable(np.float32, lambda yhτ, ε: yhτ * 100 - ε)
    wlτ = Variable(np.float32, lambda ylτ, ε: ylτ * 100 - ε)

class VerticalEquation(StrategyEquation):
    yo = Variable(np.float32, lambda yα, yβ: - yα + yβ)

class CollarEquation(StrategyEquation):
    yo = Variable(np.float32, lambda yα, yβ, xo: - yα + yβ - xo)

class VerticalPutEquation(StrategyEquation):
    yhτ = Variable(np.float32, lambda kα, kβ: np.maximum(kα - kβ, 0))
    ylτ = Variable(np.float32, lambda kα, kβ: np.minimum(kα - kβ, 0))

class VerticalCallEquation(StrategyEquation):
    yhτ = Variable(np.float32, lambda kα, kβ: np.maximum(-kα + kβ, 0))
    ylτ = Variable(np.float32, lambda kα, kβ: np.minimum(-kα + kβ, 0))

class CollarLongEquation(StrategyEquation):
    yhτ = Variable(np.float32, lambda kα, kβ: np.maximum(kα, kβ))
    ylτ = Variable(np.float32, lambda kα, kβ: np.minimum(kα, kβ))

class CollarShortEquation(StrategyEquation):
    yhτ = Variable(np.float32, lambda kα, kβ: np.maximum(-kα, -kβ))
    ylτ = Variable(np.float32, lambda kα, kβ: np.minimum(-kα, -kβ))


class StrategyCalculation(Calculation, ABC, fields=["strategy"], domain=strategy_domains):
    def execute(self, options, *args, fees, **kwargs):
        assert all([isinstance(option, xr.Dataset) for option in options])
        long, short = [options[option] for option in self.strategy.options]
        domain = self.domain(long, short) | {"ε": fees}
        yield self.equation.yhτ(**domain).to_dataset(name="maximum")
        yield self.equation.ylτ(**domain).to_dataset(name="minimum")
        yield self.equation.yo(**domain).to_dataset(name="spot")
        yield self.equation.xo(**domain).to_dataset(name="underlying")
        yield self.equation.qo(**domain).to_dataset(name="size")

class VerticalPutCalculation(StrategyCalculation, strategy=Strategies.Vertical.Put, equation=VerticalPutEquation): pass
class VerticalCallCalculation(StrategyCalculation, strategy=Strategies.Vertical.Call, equation=VerticalCallEquation): pass
class CollarLongCalculation(StrategyCalculation, strategy=Strategies.Collar.Long, equation=CollarLongEquation): pass
class CollarShortCalculation(StrategyCalculation, strategy=Strategies.Collar.Short, equation=CollarShortEquation): pass


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



