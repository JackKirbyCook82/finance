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

from support.calculations import Variable, Equation, Calculation, Calculator
from support.pipelines import Processor
from support.files import Files

from finance.variables import Securities, Strategies, Positions

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StrategyFile", "StrategyCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


strategy_index = {security: str for security in list(map(str, Securities))} | {"strategy": str, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
strategy_columns = {"spot": np.float32, "minimum": np.float32, "maximum": np.float32, "size": np.float32, "underlying": np.float32}


class StrategyFile(Files.Dataframe, variable="strategies", index=strategy_index, columns=strategy_columns):
    pass


class StrategyEquation(Equation):
    qo = Variable("size", function=lambda qα, qβ: np.minimum(qα, qβ))
    xo = Variable("underlying", function=lambda xα, xβ: (xα + xβ) / 2)
    wo = Variable("spot", function=lambda yo, ε: yo * 100 - ε)
    whτ = Variable("maximum", function=lambda yhτ, ε: yhτ * 100 - ε)
    wlτ = Variable("minimum", function=lambda ylτ, ε: ylτ * 100 - ε)

    αq = Variable("size", locator=Positions.LONG)
    αx = Variable("underlying", locator=Positions.LONG)
    αy = Variable("price", locator=Positions.LONG)
    αk = Variable("strike", locator=Positions.LONG)
    αq = Variable("size", locator=Positions.SHORT)
    αx = Variable("underlying", locator=Positions.SHORT)
    αy = Variable("price", locator=Positions.SHORT)
    αk = Variable("strike", locator=Positions.SHORT)
    ε = Variable("fees", locator="fees")

class VerticalEquation(StrategyEquation):
    yo = Variable("spot", function=lambda yα, yβ: - yα + yβ)

class CollarEquation(StrategyEquation):
    yo = Variable("spot", function=lambda yα, yβ, xo: - yα + yβ - xo)

class VerticalPutEquation(StrategyEquation):
    yhτ = Variable("maximum", function=lambda kα, kβ: np.maximum(kα - kβ, 0))
    ylτ = Variable("minimum", function=lambda kα, kβ: np.minimum(kα - kβ, 0))

class VerticalCallEquation(StrategyEquation):
    yhτ = Variable("maximum", function=lambda kα, kβ: np.maximum(-kα + kβ, 0))
    ylτ = Variable("minimum", function=lambda kα, kβ: np.minimum(-kα + kβ, 0))

class CollarLongEquation(StrategyEquation):
    yhτ = Variable("maximum", function=lambda kα, kβ: np.maximum(kα, kβ))
    ylτ = Variable("minimum", function=lambda kα, kβ: np.minimum(kα, kβ))

class CollarShortEquation(StrategyEquation):
    yhτ = Variable("maximum", function=lambda kα, kβ: np.maximum(-kα, -kβ))
    ylτ = Variable("minimum", function=lambda kα, kβ: np.minimum(-kα, -kβ))


class StrategyCalculation(Calculation, ABC, fields=["strategy"]):
    def execute(self, options, *args, fees, **kwargs):
        pass

class VerticalPutCalculation(StrategyCalculation, strategy=Strategies.Vertical.Put, equation=VerticalPutEquation): pass
class VerticalCallCalculation(StrategyCalculation, strategy=Strategies.Vertical.Call, equation=VerticalCallEquation): pass
class CollarLongCalculation(StrategyCalculation, strategy=Strategies.Collar.Long, equation=CollarLongEquation): pass
class CollarShortCalculation(StrategyCalculation, strategy=Strategies.Collar.Short, equation=CollarShortEquation): pass


class StrategyCalculator(Calculator, Processor, calculation=StrategyCalculation):
    def execute(self, contents, *args, **kwargs):
        options = contents["options"]
        assert isinstance(options, pd.DataFrame)
        if self.empty(options):
            return
        options = {option: dataset for option, dataset in self.options(options) if not self.empty(dataset["size"])}
        strategies = [strategy for strategy in self.calculate(options, *args, **kwargs)]
        strategies = xr.concat(strategies, dim="scenario")
        yield contents | dict(strategies=strategies)

    def calculate(self, options, *args, **kwargs):
        assert isinstance(options, dict)
        assert all([isinstance(option, xr.Dataset) for option in options.values()])
        for fields, calculation in self.calculations.items():
            if not all([option in options.keys() for option in list(fields["strategy"].options)]):
                continue
            variable = str(fields["strategy"]).lower()
            options = {option: options[option] for option in list(fields["strategy"].options)}
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



