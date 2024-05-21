# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Strategy Objects
@author: Jack Kirby Cook

"""

import numpy as np
import xarray as xr
from abc import ABC
from itertools import product

from support.calculations import Variable, Equation, Calculation, Calculator
from support.query import Data, Header, Query
from support.files import Files

from finance.variables import Securities, Strategies, Positions

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StrategyFile", "StrategyCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


strategies_index = {option: str for option in list(map(str, Securities.Options))} | {"strategy": str, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
strategies_columns = {"spot": np.float32, "minimum": np.float32, "maximum": np.float32, "size": np.float32, "underlying": np.float32}
strategies_header = Header(xr.Dataset, index=list(strategies_index.keys()), columns=list(strategies_columns.keys()))


class StrategyFile(Files.Dataframe, variable="strategies", index=strategies_index, columns=strategies_columns):
    pass


class StrategyEquation(Equation):
    qo = Variable("qo", "size", np.float32, function=lambda qα, qβ: np.minimum(qα, qβ))
    xo = Variable("xo", "underlying", np.float32, function=lambda xα, xβ: (xα + xβ) / 2)
    wo = Variable("wo", "spot", np.float32, function=lambda yo, ε: yo * 100 - ε)
    whτ = Variable("whτ", "maximum", np.float32, function=lambda yhτ, ε: yhτ * 100 - ε)
    wlτ = Variable("wlτ", "minimum", np.float32, function=lambda ylτ, ε: ylτ * 100 - ε)
    qα = Variable("qα", "size", np.float32, position=Positions.LONG, locator="size")
    xα = Variable("xα", "underlying", np.float32, position=Positions.LONG, locator="underlying")
    yα = Variable("yα", "price", np.float32, position=Positions.LONG, locator="price")
    kα = Variable("kα", "strike", np.float32, position=Positions.LONG, locator="strike")
    qβ = Variable("qβ", "size", np.float32, position=Positions.SHORT, locator="size")
    xβ = Variable("xβ", "underlying", np.float32, position=Positions.SHORT, locator="underlying")
    yβ = Variable("yβ", "price", np.float32, position=Positions.SHORT, locator="price")
    kβ = Variable("kβ", "strike", np.float32, position=Positions.SHORT, locator="strike")
    ε = Variable("ε", "fees", np.float32, position="fees")

class VerticalEquation(StrategyEquation):
    yo = Variable("yo", "spot", np.float32, function=lambda yα, yβ: - yα + yβ)

class CollarEquation(StrategyEquation):
    yo = Variable("yo", "spot", np.float32, function=lambda yα, yβ, xo: - yα + yβ - xo)

class VerticalPutEquation(VerticalEquation):
    yhτ = Variable("yhτ", "maximum", np.float32, function=lambda kα, kβ: np.maximum(kα - kβ, 0))
    ylτ = Variable("ylτ", "minimum", np.float32, function=lambda kα, kβ: np.minimum(kα - kβ, 0))

class VerticalCallEquation(VerticalEquation):
    yhτ = Variable("yhτ", "maximum", np.float32, function=lambda kα, kβ: np.maximum(-kα + kβ, 0))
    ylτ = Variable("ylτ", "minimum", np.float32, function=lambda kα, kβ: np.minimum(-kα + kβ, 0))

class CollarLongEquation(CollarEquation):
    yhτ = Variable("yhτ", "maximum", np.float32, function=lambda kα, kβ: np.maximum(kα, kβ))
    ylτ = Variable("ylτ", "minimum", np.float32, function=lambda kα, kβ: np.minimum(kα, kβ))

class CollarShortEquation(CollarEquation):
    yhτ = Variable("yhτ", "maximum", np.float32, function=lambda kα, kβ: np.maximum(-kα, -kβ))
    ylτ = Variable("ylτ", "minimum", np.float32, function=lambda kα, kβ: np.minimum(-kα, -kβ))


class StrategyCalculation(Calculation, ABC, fields=["strategy"]):
    def execute(self, options, *args, fees, **kwargs):
        positions = [option.position for option in options.keys()]
        assert len(set(positions)) == len(list(positions))
        options = {str(option.position.name).lower(): dataset for option, dataset in options.items()}
        equation = self.equation(*args, **kwargs)
        yield equation.whτ(**options, fees=fees)
        yield equation.wlτ(**options, fees=fees)
        yield equation.wo(**options, fees=fees)
        yield equation.xo(**options, fees=fees)
        yield equation.qo(**options, fees=fees)

class VerticalPutCalculation(StrategyCalculation, strategy=Strategies.Vertical.Put, equation=VerticalPutEquation): pass
class VerticalCallCalculation(StrategyCalculation, strategy=Strategies.Vertical.Call, equation=VerticalCallEquation): pass
class CollarLongCalculation(StrategyCalculation, strategy=Strategies.Collar.Long, equation=CollarLongEquation): pass
class CollarShortCalculation(StrategyCalculation, strategy=Strategies.Collar.Short, equation=CollarShortEquation): pass


class StrategyCalculator(Data, Calculator, calculations=StrategyCalculation):
    @Query("options", strategies=strategies_header)
    def execute(self, options, *args, **kwargs):
        assert isinstance(options, dict) and all([isinstance(dataset, xr.Dataset) for dataset in options.values()])
        options = {option: dataset for option, dataset in self.options(options) if not self.empty(dataset["size"])}
        strategies = list(self.calculate(options, *args, **kwargs))
        yield dict(strategies=strategies)

    def calculate(self, options, *args, **kwargs):
        function = lambda key, value: {key: xr.Variable(key, [value]).squeeze(key)}
        for variables, calculation in self.calculations.items():
            if not all([option in options.keys() for option in list(variables["strategy"].options)]):
                continue
            variable = str(variables["strategy"]).lower()
            variables = function("strategy", variable)
            datasets = {option: options[option] for option in list(variables["strategy"].options)}
            results = calculation(datasets, *args, **kwargs)
            results = results.assign_coords(variables)
            yield results

    @staticmethod
    def options(dataframe):
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




