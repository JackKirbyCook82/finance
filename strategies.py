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

from finance.variables import Contract, Securities, Strategies, Positions
from support.calculations import Variable, Equation, Calculation
from support.files import FileDirectory, FileQuery, FileData
from support.pipelines import Processor

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StrategyFile", "StrategyCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


securities_index = {option: str for option in list(map(str, Securities))}
strategies_index = {"strategy": str, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
strategies_columns = {"spot": np.float32, "minimum": np.float32, "maximum": np.float32, "size": np.float32, "underlying": np.float32}
strategies_data = FileData.Dataframe(header=securities_index | strategies_index | strategies_columns)
contract_query = FileQuery("contract", Contract.tostring, Contract.fromstring)


class StrategyFile(FileDirectory, variable="strategies", query=contract_query, data=strategies_data):
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


class StrategyCalculator(Processor):
    def __init__(self, *args, strategies=[], name=None, **kwargs):
        assert isinstance(strategies, list) and all([strategy in list(Strategies) for strategy in strategies])
        super().__init__(*args, name=name, **kwargs)
        calculations = {variables["strategy"]: calculation for variables, calculation in ODict(list(StrategyCalculation)).items() if variables["strategy"] in strategies}
        self.__calculations = {str(strategy).lower(): calculation(*args, **kwargs) for strategy, calculation in calculations.items()}
        self.__variables = lambda **mapping: {key: xr.Variable(key, [value]).squeeze(key) for key, value in mapping.items()}

    def execute(self, contents, *args, **kwargs):
        options = contents["options"]
        assert isinstance(options, pd.DataFrame)
        options = ODict(list(self.options(options, *args, **kwargs)))
        strategies = ODict(list(self.calculate(options, *args, **kwargs)))
        strategies = {"strategies": list(strategies.values())}
        yield contents | strategies

    def calculate(self, options, *args, **kwargs):
        for strategy, calculation in self.calculations.items():
            variables = self.variables(strategy=strategy)
            if not all([option in options.keys() for option in list(Strategies[strategy].options)]):
                continue
            datasets = {option: options[option] for option in list(Strategies[strategy].options)}
            dataset = calculation(datasets, *args, **kwargs)
            dataset = dataset.assign_coords(variables)
            yield strategy, dataset

    @staticmethod
    def options(dataframe, *args, **kwargs):
        if bool(dataframe.empty):
            return
        empty = lambda dataarray: not bool(np.count_nonzero(~np.isnan(dataarray.values)))
        datasets = xr.Dataset.from_dataframe(dataframe)
        datasets = datasets.squeeze("ticker").squeeze("expire").squeeze("date")
        for instrument, position in product(datasets["instrument"].values, datasets["position"].values):
            option = Securities[f"{instrument}|{position}"]
            dataset = datasets.sel({"instrument": instrument, "position": position})
            if empty(dataset["size"]):
                continue
            dataset = dataset.rename({"strike": str(option)})
            dataset["strike"] = dataset[str(option)]
            yield option, dataset

    @property
    def calculations(self): return self.__calculations
    @property
    def variables(self): return self.__variables



