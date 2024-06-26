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

from finance.variables import Variables, Strategies
from support.calculations import Variable, Equation, Calculation
from support.pipelines import Processor

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StrategyCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class StrategyEquation(Equation):
    ti = Variable("ti", "current", np.datetime64, function=lambda tα, tβ: np.minimum(np.datetime64(tα, "ns"), np.datetime64(tβ, "ns")))
    qi = Variable("qi", "size", np.float32, function=lambda qα, qβ: np.minimum(qα, qβ))
    xi = Variable("xi", "underlying", np.float32, function=lambda xα, xβ: (xα + xβ) / 2)
    wi = Variable("wi", "spot", np.float32, function=lambda yi, ε: yi * 100 - ε)
    whτ = Variable("whτ", "maximum", np.float32, function=lambda yhτ, ε: yhτ * 100 - ε)
    wlτ = Variable("wlτ", "minimum", np.float32, function=lambda ylτ, ε: ylτ * 100 - ε)
    tα = Variable("tα", "current", np.datetime64, position=Variables.Positions.LONG, locator="current")
    qα = Variable("qα", "size", np.float32, position=Variables.Positions.LONG, locator="size")
    xα = Variable("xα", "underlying", np.float32, position=Variables.Positions.LONG, locator="underlying")
    yα = Variable("yα", "price", np.float32, position=Variables.Positions.LONG, locator="price")
    kα = Variable("kα", "strike", np.float32, position=Variables.Positions.LONG, locator="strike")
    tβ = Variable("tβ", "current", np.datetime64, position=Variables.Positions.SHORT, locator="current")
    qβ = Variable("qβ", "size", np.float32, position=Variables.Positions.SHORT, locator="size")
    xβ = Variable("xβ", "underlying", np.float32, position=Variables.Positions.SHORT, locator="underlying")
    yβ = Variable("yβ", "price", np.float32, position=Variables.Positions.SHORT, locator="price")
    kβ = Variable("kβ", "strike", np.float32, position=Variables.Positions.SHORT, locator="strike")
    ε = Variable("ε", "fees", np.float32, position="fees")

class VerticalEquation(StrategyEquation):
    yi = Variable("yi", "spot", np.float32, function=lambda yα, yβ: - yα + yβ)

class CollarEquation(StrategyEquation):
    yi = Variable("yi", "spot", np.float32, function=lambda yα, yβ, xi: - yα + yβ - xi)

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
        options = {str(option): dataset for option, dataset in options.items()}
        equation = self.equation(*args, **kwargs)
        yield equation.whτ(**options, fees=fees)
        yield equation.wlτ(**options, fees=fees)
        yield equation.wi(**options, fees=fees)
        yield equation.xi(**options, fees=fees)
        yield equation.qi(**options, fees=fees)
        yield equation.ti(**options, fees=fees)

class VerticalPutCalculation(StrategyCalculation, strategy=Strategies.Vertical.Put, equation=VerticalPutEquation): pass
class VerticalCallCalculation(StrategyCalculation, strategy=Strategies.Vertical.Call, equation=VerticalCallEquation): pass
class CollarLongCalculation(StrategyCalculation, strategy=Strategies.Collar.Long, equation=CollarLongEquation): pass
class CollarShortCalculation(StrategyCalculation, strategy=Strategies.Collar.Short, equation=CollarShortEquation): pass


class StrategyCalculator(Processor):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        calculations = {variables["strategy"]: calculation for variables, calculation in ODict(list(StrategyCalculation)).items()}
        self.__calculations = {strategy: calculation(*args, **kwargs) for strategy, calculation in calculations.items()}
        self.__variables = lambda **mapping: {key: xr.Variable(key, [value]).squeeze(key) for key, value in mapping.items()}

    def execute(self, contents, *args, **kwargs):
        options = contents[Variables.Instruments.OPTION]
        assert isinstance(options, pd.DataFrame)
        options = ODict(list(self.options(options, *args, **kwargs)))
        strategies = ODict(list(self.calculate(options, *args, **kwargs)))
        strategies = list(strategies.values())
        if not bool(strategies):
            return
        strategies = {Variables.Datasets.STRATEGY: strategies}
        yield contents | strategies

    def calculate(self, options, *args, **kwargs):
        for strategy, calculation in self.calculations.items():
            if not all([option in options.keys() for option in list(strategy.options)]):
                continue
            variables = self.variables(strategy=str(strategy))
            datasets = {option: options[option] for option in list(strategy.options)}
            dataset = calculation(datasets, *args, **kwargs)
            if self.empty(dataset["size"]):
                continue
            dataset = dataset.assign_coords(variables)
            yield strategy, dataset

    def options(self, dataframe, *args, **kwargs):
        if bool(dataframe.empty):
            return
        dataframe = dataframe.set_index(["ticker", "expire", "strike", "instrument", "option", "position"], drop=True, inplace=False)
        datasets = xr.Dataset.from_dataframe(dataframe)
        datasets = datasets.squeeze("ticker").squeeze("expire")
        for instrument, option, position in product(datasets["instrument"].values, datasets["option"].values, datasets["position"].values):
            dataset = datasets.sel(indexers={"instrument": instrument, "option": option, "position": position})
            dataset = dataset.drop_vars(["instrument", "option", "position"], errors="ignore")
            if self.empty(dataset["size"]):
                continue
            axis = f"{str(instrument)}|{str(option)}|{str(position)}"
            dataset = dataset.rename({"strike": str(axis)})
            dataset["strike"] = dataset[str(axis)]
            yield axis, dataset

    @staticmethod
    def empty(dataarray): return not bool(np.count_nonzero(~np.isnan(dataarray.values)))
    @property
    def calculations(self): return self.__calculations
    @property
    def variables(self): return self.__variables



