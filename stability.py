# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 2025
@name:   Stability Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
import xarray as xr
from functools import reduce

from finance.variables import Querys, Variables, Securities
from support.calculations import Calculation, Equation, Variable
from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StabilityCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class StabilityEquation(Equation):
    y = Variable("y", "value", np.float32, xr.DataArray, vectorize=True, function=lambda q, Θ, Φ, Ω, Δ: q * Θ * Φ * Δ * (1 - Θ * Ω) / 2)
    m = Variable("m", "trend", np.float32, xr.DataArray, vectorize=True, function=lambda q, Θ, Φ, Ω: q * Θ * Φ * Ω)
    Ω = Variable("Ω", "omega", np.int32, xr.DataArray, vectorize=True, function=lambda x, k: np.sign(x / k - 1))
    Δ = Variable("Δ", "delta", np.int32, xr.DataArray, vectorize=True, function=lambda x, k: np.subtract(x, k))
    Θ = Variable("Θ", "theta", np.int32, xr.DataArray, vectorize=True, function=lambda i: int(i))
    Φ = Variable("Φ", "phi", np.int32, xr.DataArray, vectorize=True, function=lambda j: int(j))

    Σy = Variable("Σy", "value", np.float32, xr.DataArray, vectorize=False, function=lambda y: y.sum("holdings").drop(list(Querys.Settlement)))
    Σm = Variable("Σm", "value", np.float32, xr.DataArray, vectorize=False, function=lambda m: m.sum("holdings").drop(list(Querys.Settlement)))

    Σyh = Variable("Σyh", "maximum", np.float32, xr.DataArray, vectorize=False, function=lambda Σy: Σy.max(dim="underlying"))
    Σyl = Variable("Σyl", "minimum", np.float32, xr.DataArray, vectorize=False, function=lambda Σy: Σy.min(dim="underlying"))
    Σmh = Variable("Σmh", "bull", np.float32, xr.DataArray, vectorize=False, function=lambda Σm: Σm.isel(underlying=0).drop_vars(["underlying"]))
    Σml = Variable("Σml", "bear", np.float32, xr.DataArray, vectorize=False, function=lambda Σm: Σm.isel(underlying=-1).drop_vars(["underlying"]))

    x = Variable("x", "underlying", np.float32, xr.DataArray, locator="underlying")
    q = Variable("q", "exposure", np.int32, xr.DataArray, locator="exposure")
    i = Variable("i", "option", Variables.Securities.Option, xr.DataArray, locator="option")
    j = Variable("j", "position", Variables.Securities.Position, xr.DataArray, locator="position")
    k = Variable("k", "strike", np.float32, xr.DataArray, locator="strike")


class StabilityCalculation(Calculation, equation=StabilityEquation):
    def execute(self, portfolios, *args, **kwargs):
        with self.equation(portfolios) as equation:
            equation.y(), equation.m()
            yield equation.Σyh()
            yield equation.Σyl()
            yield equation.Σmh()
            yield equation.Σml()


class StabilityCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__calculation = StabilityCalculation(*args, **kwargs)

    def execute(self, valuations, options, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame) and isinstance(options, pd.DataFrame)
        if self.empty(valuations): return
        for settlement, primary in self.partition(valuations, by=Querys.Settlement):
            secondary = self.alignment(options, by=settlement)
            valuations = self.calculate(primary, secondary, *args, **kwargs)
            size = self.size(valuations)
            self.console(f"{str(settlement)}[{int(size):.0f}]")
            if self.empty(valuations): continue
            yield valuations

    def calculate(self, valuations, options, *args, **kwargs):
        valuations = valuations.assign(portfolio=list(range(1, len(valuations) + 1)))
        orders = self.orders(valuations, *args, **kwargs)
        exposure = self.exposure(options, *args, **kwargs)
        portfolios = self.portfolios(orders, exposure, *args, **kwargs)
        stability = self.calculation(portfolios, *args, **kwargs)
        stability = stability.to_dataframe()
        stable = (stability["bear"] == 0) & (stability["bull"] == 0)
        valuations = valuations.set_index("portfolio", drop=True, inplace=False)
        valuations = valuations.where(stable).dropna(how="all", inplace=False)
        valuations = valuations.reset_index(drop=True, inplace=False)
        return valuations

    @staticmethod
    def orders(valuations, *args, **kwargs):
        parameters = dict(id_vars=list(Querys.Settlement) + ["portfolio"], value_name="strike", var_name="security")
        function = lambda security: pd.Series(dict(Securities[security].items()))
        header = list(Querys.Settlement) + list(map(str, Securities.Options)) + ["portfolio"]
        valuations = pd.melt(valuations[header], **parameters)
        valuations = pd.concat([valuations, valuations["security"].apply(function)], axis=1)
        valuations = valuations.dropna(subset="strike", how="all", inplace=False)
        index = list(Querys.Settlement) + list(Variables.Securities.Security) + ["strike", "portfolio"]
        valuations = valuations[index].assign(exposure=1)
        valuations = valuations.set_index(index, drop=True, inplace=False)["exposure"].squeeze()
        valuations = xr.DataArray.from_series(valuations).fillna(0)
        function = lambda content, axis: content.squeeze(axis)
        valuations = reduce(function, list(Querys.Settlement), valuations)
        valuations = valuations.stack(holdings=list(Variables.Securities.Security) + ["strike"])
        return valuations

    @staticmethod
    def exposure(options, *args, **kwargs):
        index = list(Querys.Settlement) + list(Variables.Securities.Security) + ["strike"]
        options = options.set_index(index, drop=True, inplace=False)["exposure"].squeeze()
        options = options.where(options > 0).dropna(how="all", inplace=False)
        options = xr.DataArray.from_series(options).fillna(0)
        function = lambda content, axis: content.squeeze(axis)
        options = reduce(function, list(Querys.Settlement), options)
        options = options.stack(holdings=list(Variables.Securities.Security) + ["strike"])
        return options

    @staticmethod
    def portfolios(orders, exposures, *args, **kwargs):
        portfolios = sum(xr.align(orders, exposures, fill_value=0, join="outer")).to_dataset()
        underlying = np.unique(portfolios["strike"].values)
        underlying = np.sort(np.concatenate([underlying - 0.001, underlying + 0.001]))
        portfolios["underlying"] = underlying
        return portfolios

    @staticmethod
    def alignment(dataframe, *args, by, **kwargs):
        mask = [dataframe[key] == value for key, value in iter(by)]
        mask = reduce(lambda lead, lag: lead & lag, list(mask))
        return dataframe.where(mask)

    @property
    def calculation(self): return self.__calculation



