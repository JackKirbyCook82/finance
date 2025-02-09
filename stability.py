# -*- coding: utf-8 -*-
"""
Created on Fri May 17 2024
@name:   Stability Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
import xarray as xr
from functools import reduce
from collections import OrderedDict as ODict

from finance.variables import Variables, Querys
from support.calculations import Calculation, Equation, Variable
from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StabilityCalculator", "StabilityFilter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class StabilityEquation(Equation):
    y = Variable("y", "value", np.float32, xr.DataArray, vectorize=True, function=lambda q, Θ, Φ, Ω, Δ: q * Θ * Φ * Δ * (1 - Θ * Ω) / 2)
    m = Variable("m", "trend", np.float32, xr.DataArray, vectorize=True, function=lambda q, Θ, Φ, Ω: q * Θ * Φ * Ω)
    Ω = Variable("Ω", "omega", np.int32, xr.DataArray, vectorize=True, function=lambda x, k: np.sign(x / k - 1))
    Δ = Variable("Δ", "delta", np.int32, xr.DataArray, vectorize=True, function=lambda x, k: np.subtract(x, k))
    Θ = Variable("Θ", "theta", np.int32, xr.DataArray, vectorize=True, function=lambda i: + int(Variables.Greeks.Theta(str(i))))
    Φ = Variable("Φ", "phi", np.int32, xr.DataArray, vectorize=True, function=lambda j: + int(Variables.Greeks.Phi(str(j))))

    Σy = Variable("Σy", "value", np.float32, xr.DataArray, vectorize=False, function=lambda y: y.sum("holdings").drop(list(Querys.Settlement)))
    Σm = Variable("Σm", "value", np.float32, xr.DataArray, vectorize=False, function=lambda m: m.sum("holdings").drop(list(Querys.Settlement)))

    Σyh = Variable("Σyh", "maximum", np.float32, xr.DataArray, vectorize=False, function=lambda Σy: Σy.max(dim="underlying"))
    Σyl = Variable("Σyl", "minimum", np.float32, xr.DataArray, vectorize=False, function=lambda Σy: Σy.min(dim="underlying"))
    Σmh = Variable("Σmh", "bull", np.float32, xr.DataArray, vectorize=False, function=lambda Σm: Σm.isel(underlying=0).drop_vars(["underlying"]))
    Σml = Variable("Σml", "bear", np.float32, xr.DataArray, vectorize=False, function=lambda Σm: Σm.isel(underlying=-1).drop_vars(["underlying"]))

    x = Variable("x", "underlying", np.float32, xr.DataArray, locator="underlying")
    q = Variable("q", "quantity", np.int32, xr.DataArray, locator="quantity")
    i = Variable("i", "option", Variables.Securities.Option, xr.DataArray, locator="option")
    j = Variable("j", "position", Variables.Securities.Position, xr.DataArray, locator="position")
    k = Variable("k", "strike", np.float32, xr.DataArray, locator="strike")


class StabilityCalculation(Calculation, equation=StabilityEquation):
    def execute(self, portfolios, *args, **kwargs):
        with self.equation(portfolios) as equation:
            yield equation.Σyh(portfolios)
            yield equation.Σyl(portfolios)
            yield equation.Σmh(portfolios)
            yield equation.Σml(portfolios)


class StabilityCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__calculation = StabilityCalculation(*args, **kwargs)

    def execute(self, orders, exposures, *args, **kwargs):
        assert isinstance(orders, pd.DataFrame) and isinstance(exposures, pd.DataFrame)
        if self.empty(orders): return
        for settlement, (primary, secondary) in self.partition(orders, exposures, by=Querys.Settlement):
            primary = ODict(list(self.orders(primary, *args, **kwargs)))
            stabilities = self.calculate(primary, secondary, *args, **kwargs)
            size = self.size(stabilities)
            self.console(f"{str(settlement)}[{int(size):.0f}]")
            if self.empty(orders): continue
            yield orders

    def partition(self, orders, exposures, *args, **kwargs):
        assert isinstance(orders, pd.DataFrame) and isinstance(exposures, pd.DataFrame)
        for partition, primary in super().partition(orders, *args, **kwargs):
            mask = [exposures[key] == value for key, value in iter(partition)]
            mask = reduce(lambda lead, lag: lead & lag, list(mask))
            secondary = exposures.where(mask)
            yield partition, (primary, secondary)

    def calculate(self, orders, exposures, *args, **kwargs):
        assert isinstance(orders, pd.DataFrame) and isinstance(exposures, pd.DataFrame)
        exposures = self.exposures(exposures, *args, **kwargs)
        portfolios = list(self.portfolios(orders, exposures, *args, **kwargs))
        portfolios = xr.concat(portfolios, join="outer", fill_value=0, dim="order")
        portfolios = portfolios.stack(holdings=list(Variables.Securities.Security) + ["strike"]).to_dataset()
        portfolios = self.underlying(portfolios, *args, **kwargs)
        stabilities = self.calculation(portfolios, *args, **kwargs)
        stabilities = stabilities.to_dataframe()
        stabilities["stable"] = (stabilities["bear"] == 0) & (stabilities["bull"] == 0)
        stabilities = stabilities.reset_index(drop=False, inplace=False)
        stabilities.columns = pd.MultiIndex.from_product([stabilities.columns, [""]])
        return stabilities

    @staticmethod
    def orders(orders, *args, **kwargs):
        assert isinstance(orders, pd.DataFrame)
        for order, dataframe in orders.groupby("order"):
            dataframe = dataframe.drop(columns="order", inplace=False)
            series = dataframe.set_index(list(Querys.Settlement) + ["strike"] + list(Variables.Securities.Security), drop=True, inplace=False).squeeze()
            dataarray = xr.DataArray.from_series(series).fillna(0)
            function = lambda content, axis: content.squeeze(axis)
            dataarray = reduce(function, list(Querys.Settlement), dataarray)
            yield order, dataarray

    @staticmethod
    def exposures(exposures, *args, **kwargs):
        assert isinstance(exposures, pd.DataFrame)
        exposures = exposures.set_index(list(Querys.Settlement) + ["strike"] + list(Variables.Securities.Security), drop=True, inplace=False).squeeze()
        exposures = xr.DataArray.from_series(exposures).fillna(0)
        function = lambda content, axis: content.squeeze(axis)
        exposures = reduce(function, list(Querys.Settlement), exposures)
        return exposures

    @staticmethod
    def portfolios(orders, exposures, *args, **kwargs):
        assert isinstance(orders, dict) and isinstance(exposures, xr.DataArray)
        assert all([isinstance(dataarray, xr.DataArray) for dataarray in orders.values()])
        for order, dataarray in orders.items():
            dataarrays = xr.align(dataarray, exposures, fill_value=0, join="outer")
            dataarrays = sum(dataarrays).assign_coords(order=order).expand_dims("order")
            yield dataarrays

    @staticmethod
    def underlying(portfolios, *args, **kwargs):
        assert isinstance(portfolios, xr.Dataset)
        underlying = np.unique(portfolios["strike"].values)
        underlying = np.sort(np.concatenate([underlying - 0.001, underlying + 0.001]))
        portfolios["underlying"] = underlying
        return portfolios

    @property
    def calculation(self): return self.__calculation


class StabilityFilter(Sizing, Emptying, Partition, Logging, title="Filtered"):
    def execute(self, prospects, stabilities, *args, **kwargs):
        assert isinstance(prospects, pd.DataFrame) and isinstance(stabilities, pd.DataFrame)
        if self.empty(prospects): return
        header = list(prospects.columns)
        for settlement, dataframe in self.partition(prospects, by=Querys.Settlement):
            dataframe = dataframe.merge(stabilities, on="order", how="inner")
            dataframe = dataframe.where(dataframe["stable"])
            dataframe = dataframe[header].reset_index(drop=True, inplace=False)
            size = self.size(dataframe)
            string = f"{str(settlement)}[{int(size):.0f}]"
            self.console(string)
            if self.empty(dataframe): continue
            yield dataframe





