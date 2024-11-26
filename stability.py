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
from support.mixins import Emptying, Sizing, Logging

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
    Θ = Variable("Θ", "theta", np.int32, xr.DataArray, vectorize=True, function=lambda i: + int(Variables.Theta(str(i))))
    Φ = Variable("Φ", "phi", np.int32, xr.DataArray, vectorize=True, function=lambda j: + int(Variables.Phi(str(j))))

    Σy = Variable("Σy", "value", np.float32, xr.DataArray, vectorize=False, function=lambda y: y.sum("holdings").drop(list(Querys.Contract)))
    Σm = Variable("Σm", "value", np.float32, xr.DataArray, vectorize=False, function=lambda m: m.sum("holdings").drop(list(Querys.Contract)))

    Σyh = Variable("Σyh", "maximum", np.float32, xr.DataArray, vectorize=False, function=lambda Σy: Σy.max(dim="underlying"))
    Σyl = Variable("Σyl", "minimum", np.float32, xr.DataArray, vectorize=False, function=lambda Σy: Σy.min(dim="underlying"))
    Σmh = Variable("Σmh", "bull", np.float32, xr.DataArray, vectorize=False, function=lambda Σm: Σm.isel(underlying=0).drop_vars(["underlying"]))
    Σml = Variable("Σml", "bear", np.float32, xr.DataArray, vectorize=False, function=lambda Σm: Σm.isel(underlying=-1).drop_vars(["underlying"]))

    x = Variable("x", "underlying", np.float32, xr.DataArray, locator="underlying")
    q = Variable("q", "quantity", np.int32, xr.DataArray, locator="quantity")
    i = Variable("i", "option", Variables.Options, xr.DataArray, locator="option")
    j = Variable("j", "position", Variables.Positions, xr.DataArray, locator="position")
    k = Variable("k", "strike", np.float32, xr.DataArray, locator="strike")


class StabilityCalculation(Calculation, equation=StabilityEquation):
    def execute(self, portfolios, *args, **kwargs):
        with self.equation(portfolios) as equation:
            equation.y(portfolios)
            equation.m(portfolios)
            yield equation.Σyh(portfolios)
            yield equation.Σyl(portfolios)
            yield equation.Σmh(portfolios)
            yield equation.Σml(portfolios)


class StabilityCalculator(Logging, Sizing, Emptying):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calculation = StabilityCalculation(*args, **kwargs)

    def execute(self, orders, exposures, *args, **kwargs):
        if self.empty(orders): return
        assert isinstance(orders, pd.DataFrame) and isinstance(exposures, pd.DataFrame)
        for contract, dataframes in self.source(orders, exposures, *args, **kwargs):
            stabilities = self.calculate(*dataframes, *args, **kwargs)
            size = self.size(stabilities)
            string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
            self.logger.info(string)
            if self.empty(stabilities): continue
            return stabilities

    def calculate(self, orders, exposures, *args, **kwargs):
        assert isinstance(orders, pd.DataFrame) and isinstance(exposures, pd.DataFrame)
        orders = ODict(list(self.orders(orders, *args, **kwargs)))
        exposures = self.exposures(exposures, *args, **kwargs)
        portfolios = list(self.portfolios(orders, exposures, *args, **kwargs))
        portfolios = xr.concat(portfolios, join="outer", fill_value=0, dim="order")
        holdings = list(Variables.Security) + ["strike"]
        portfolios = portfolios.stack(holdings=holdings).to_dataset()
        portfolios = self.underlying(portfolios, *args, **kwargs)
        stabilities = self.calculation(portfolios, *args, **kwargs)
        stabilities = stabilities.to_dataframe()
        stabilities["stable"] = (stabilities["bear"] == 0) & (stabilities["bull"] == 0)
        stabilities = stabilities.reset_index(drop=False, inplace=False)
        stabilities.columns = pd.MultiIndex.from_product([stabilities.columns, [""]])
        return stabilities

    @staticmethod
    def source(orders, exposures, *args, **kwargs):
        assert isinstance(orders, pd.DataFrame) and isinstance(exposures, pd.DataFrame)
        for (ticker, expire), primary in orders.groupby(["ticker", "expire"]):
            contract = Querys.Contract(ticker, expire)
            mask = exposures["ticker"] == ticker & exposures["expire"] == expire
            secondary = exposures.where(mask)
            yield contract, (primary, secondary)

    @staticmethod
    def orders(orders, *args, **kwargs):
        assert isinstance(orders, pd.DataFrame)
        for order, dataframe in orders.groupby("order"):
            dataframe = dataframe.drop(columns="order", inplace=False)
            index = list(Querys.Product) + list(Variables.Security)
            series = dataframe.set_index(index, drop=True, inplace=False).squeeze()
            dataarray = xr.DataArray.from_series(series).fillna(0)
            function = lambda content, axis: content.squeeze(axis)
            dataarray = reduce(function, list(Querys.Contract), dataarray)
            yield order, dataarray

    @staticmethod
    def exposures(exposures, *args, **kwargs):
        assert isinstance(exposures, pd.DataFrame)
        index = list(Querys.Product) + list(Variables.Security)
        exposures = exposures.set_index(index, drop=True, inplace=False).squeeze()
        exposures = xr.DataArray.from_series(exposures).fillna(0)
        function = lambda content, axis: content.squeeze(axis)
        exposures = reduce(function, list(Querys.Contract), exposures)
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


class StabilityFilter(Logging, Sizing, Emptying):
    def execute(self, prospects, stabilities, *args, **kwargs):
        if self.empty(prospects): return
        assert isinstance(prospects, pd.DataFrame) and isinstance(stabilities, pd.DataFrame)
        for contract, dataframes in self.source(prospects, stabilities, *args, **kwargs):
            filtered = self.calculate(*dataframes, *args, **kwargs)
            size = self.size(filtered)
            string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
            self.logger.info(string)
            if self.empty(filtered): continue
            return filtered

    @staticmethod
    def source(prospects, stabilities, *args, **kwargs):
        assert isinstance(prospects, pd.DataFrame) and isinstance(stabilities, pd.DataFrame)
        for (ticker, expire), primary in prospects.groupby(["ticker", "expire"]):
            contract = Querys.Contract(ticker, expire)
            mask = stabilities["ticker"] == ticker & stabilities["expire"] == expire
            secondary = stabilities.where(mask)
            yield contract, (primary, secondary)

    @staticmethod
    def calculate(prospects, stabilities, *args, **kwargs):
        assert isinstance(prospects, pd.DataFrame) and isinstance(stabilities, pd.DataFrame)
        prospects = prospects.merge(stabilities, on="order", how="inner")
        prospects = prospects.where(prospects["stable"])
        prospects = prospects.reset_index(drop=True, inplace=False)
        return prospects



