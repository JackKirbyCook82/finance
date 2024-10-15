# -*- coding: utf-8 -*-
"""
Created on Fri May 17 2024
@name:   Stability Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
import xarray as xr
from functools import reduce
from collections import OrderedDict as ODict

from finance.variables import Variables, Querys
from support.mixins import Emptying, Sizing, Logging, Pipelining, Sourcing
from support.calculations import Variable, Equation, Calculation

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StabilityCalculator", "StabilityFilter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class StabilityEquation(Equation):
    y = Variable("y", "value", np.float32, function=lambda q, Θ, Φ, Ω, Δ: q * Θ * Φ * Δ * (1 - Θ * Ω) / 2)
    m = Variable("m", "trend", np.float32, function=lambda q, Θ, Φ, Ω: q * Θ * Φ * Ω)
    Ω = Variable("Ω", "omega", np.int32, function=lambda x, k: np.sign(x / k - 1))
    Δ = Variable("Δ", "delta", np.int32, function=lambda x, k: np.subtract(x, k))
    Θ = Variable("Θ", "theta", np.int32, function=lambda i: + int(Variables.Theta(str(i))))
    Φ = Variable("Φ", "phi", np.int32, function=lambda j: + int(Variables.Phi(str(j))))

    x = Variable("x", "underlying", np.float32, position=0, locator="underlying")
    q = Variable("q", "quantity", np.int32, position=0, locator="quantity")
    i = Variable("i", "option", Variables.Options, position=0, locator="option")
    j = Variable("j", "position", Variables.Positions, position=0, locator="position")
    k = Variable("k", "strike", np.float32, position=0, locator="strike")


class StabilityCalculation(Calculation, equation=StabilityEquation):
    def execute(self, portfolios, *args, **kwargs):
        equation = self.equation(*args, **kwargs)
        yield equation.y(portfolios)
        yield equation.m(portfolios)


class StabilityFilter(Pipelining, Sourcing, Logging, Sizing, Emptying):
    def __init__(self, *args, **kwargs):
        Pipelining.__init__(self, *args, **kwargs)
        Logging.__init__(self, *args, **kwargs)

    def execute(self, valuations, stabilities, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame) and isinstance(stabilities, pd.DataFrame)
        if self.empty(valuations): return
        for contract, primary in self.source(valuations, keys=list(Querys.Contract)):
            contract = Querys.Contract(contract)
            if self.empty(primary): continue
            parameters = dict(keys=list(contract.keys()), values=list(contract.values()))
            secondary = self.align(stabilities, **parameters)
            dataframe = self.calculate(primary, secondary, *args, **kwargs)
            size = self.size(dataframe)
            string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
            self.logger.info(string)
            if self.empty(dataframe): continue
            yield dataframe

    def calculate(self, valuations, stabilities, *args, **kwargs):
        pass


class StabilityCalculator(Pipelining, Sourcing, Logging, Sizing, Emptying):
    def __init__(self, *args, **kwargs):
        Pipelining.__init__(self, *args, **kwargs)
        Logging.__init__(self, *args, **kwargs)
        self.__calculation = StabilityCalculation(*args, **kwargs)

    def execute(self, orders, exposures, *args, **kwargs):
        assert isinstance(orders, pd.DataFrame) and isinstance(exposures, pd.DataFrame)
        if self.empty(orders): return
        for contract, primary in self.source(orders, keys=list(Querys.Contract)):
            contract = Querys.Contract(contract)
            if self.empty(primary): continue
            parameters = dict(keys=list(contract.keys()), values=list(contract.values()))
            secondary = self.align(exposures, **parameters)
            stabilities = self.calculate(primary, secondary, *args, **kwargs)
            size = self.size(stabilities)
            string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
            self.logger.info(string)
            if self.empty(stabilities): continue
            yield stabilities

    def calculate(self, orders, exposures, *args, **kwargs):
        assert isinstance(orders, pd.DataFrame) and isinstance(exposures, pd.DataFrame)
        orders = ODict(list(self.orders(orders, *args, **kwargs)))
        exposures = self.exposures(exposures, *args, **kwargs)
        portfolios = list(self.portfolios(orders, exposures, *args, **kwargs))
        portfolios = xr.concat(portfolios, join="outer", fill_value=0, dim="portfolio")
        holdings = list(Variables.Security) + ["strike"]
        portfolios = portfolios.stack(holdings=holdings).to_dataset()
        stabilities = self.stabilities(portfolios, *args, **kwargs)
        return stabilities

    def stabilities(self, portfolios, *args, **kwargs):
        assert isinstance(portfolios, xr.Dataset)
        portfolios = self.underlying(portfolios, *args, **kwargs)
        stabilities = self.calculation(portfolios, *args, **kwargs)
        assert isinstance(stabilities, xr.Dataset)
        stabilities = stabilities.sum(dim="holdings")
        stabilities["maximum"] = stabilities["value"].max(dim="underlying")
        stabilities["minimum"] = stabilities["value"].min(dim="underlying")
        stabilities["bull"] = stabilities["trend"].isel(underlying=0).drop_vars(["underlying"])
        stabilities["bear"] = stabilities["trend"].isel(underlying=-1).drop_vars(["underlying"])
        stabilities = xr.Dataset(stabilities).to_dataframe()
        stabilities["stable"] = (stabilities["bear"] == 0) & (stabilities["bull"] == 0)
        return stabilities

    @staticmethod
    def orders(orders, *args, **kwargs):
        assert isinstance(orders, pd.DataFrame)
        for portfolio, dataframe in orders.groupby("portfolio"):
            dataframe = dataframe.drop(columns="portfolio", inplace=False)
            index = list(Variables.Contract) + list(Variables.Product)
            dataframe = dataframe.set_index(index, drop=True, inplace=False).squeeze()
            dataarray = xr.DataArray.from_series(dataframe).fillna(0)
            function = lambda content, axis: content.squeeze(axis)
            dataarray = reduce(function, list(Variables.Contract), dataarray)
            yield portfolio, dataarray

    @staticmethod
    def exposures(exposures, *args, **kwargs):
        assert isinstance(exposures, pd.DataFrame)
        index = list(Variables.Contract) + list(Variables.Product)
        exposures = exposures.set_index(index, drop=True, inplace=False).squeeze()
        exposures = xr.DataArray.from_series(exposures).fillna(0)
        function = lambda content, axis: content.squeeze(axis)
        exposures = reduce(function, list(Variables.Contract), exposures)
        return exposures

    @staticmethod
    def portfolios(orders, exposures, *args, **kwargs):
        assert isinstance(orders, dict) and isinstance(exposures, xr.DataArray)
        assert all([isinstance(dataframe, xr.DataArray) for dataframe in orders.values()])
        for portfolio, dataframe in orders.items():
            dataarray = xr.align(dataframe, exposures, fill_value=0, join="outer")[0]
            dataarray = dataarray.assign_coords(portfolio=portfolio).expand_dims("portfolio")
            yield dataarray

    @staticmethod
    def underlying(portfolios, *args, **kwargs):
        assert isinstance(portfolios, xr.Dataset)
        underlying = np.unique(portfolios["strike"].values)
        underlying = np.sort(np.concatenate([underlying - 0.001, underlying + 0.001]))
        portfolios["underlying"] = underlying
        return portfolios

    @property
    def calculation(self): return self.__calculations



