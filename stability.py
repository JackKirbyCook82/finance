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

from finance.variables import Variables, Contract
from support.calculations import Variable, Equation, Calculation
from support.processes import Calculator

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StabilityCalculator"]
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


class StabilityVariables(object):
    axes = {Variables.Querys.CONTRACT: ["ticker", "expire"], Variables.Datasets.SECURITY: ["instrument", "option", "position"]}

    def __init__(self, *args, **kwargs):
        self.security = self.axes[Variables.Datasets.SECURITY]
        self.contract = self.axes[Variables.Querys.CONTRACT]


class StabilityCalculator(Calculator, calculation=StabilityCalculation, variables=StabilityVariables):
    def execute(self, contract, valuations, exposures, allocations, *args, **kwargs):
        assert isinstance(contract, Contract) and all([isinstance(dataframe, pd.DataFrame) for dataframe in (valuations, exposures, allocations)])
        exposures = self.exposures(exposures, *args, **kwargs)
        allocations = ODict(list(self.allocations(allocations, *args, **kwargs)))
        portfolios = list(self.portfolios(exposures, allocations, *args, **kwargs))
        portfolios = xr.concat(portfolios, join="outer", fill_value=0, dim="portfolio")
        portfolios = portfolios.stack({"holdings": ["strike"] + self.variables.security}).to_dataset()
        portfolios = self.underlying(portfolios, *args, **kwargs)
        stability = ODict(list(self.stability(portfolios, *args, **kwargs)))
        valuations = self.valuations(valuations, stability, *args, **kwargs)
        size = self.size(valuations)
        string = f"{str(self.title)}: {repr(self)}|{str(contract)}[{size:.0f}]"
        self.logger.info(string)
        return valuations

    def exposures(self, exposures, *args, **kwargs):
        index = [column for column in exposures.columns if column != "quantity"]
        exposures = exposures.set_index(index, drop=True, inplace=False).squeeze()
        exposures = xr.DataArray.from_series(exposures).fillna(0)
        function = lambda content, axis: content.squeeze(axis)
        exposures = reduce(function, self.variables.contract, exposures)
        return exposures

    def allocations(self, allocations, *args, **kwargs):
        for portfolio, allocation in allocations.groupby("portfolio"):
            allocation = allocation.drop(columns="portfolio", inplace=False)
            index = [column for column in allocation.columns if column != "quantity"]
            allocation = allocation.set_index(index, drop=True, inplace=False).squeeze()
            allocation = xr.DataArray.from_series(allocation).fillna(0)
            function = lambda content, axis: content.squeeze(axis)
            allocation = reduce(function, self.variables.contract, allocation)
            yield portfolio, allocation

    def stability(self, portfolios, *args, **kwargs):
        stability = self.calculation(portfolios, *args, **kwargs)
        assert isinstance(stability, xr.Dataset)
        stability = stability.sum(dim="holdings")
        yield "maximum", stability["value"].max(dim="underlying")
        yield "minimum", stability["value"].min(dim="underlying")
        yield "bull", stability["trend"].isel(underlying=0).drop_vars(["underlying"])
        yield "bear", stability["trend"].isel(underlying=-1).drop_vars(["underlying"])

    @staticmethod
    def portfolios(exposures, allocations, *args, **kwargs):
        yield exposures.assign_coords(portfolio=0).expand_dims("portfolio")
        for portfolio, allocation in allocations.items():
            allocation = xr.align(allocation, exposures, fill_value=0, join="outer")[0]
            allocation = allocation.assign_coords(portfolio=portfolio).expand_dims("portfolio")
            yield allocation

    @staticmethod
    def underlying(portfolios, *args, **kwargs):
        underlying = np.unique(portfolios["strike"].values)
        underlying = np.sort(np.concatenate([underlying - 0.001, underlying + 0.001]))
        portfolios["underlying"] = underlying
        return portfolios

    @staticmethod
    def valuations(valuations, stability, *args, **kwargs):
        stability = xr.Dataset(stability).to_dataframe()
        stable = (stability["bear"] == 0) & (stability["bull"] == 0)
        valuations = valuations.where(stable).dropna(how="all", inplace=False)
        valuations = valuations.drop(columns="portfolio", inplace=False)
        return valuations




