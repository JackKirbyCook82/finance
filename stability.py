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
from collections import OrderedDict as ODict

from finance.variables import Variables
from finance.operations import Operations
from support.calculations import Variable, Equation, Calculation

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


class StabilityCalculator(Operations.Processor):
    def __init__(self, *args, valuation, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__calculation = StabilityCalculation(*args, **kwargs)
        self.__valuation = valuation

    def processor(self, contents, *args, **kwargs):
        valuations, exposures, allocations = contents[self.valuation], contents[Variables.Datasets.EXPOSURE], contents[Variables.Datasets.ALLOCATION]
        assert all([isinstance(dataframe, pd.DataFrame) for dataframe in (valuations, exposures, allocations)])
        exposures = self.exposures(exposures, *args, **kwargs)
        allocations = ODict(list(self.allocations(allocations, *args, **kwargs)))
        portfolios = list(self.portfolios(exposures, allocations, *args, **kwargs))
        portfolios = xr.concat(portfolios, join="outer", fill_value=0, dim="portfolio")
        portfolios = portfolios.stack({"holdings": ["strike", "option", "position"]}).to_dataset()
        portfolios = self.underlying(portfolios, *args, **kwargs)
        stability = ODict(list(self.calculate(portfolios, *args, **kwargs)))
        stability = xr.Dataset(stability).to_dataframe()
        stability["stable"] = (stability["bear"] == 0) & (stability["bull"] == 0)
        valuations = self.valuations(valuations, stability, *args, **kwargs)
        valuations = {self.valuation: valuations}
        yield contents | dict(valuations)

    def calculate(self, portfolios, *args, **kwargs):
        stability = self.calculation(portfolios, *args, **kwargs)
        stability = stability.sum(dim="holdings")
        yield "maximum", stability["value"].max(dim="underlying").drop_vars("instrument")
        yield "minimum", stability["value"].min(dim="underlying").drop_vars("instrument")
        yield "bull", stability["trend"].isel(underlying=0).drop_vars(["underlying", "instrument"])
        yield "bear", stability["trend"].isel(underlying=-1).drop_vars(["underlying", "instrument"])

    @staticmethod
    def exposures(exposures, *args, **kwargs):
        index = [column for column in exposures.columns if column != "quantity"]
        exposures = exposures.set_index(index, drop=True, inplace=False).squeeze()
        exposures = xr.DataArray.from_series(exposures).fillna(0)
        exposures = exposures.squeeze("ticker").squeeze("expire").squeeze("instrument")
        return exposures

    @staticmethod
    def allocations(allocations, *args, **kwargs):
        for portfolio, dataframe in allocations.groupby("portfolio"):
            dataframe = dataframe.drop(columns="portfolio", inplace=False)
            index = [column for column in dataframe.columns if column != "quantity"]
            dataframe = dataframe.set_index(index, drop=True, inplace=False).squeeze()
            dataframe = xr.DataArray.from_series(dataframe).fillna(0)
            dataframe = dataframe.squeeze("ticker").squeeze("expire").squeeze("instrument")
            yield portfolio, dataframe

    @staticmethod
    def portfolios(exposures, allocations, *args, **kwargs):
        function = lambda dataarray: xr.align(dataarray, exposures, fill_value=0, join="outer")[0]
        yield exposures.assign_coords(portfolio=0)
        for portfolio, allocation in allocations.items():
            allocation = function(allocation)
            allocation = allocation.assign_coords(portfolio=portfolio)
            yield allocation

    @staticmethod
    def underlying(portfolios, *args, **kwargs):
        underlying = np.unique(portfolios["strike"].values)
        underlying = np.sort(np.concatenate([underlying - 0.001, underlying + 0.001]))
        portfolios["underlying"] = underlying
        return portfolios

    @staticmethod
    def valuations(valuations, stability, *args, **kwargs):


        print(valuations)
        print(stability)
        raise Exception()

    @property
    def calculation(self): return self.__calculation
    @property
    def valuation(self): return self.__valuation




