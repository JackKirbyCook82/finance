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
from support.mixins import Function, Emptying, Sizing, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StabilityCalculator", "StabilityFilter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class StabilityFilter(Function, Logging, Sizing, Emptying):
    def __init__(self, *args, **kwargs):
        Function.__init__(self, *args, **kwargs)
        Logging.__init__(self, *args, **kwargs)

    def execute(self, source, *args, **kwargs):
        assert isinstance(source, tuple)
        contract, valuations, stabilities = source
        assert isinstance(contract, Querys.Contract) and isinstance(valuations, pd.DataFrame) and isinstance(stabilities, pd.DataFrame)
        if self.empty(valuations): return
        valuations = self.calculate(valuations, stabilities, *args, **kwargs)
        size = self.size(valuations)
        string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
        self.logger.info(string)
        if self.empty(valuations): return
        return valuations

    def calculate(self, valuations, stabilities, *args, **kwargs):
        pass


class StabilityCalculator(Function, Logging, Sizing, Emptying):
    def __init__(self, *args, **kwargs):
        Function.__init__(self, *args, **kwargs)
        Logging.__init__(self, *args, **kwargs)

    def execute(self, source, *args, **kwargs):
        assert isinstance(source, tuple)
        contract, orders, exposures = source
        assert isinstance(contract, Querys.Contract) and isinstance(orders, pd.DataFrame) and isinstance(exposures, pd.DataFrame)
        if self.empty(orders): return
        stabilities = self.calculate(orders, exposures, *args, **kwargs)
        size = self.size(stabilities)
        string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
        self.logger.info(string)
        if self.empty(stabilities): return
        return stabilities

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
        assert isinstance(stabilities, pd.DataFrame)
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





