# -*- coding: utf-8 -*-
"""
Created on Fri May 17 2024
@name:   Exposures Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
import xarray as xr

from finance.variables import Variables
from support.calculations import Variable, Equation, Calculation
from support.pipelines import Processor

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ExposureCalculator", "StabilityCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class StabilityEquation(Equation):
    y = Variable("y", "value", np.float32, function=lambda q, Θ, Φ, Ω, Δ: q * Θ * Φ * Δ * (1 - Θ * Ω) / 2)
    m = Variable("m", "trend", np.int32, function=lambda q, Θ, Φ, Ω: q * Θ * Φ * Ω)
    Ω = Variable("Ω", "omega", np.int32, function=lambda x, k: np.sign(x / k - 1))
    Δ = Variable("Δ", "delta", np.int32, function=lambda x, k: np.subtract(x, k))
    Θ = Variable("Θ", "theta", np.int32, function=lambda i: int(Variables.Theta(str(i))))
    Φ = Variable("Φ", "phi", np.int32, function=lambda j: int(Variables.Phi(str(j))))

    x = Variable("x", "underlying", np.float32, position=0, locator="underlying")
    q = Variable("q", "quantity", np.int32, position=0, locator="quantity")
    i = Variable("i", "option", Variables.Options, position=0, locator="option")
    j = Variable("j", "position", Variables.Positions, position=0, locator="position")
    k = Variable("k", "strike", np.float32, position=0, locator="strike")


class StabilityCalculation(Calculation, equation=StabilityEquation):
    def execute(self, exposures, *args, **kwargs):
        equation = self.equation(*args, **kwargs)
        yield equation.y(exposures)
        yield equation.m(exposures)


class ExposureCalculator(Processor):
    def execute(self, contents, *args, **kwargs):
        holdings = contents[Variables.Datasets.HOLDINGS]
        stocks = self.stocks(holdings, *args, **kwargs)
        options = self.options(holdings, *args, **kwargs)
        virtuals = self.virtuals(stocks, *args, **kwargs)
        securities = pd.concat([options, virtuals], axis=0)
        securities = securities.reset_index(drop=True, inplace=False)
        exposures = self.holdings(securities, *args, *kwargs)
        exposures = exposures.reset_index(drop=True, inplace=False)
        exposures = {Variables.Datasets.EXPOSURE: exposures}
        yield contents | exposures

    @staticmethod
    def stocks(holdings, *args, **kwargs):
        stocks = holdings["instrument"] == Variables.Instruments.STOCK
        dataframe = holdings.where(stocks).dropna(how="all", inplace=False)
        return dataframe

    @staticmethod
    def options(holdings, *args, **kwargs):
        options = holdings["instrument"] == Variables.Instruments.OPTION
        dataframe = holdings.where(options).dropna(how="all", inplace=False)
        puts = dataframe["option"] == Variables.Options.PUT
        calls = dataframe["option"] == Variables.Options.CALL
        dataframe = dataframe.where(puts | calls).dropna(how="all", inplace=False)
        return dataframe

    @staticmethod
    def virtuals(stocks, *args, **kwargs):
        security = lambda instrument, option, position: dict(instrument=instrument, option=option, position=position)
        function = lambda records, instrument, option, position: pd.DataFrame.from_records([record | security(instrument, option, position) for record in records])
        if bool(stocks.empty):
            return pd.DataFrame()
        stocklong = stocks["position"] == Variables.Positions.LONG
        stocklong = stocks.where(stocklong).dropna(how="all", inplace=False)
        stockshort = stocks["position"] == Variables.Positions.SHORT
        stockshort = stocks.where(stockshort).dropna(how="all", inplace=False)
        putlong = function(stockshort.to_dict("records"), Variables.Instruments.OPTION, Variables.Options.PUT, Variables.Positions.LONG)
        putshort = function(stocklong.to_dict("records"), Variables.Instruments.OPTION, Variables.Options.PUT, Variables.Positions.SHORT)
        calllong = function(stocklong.to_dict("records"), Variables.Instruments.OPTION, Variables.Options.CALL, Variables.Positions.LONG)
        callshort = function(stockshort.to_dict("records"), Variables.Instruments.OPTION, Variables.Options.CALL, Variables.Positions.SHORT)
        virtuals = pd.concat([putlong, putshort, calllong, callshort], axis=0)
        virtuals["strike"] = virtuals["strike"].apply(lambda strike: np.round(strike, decimals=2))
        return virtuals

    @staticmethod
    def holdings(securities, *args, **kwargs):
        index = [value for value in securities.columns if value not in ("position", "quantity")]
        numerical = lambda position: 2 * int(bool(position is Variables.Positions.LONG)) - 1
        enumerical = lambda value: Variables.Positions.LONG if value > 0 else Variables.Positions.SHORT
        holdings = lambda cols: cols["quantity"] * numerical(cols["position"])
        securities["quantity"] = securities.apply(holdings, axis=1)
        dataframe = securities.groupby(index, as_index=False, sort=False).agg({"quantity": np.sum})
        dataframe = dataframe.where(dataframe["quantity"] != 0).dropna(how="all", inplace=False)
        dataframe["position"] = dataframe["quantity"].apply(enumerical)
        dataframe["quantity"] = dataframe["quantity"].apply(np.abs)
        return dataframe


class StabilityCalculator(Processor):
    def __init__(self, *args, valuation, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__calculation = StabilityCalculation(*args, **kwargs)
        self.__valuation = valuation

    def execute(self, contents, *args, **kwargs):
        valuations, exposures = contents[self.valuation], contents[Variables.Datasets.EXPOSURE]
        assert isinstance(valuations, pd.DataFrame) and isinstance(exposures, pd.DataFrame)

        pd.set_option("display.max_columns", 100)
        pd.set_option("display.width", 250)
        xr.set_options(display_width=250)
        np.set_printoptions(linewidth=250)

        series = exposures.set_index(["ticker", "expire", "strike", "instrument", "option", "position"], drop=True, inplace=False).squeeze()
        exposures = xr.DataArray.from_series(series).fillna(0)
        exposures = exposures.squeeze("ticker").squeeze("expire").squeeze("instrument")
        exposures = exposures.stack({"holdings": ["strike", "option", "position"]})
        print(exposures, "\n")

        index = ["ticker", "expire", "strategy", "valuation"] + list(map(str, Variables.Securities.Options))
        valuations = valuations.pivot(index=index, columns="scenario").reset_index(drop=False, inplace=False)
        print(valuations, "\n")

        records = list(valuations.droplevel(level="scenario", axis=1)[index].to_dict("records"))
        function = lambda key, value: key in list(map(str, Variables.Securities.Options)) and not np.isnan(value)
        records = [{key: value for key, value in record.items() if function(key, value)} for record in records]
        options = [{Variables.Securities(option): strike for option, strike in record.items()} for record in records]
        for index, option in enumerate(options):
            for key, value in option.items():
                print(str(index), str(key), str(value))
        raise Exception()

    @property
    def calculation(self): return self.__calculation
    @property
    def valuation(self): return self.__valuation


#    def calculate(self, valuations, exposures, *args, **kwargs):
#        adjustment = xr.DataArray(data=, coords=exposures.coords)
#        columns = list(map(str, Variables.Securities.Options))
#        options = valuations[columns].droplevel(level="scenario", axis=1)
#        valuations["stable"] = options.apply(self.stability, axis=1, exposures=exposures)
#        valuations = valuations.where(valuations["stable"])
#        valuations = valuations.dropna(how="all", inplace=False)
#        valuations = valuations.drop("stable", inplace=False)
#        return valuations

#    def stability(self, options, *args, exposures, **kwargs):
#        options = options.dropna(how="all", inplace=False).to_dict()
#        options = {Variables.Securities(key): value for key, value in options.items()}
#        adjustment = xr.zeros_like(exposures)
#        dataset = self.calculation(exposures, *args, **kwargs)
#        dataset = dataset.reduce(np.sum, dim="holdings", keepdims=False)

#        underlying = np.unique(dataset["strike"].values)
#        underlying = np.sort(np.concatenate([underlying - 0.001, underlying + 0.001]))
#        dataset["underlying"] = underlying




