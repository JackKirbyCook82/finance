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
from functools import reduce
from collections import OrderedDict as ODict

from finance.variables import Variables
from finance.operations import Operations
from support.calculations import Variable, Equation, Calculation

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["FeasibilityCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


feasibility_index = ["ticker", "expire", "strike", "instrument", "option", "position"]
feasibility_columns = ["current", "apy", "npv", "cost", "size", "underlying"]
feasibility_stacking = {Variables.Valuations.ARBITRAGE: {"apy", "npv", "cost"}}
feasibility_options = list(map(str, Variables.Securities.Options))
feasibility_stocks = list(map(str, Variables.Securities.Stocks))
feasibility_contract = ["ticker", "expire"]


class FeasibilityEquation(Equation):
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


class FeasibilityCalculation(Calculation, equation=FeasibilityEquation):
    def execute(self, portfolios, *args, **kwargs):
        equation = self.equation(*args, **kwargs)
        yield equation.y(portfolios)
        yield equation.m(portfolios)


class FeasibilityCalculator(Operations.Processor):
    def __init__(self, *args, valuation, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__calculation = FeasibilityCalculation(*args, **kwargs)
        self.__valuation = valuation

    def processor(self, contents, *args, **kwargs):
        valuations, exposures = contents[self.valuation], contents[Variables.Datasets.EXPOSURE]
        assert isinstance(valuations, pd.DataFrame) and isinstance(exposures, pd.DataFrame)
        valuations = self.valuations(valuations, *args, **kwargs)
        securities = self.securities(valuations, *args, **kwargs)
        exposures = self.exposures(exposures, *args, **kwargs)
        allocations = ODict(list(self.allocate(securities, *args, **kwargs)))
        portfolios = self.portfolios(exposures, allocations, *args, **kwargs)
        portfolios = self.underlying(portfolios, *args, **kwargs)
        payoffs = self.calculation(portfolios, *args, **kwargs)
        stable = self.stability(payoffs, *args, **kwargs)
        valuations = self.feasibility(valuations, stable, *args, **kwargs)
        if bool(valuations.empty):
            return
        valuations = {self.valuation: valuations}
        yield contents | dict(valuations)

    def valuations(self, valuations, *args, **kwargs):
        columns = {column: np.NaN for column in feasibility_options if column not in valuations.columns}
        for column, value in columns.items():
            valuations[column] = value
        index = set(valuations.columns) - ({"scenario"} | feasibility_stacking[self.valuation])
        valuations = valuations.pivot(index=list(index), columns="scenario")
        valuations = valuations.reset_index(drop=False, inplace=False)
        valuations.index += 1
        return valuations

    def allocate(self, securities, *args, **kwargs):
        for identity, securities in securities.iterrows():
            stocks = self.separate(securities, *args, columns=feasibility_stocks, **kwargs)
            options = self.separate(securities, *args, columns=feasibility_options, **kwargs)
            virtuals = self.virtuals(stocks, *args, **kwargs)
            portfolio = pd.concat([options, virtuals], axis=0).dropna(how="any", inplace=False)
            portfolio = portfolio.reset_index(drop=True, inplace=False)
            index = [column for column in portfolio.columns if column != "quantity"]
            portfolio = portfolio.set_index(index, drop=True, inplace=False).squeeze()
            portfolio = xr.DataArray.from_series(portfolio).fillna(0)
            portfolio = portfolio.squeeze("ticker").squeeze("expire").squeeze("instrument")
            portfolio = portfolio.stack({"holdings": ["strike", "option", "position"]})
            yield identity, portfolio

    @staticmethod
    def securities(valuations, *args, **kwargs):
        stocks = list(map(str, Variables.Securities.Stocks))
        strategy = lambda cols: list(map(str, cols["strategy"].stocks))
        function = lambda cols: {stock: cols["underlying"] if stock in strategy(cols) else np.NaN for stock in stocks}
        options = valuations[feasibility_contract + feasibility_options + ["valuation", "strategy", "underlying"]]
        options = options.droplevel("scenario", axis=1)
        stocks = options.apply(function, axis=1, result_type="expand")
        securities = pd.concat([options, stocks], axis=1)
        securities = securities[feasibility_contract + feasibility_options + feasibility_stocks]
        return securities

    @staticmethod
    def separate(securities, *args, columns, **kwargs):
        security = lambda cols: list(Variables.Securities(cols["security"])) + [1]
        dataframe = securities[columns].to_frame("strike")
        dataframe = dataframe.reset_index(names="security", drop=False, inplace=False)
        dataframe[["instrument", "option", "position", "quantity"]] = dataframe.apply(security, axis=1, result_type="expand")
        dataframe = dataframe[[column for column in dataframe.columns if column != "security"]]
        for key, value in securities[feasibility_contract].to_dict().items():
            dataframe[key] = value
        return dataframe

    @staticmethod
    def virtuals(stocks, *args, **kwargs):
        security = lambda instrument, option, position: dict(instrument=instrument, option=option, position=position)
        function = lambda records, instrument, option, position: pd.DataFrame.from_records([record | security(instrument, option, position) for record in records])
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
    def exposures(exposures, *args, **kwargs):
        series = exposures.set_index(feasibility_index, drop=True, inplace=False).squeeze()
        exposures = xr.DataArray.from_series(series).fillna(0)
        exposures = exposures.squeeze("ticker").squeeze("expire").squeeze("instrument")
        exposures = exposures.stack({"holdings": ["strike", "option", "position"]})
        return exposures

    @staticmethod
    def portfolios(exposures, allocations, *args, **kwargs):
        combinations = {key: xr.align(value, exposures, join="outer") for key, value in allocations.items()}
        combinations = {key: [value.fillna(0) for value in values] for key, values in combinations.items()}
        portfolios = {key: reduce(lambda x, y: x + y, values) for key, values in combinations.items()}
        portfolios = [portfolio.assign_coords(portfolio=index) for index, portfolio in portfolios.items()]
        portfolios = xr.concat([exposures.assign_coords(portfolio=0)] + portfolios, dim="portfolio")
        portfolios = portfolios.to_dataset(name="quantity")
        return portfolios

    @staticmethod
    def underlying(portfolios, *args, **kwargs):
        underlying = np.unique(portfolios["strike"].values)
        underlying = np.sort(np.concatenate([underlying - 0.001, underlying + 0.001]))
        portfolios["underlying"] = underlying
        return portfolios

    @staticmethod
    def stability(payoffs, *args, **kwargs):
        payoffs = payoffs.sum(dim="holdings")
        payoffs["maximum"] = payoffs["value"].max(dim="underlying")
        payoffs["minimum"] = payoffs["value"].min(dim="underlying")
        payoffs["bull"] = payoffs["trend"].isel(underlying=0)
        payoffs["bear"] = payoffs["trend"].isel(underlying=-1)
        payoffs["stable"] = (payoffs["bear"] == 0) & (payoffs["bull"] == 0)
        stable = payoffs["stable"].to_series()
        return stable

    @staticmethod
    def feasibility(valuations, stable, *args, **kwargs):
        valuations = valuations.where(stable).dropna(how="all", inplace=False)
        index = [column for column in valuations.columns if len(list(filter(bool, column))) <= 1]
        valuations = valuations.set_index(index, drop=True, inplace=False)
        valuations = valuations.stack("scenario")
        valuations = valuations.reset_index(drop=False, inplace=False)
        valuations.columns = [column[0] if isinstance(column, tuple) else column for column in valuations.columns]
        return valuations

    @property
    def calculation(self): return self.__calculation
    @property
    def valuation(self): return self.__valuation

