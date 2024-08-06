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
from collections import OrderedDict as ODict

from finance.variables import Variables
from support.calculations import Variable, Equation, Calculation
from support.pipelines import Processor

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ExposureCalculator", "FeasibilityCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


feasibility_formatter = lambda self, *, contents, elapsed, **kw: f"{str(self.title)}: {repr(self)}|{str(contents[Variables.Querys.CONTRACT])}[{elapsed:.02f}s]"
feasibility_index = ["ticker", "expire", "strike", "instrument", "option", "position"]
feasibility_columns = ["current", "apy", "npv", "cost", "size", "underlying"]
feasibility_options = list(map(str, Variables.Securities.Options))
feasibility_contract = ["ticker", "expire"]


class FeasibilityEquation(Equation):
    y = Variable("y", "value", np.float32, function=lambda q, Θ, Φ, Ω, Δ: q * Θ * Φ * Δ * (1 - Θ * Ω) / 2)
    m = Variable("m", "trend", np.int32, function=lambda q, Θ, Φ, Ω: q * Θ * Φ * Ω)
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


class ExposureCalculator(Processor, formatter=feasibility_formatter):
    def processor(self, contents, *args, **kwargs):
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


class FeasibilityCalculator(Processor, formatter=feasibility_formatter):
    def __init__(self, *args, valuation, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__calculation = FeasibilityCalculation(*args, **kwargs)
        self.__valuation = valuation

    def processor(self, contents, *args, **kwargs):
        valuations, exposures = contents[self.valuation], contents[Variables.Datasets.EXPOSURE]
        assert isinstance(valuations, pd.DataFrame) and isinstance(exposures, pd.DataFrame)
        valuations = self.valuations(valuations, *args, **kwargs)
        exposures = self.exposures(exposures, *args, **kwargs)
        divestitures = ODict(list(self.divestitures(valuations, *args, **kwargs)))
        portfolios = self.portfolios(exposures, divestitures, *args, **kwargs)
        portfolios = self.underlying(portfolios, *args, **kwargs)
        payoffs = self.calculation(portfolios, *args, **kwargs)
        stable = self.stability(payoffs, *args, **kwargs)
        valuations = self.feasibility(valuations, stable, *args, **kwargs)
        if bool(valuations.empty):
            return
        valuations = {self.valuation: valuations}
        yield contents | valuations

    @staticmethod
    def valuations(valuations, *args, **kwargs):
        index = set(valuations.columns) - ({"scenario"} | set(feasibility_columns))
        valuations = valuations.pivot(index=list(index), columns="scenario")
        portfolios = pd.Series(range(1, len(valuations)+1), name="portfolio")
        securities = valuations.index.to_frame().reset_index(drop=True, inplace=False)
        portfolios = pd.concat([portfolios, securities], axis=1)
        valuations.index = pd.MultiIndex.from_frame(portfolios)
        return valuations

    @staticmethod
    def exposures(exposures, *args, **kwargs):
        series = exposures.set_index(feasibility_index, drop=True, inplace=False).squeeze()
        exposures = xr.DataArray.from_series(series).fillna(0)
        exposures = exposures.squeeze("ticker").squeeze("expire").squeeze("instrument")
        exposures = exposures.stack({"holdings": ["strike", "option", "position"]})
        return exposures

    @staticmethod
    def divestitures(valuations, *args, **kwargs):
        position = lambda cols: Variables.Positions.LONG if cols["position"] == Variables.Positions.SHORT else Variables.Positions.SHORT
        security = lambda cols: list(Variables.Securities(cols["security"]))
        dataframe = valuations.index.to_frame().set_index("portfolio", drop=True, inplace=False)
        for portfolio, series in dataframe.iterrows():
            options = series[feasibility_options].dropna(how="all", inplace=False).to_frame("strike")
            options = options.reset_index(names="security", drop=False, inplace=False)
            options[["instrument", "option", "position"]] = options.apply(security, axis=1, result_type="expand")
            options = options[[column for column in options.columns if column != "security"]]
            for key, value in series[feasibility_contract].to_dict().items():
                options[key] = value
            options["position"] = options.apply(position, axis=1)
            options["quantity"] = 1
            index = [column for column in options.columns if column != "quantity"]
            options = options.set_index(index, drop=True, inplace=False).squeeze()
            options = xr.DataArray.from_series(options).fillna(0)
            options = options.squeeze("ticker").squeeze("expire").squeeze("instrument")
            options = options.stack({"holdings": ["strike", "option", "position"]})
            yield portfolio, options

    @staticmethod
    def portfolios(exposures, divestitures, *args, **kwargs):
        divestitures = {key: xr.align(value, exposures, join="right")[0].fillna(0) for key, value in divestitures.items()}
        portfolios = {key: exposures - value for key, value in divestitures.items()}
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
        valuations = valuations.stack("scenario")
        valuations = valuations.reset_index(drop=False, inplace=False)
        valuations = valuations.set_index("portfolio", drop=True, inplace=False)
        valuations = pd.merge(valuations, stable, how="left", on="portfolio")
        valuations = valuations.where(valuations["stable"]).dropna(how="all", inplace=False)
        valuations = valuations.reset_index(drop=True, inplace=False).drop("stable", axis=1)
        return valuations

    @property
    def calculation(self): return self.__calculation
    @property
    def valuation(self): return self.__valuation



