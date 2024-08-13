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
__all__ = ["FeasibilityCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


feasibility_formatter = lambda self, *, results, elapsed, **kw: f"{str(self.title)}: {repr(self)}|{str(results[Variables.Querys.CONTRACT])}[{elapsed:.02f}s]"
feasibility_index = ["ticker", "expire", "strike", "instrument", "option", "position"]
feasibility_columns = ["current", "apy", "npv", "cost", "size", "underlying"]
feasibility_stacking = {Variables.Valuations.ARBITRAGE: {"apy", "npv", "cost"}}
feasibility_options = list(map(str, Variables.Securities.Options))
feasibility_stocks = list(map(str, Variables.Securities.Stocks))
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


class FeasibilityCalculator(Processor, formatter=feasibility_formatter):
    def __init__(self, *args, valuation, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__calculation = FeasibilityCalculation(*args, **kwargs)
        self.__valuation = valuation

    def processor(self, contents, *args, **kwargs):
        valuations, exposures = contents[self.valuation], contents[Variables.Datasets.EXPOSURE]
        assert isinstance(valuations, pd.DataFrame) and isinstance(exposures, pd.DataFrame)
        valuations = self.valuations(valuations, *args, **kwargs)
        securities = self.securities(valuations, *args, **kwargs)

        print(exposures)
        print(securities)

        portfolios = self.portfolios(securities, *args, **kwargs)

        raise Exception()


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

    def valuations(self, valuations, *args, **kwargs):
        index = set(valuations.columns) - ({"scenario"} | feasibility_stacking[self.valuation])
        valuations = valuations.pivot(index=list(index), columns="scenario")
        valuations = valuations.reset_index(drop=False, inplace=False)
        return valuations

    @staticmethod
    def securities(valuations, *args, **kwargs):
        stocks = list(map(str, Variables.Securities.Stocks))
        strategy = lambda cols: list(map(str, cols["strategy"].stocks))
        function = lambda cols: {stock: cols["underlying"] if stock in strategy(cols) else np.NaN for stock in stocks}
        options = valuations[feasibility_contract + feasibility_options + ["valuation", "strategy", "underlying"]]
        options = options.droplevel("scenario", axis=1)
        stocks = options.apply(function, axis=1, result_type="expand")
        securities = pd.concat([options, stocks], axis=1)
        return securities[feasibility_contract + feasibility_options + feasibility_stocks]

    @staticmethod
    def portfolios(securities, *args, **kwargs):
        for portfolio, series in securities.iterrows():
            security = lambda cols: list(Variables.Securities(cols["security"])) + [1]

            print(series)

            options = series[feasibility_options].to_frame("strike")
            options = options.reset_index(names="security", drop=False, inplace=False)
            options[["instrument", "option", "position", "quantity"]] = options.apply(security, axis=1, result_type="expand")
            options = options[[column for column in options.columns if column != "security"]]
            for key, value in series[feasibility_contract].to_dict().items():
                options[key] = value

            print(options)

            stocks = series[feasibility_stocks].to_frame("strike")
            stocks = stocks.reset_index(names="security", drop=False, inplace=False)
            stocks[["instrument", "option", "position", "quantity"]] = stocks.apply(security, axis=1, result_type="expand")
            stocks = stocks[[column for column in stocks.columns if column != "security"]]
            for key, value in series[feasibility_contract].to_dict().items():
                stocks[key] = value

            print(stocks)

            


            raise Exception()

    @property
    def calculation(self): return self.__calculation
    @property
    def valuation(self): return self.__valuation

#    @staticmethod
#    def valuations(valuations, *args, **kwargs):
#        index = set(valuations.columns) - ({"scenario"} | set(feasibility_columns))
#        valuations = valuations.pivot(index=list(index), columns="scenario")
#        portfolios = pd.Series(range(1, len(valuations)+1), name="portfolio")
#        securities = valuations.index.to_frame().reset_index(drop=True, inplace=False)
#        portfolios = pd.concat([portfolios, securities], axis=1)
#        valuations.index = pd.MultiIndex.from_frame(portfolios)
#        return valuations

#    @staticmethod
#    def exposures(exposures, *args, **kwargs):
#        series = exposures.set_index(feasibility_index, drop=True, inplace=False).squeeze()
#        exposures = xr.DataArray.from_series(series).fillna(0)
#        exposures = exposures.squeeze("ticker").squeeze("expire").squeeze("instrument")
#        exposures = exposures.stack({"holdings": ["strike", "option", "position"]})
#        return exposures

#    @staticmethod
#    def divestitures(valuations, *args, **kwargs):
#        position = lambda cols: Variables.Positions.LONG if cols["position"] == Variables.Positions.SHORT else Variables.Positions.SHORT
#        security = lambda cols: list(Variables.Securities(cols["security"]))
#        dataframe = valuations.index.to_frame().set_index("portfolio", drop=True, inplace=False)
#        for portfolio, series in dataframe.iterrows():
#            options = series[feasibility_options].dropna(how="all", inplace=False).to_frame("strike")
#            options = options.reset_index(names="security", drop=False, inplace=False)
#            options[["instrument", "option", "position"]] = options.apply(security, axis=1, result_type="expand")
#            options = options[[column for column in options.columns if column != "security"]]
#            for key, value in series[feasibility_contract].to_dict().items():
#                options[key] = value
#            options["position"] = options.apply(position, axis=1)
#            options["quantity"] = 1
#            index = [column for column in options.columns if column != "quantity"]
#            options = options.set_index(index, drop=True, inplace=False).squeeze()
#            options = xr.DataArray.from_series(options).fillna(0)
#            options = options.squeeze("ticker").squeeze("expire").squeeze("instrument")
#            options = options.stack({"holdings": ["strike", "option", "position"]})
#            yield portfolio, options

#    @staticmethod
#    def portfolios(exposures, divestitures, *args, **kwargs):
#        divestitures = {key: xr.align(value, exposures, join="right")[0].fillna(0) for key, value in divestitures.items()}
#        portfolios = {key: exposures - value for key, value in divestitures.items()}
#        portfolios = [portfolio.assign_coords(portfolio=index) for index, portfolio in portfolios.items()]
#        portfolios = xr.concat([exposures.assign_coords(portfolio=0)] + portfolios, dim="portfolio")
#        portfolios = portfolios.to_dataset(name="quantity")
#        return portfolios

#    @staticmethod
#    def underlying(portfolios, *args, **kwargs):
#        underlying = np.unique(portfolios["strike"].values)
#        underlying = np.sort(np.concatenate([underlying - 0.001, underlying + 0.001]))
#        portfolios["underlying"] = underlying
#        return portfolios

#    @staticmethod
#    def stability(payoffs, *args, **kwargs):
#        payoffs = payoffs.sum(dim="holdings")
#        payoffs["maximum"] = payoffs["value"].max(dim="underlying")
#        payoffs["minimum"] = payoffs["value"].min(dim="underlying")
#        payoffs["bull"] = payoffs["trend"].isel(underlying=0)
#        payoffs["bear"] = payoffs["trend"].isel(underlying=-1)
#        payoffs["stable"] = (payoffs["bear"] == 0) & (payoffs["bull"] == 0)
#        stable = payoffs["stable"].to_series()
#        return stable

#    @staticmethod
#    def feasibility(valuations, stable, *args, **kwargs):
#        valuations = valuations.stack("scenario")
#        valuations = valuations.reset_index(drop=False, inplace=False)
#        valuations = valuations.set_index("portfolio", drop=True, inplace=False)
#        valuations = pd.merge(valuations, stable, how="left", on="portfolio")
#        valuations = valuations.where(valuations["stable"]).dropna(how="all", inplace=False)
#        valuations = valuations.reset_index(drop=True, inplace=False).drop("stable", axis=1)
#        return valuations

    @property
    def calculation(self): return self.__calculation
    @property
    def valuation(self): return self.__valuation

