# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Valuation Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC
from itertools import product
from collections import OrderedDict as ODict

from finance.variables import Variables, Contract
from support.calculations import Variable, Equation, Calculation
from support.meta import ParametersMeta
from support.filtering import Filter
from support.mixins import Sizing

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationFilter", "ValuationCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class ValuationEquation(Equation): pass
class ArbitrageEquation(ValuationEquation):
    xα = Variable("xα", str(Variables.Securities.Stock.Long), np.float32, function=lambda xo, stg: np.round(xo, decimals=2) if Variables.Securities.Stock.Long in list(stg.stocks) else np.NaN)
    xβ = Variable("xβ", str(Variables.Securities.Stock.Short), np.float32, function=lambda xo, stg: np.round(xo, decimals=2) if Variables.Securities.Stock.Short in list(stg.stocks) else np.NaN)
    tau = Variable("tau", "tau", np.int32, function=lambda to, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))
    irr = Variable("irr", "irr", np.float32, function=lambda inc, exp, tau: np.power(np.divide(inc, exp), np.power(tau, -1)) - 1)
    npv = Variable("npv", "npv", np.float32, function=lambda inc, exp, tau, ρ: np.divide(inc, np.power(1 + ρ, tau / 365)) - exp)
    apy = Variable("apy", "apy", np.float32, function=lambda irr, tau: np.power(irr + 1, np.power(tau / 365, -1)) - 1)
    inc = Variable("inc", "income", np.float32, function=lambda vo, vτ: + np.maximum(vo, 0) + np.maximum(vτ, 0))
    exp = Variable("exp", "cost", np.float32, function=lambda vo, vτ: - np.minimum(vo, 0) - np.minimum(vτ, 0))
    stg = Variable("stg", "strategy", Variables.Strategies, position=0, locator="strategy")
    xo = Variable("xo", "underlying", np.float32, position=0, locator="underlying")
    tτ = Variable("tτ", "expire", np.datetime64, position=0, locator="expire")
    to = Variable("to", "current", np.datetime64, position=0, locator="current")
    vo = Variable("vo", "spot", np.float32, position=0, locator="spot")
    ρ = Variable("ρ", "discount", np.float32, position="discount")

class MinimumArbitrageEquation(ArbitrageEquation):
    vτ = Variable("vτ", "minimum", np.float32, position=0, locator="minimum")

class MaximumArbitrageEquation(ArbitrageEquation):
    vτ = Variable("vτ", "maximum", np.float32, position=0, locator="maximum")


class ValuationCalculation(Calculation, ABC, fields=["valuation", "scenario"]): pass
class ArbitrageCalculation(ValuationCalculation, ABC, valuation=Variables.Valuations.ARBITRAGE):
    def execute(self, strategies, *args, discount, **kwargs):
        equation = self.equation(*args, **kwargs)
        yield equation.npv(strategies, discount=discount)
        yield equation.apy(strategies, discount=discount)
        yield equation.exp(strategies, discount=discount)
        yield equation.tau(strategies)
        yield strategies["underlying"]
        yield strategies["current"]
        yield strategies["size"]

class MinimumArbitrageCalculation(ArbitrageCalculation, scenario=Variables.Scenarios.MINIMUM, equation=MinimumArbitrageEquation): pass
class MaximumArbitrageCalculation(ArbitrageCalculation, scenario=Variables.Scenarios.MAXIMUM, equation=MaximumArbitrageEquation): pass


class ValuationFilter(Filter):
    def calculate(self, valuations, *args, variable, **kwargs):
        assert isinstance(variable, Contract) and isinstance(valuations, pd.DataFrame)
        valuations = self.filter(valuations, *args, variable=variable, **kwargs)
        valuations = valuations.reset_index(drop=True, inplace=False)
        return valuations


class ValuationAxes(object, metaclass=ParametersMeta):
    securities = list(map(str, Variables.Securities))
    options = list(map(str, Variables.Securities.Options))
    stocks = list(map(str, Variables.Securities.Stocks))
    scenarios = list(Variables.Scenarios)
    security = ["instrument", "option", "position"]
    arbitrage = ["apy", "npv", "cost"]
    contract = ["ticker", "expire"]

    def __init__(self, *args, valuation, **kwargs):
        valuation = str(valuation).lower()
        index = self.contract + ["valuation", "strategy"] + self.options
        unstacked = ["current", "size", "tau", "underlying"] + ["status", "priority"]
        stacked = list(product(getattr(self, valuation), self.scenarios))
        unstacked = list(product(unstacked, [""]))
        index = list(product(index, [""]))
        self.header = list(index) + list(stacked) + list(unstacked)
        self.valuation = str(valuation).lower()
        self.columns = list(stacked) + list(unstacked)
        self.index = list(index)

    def parse(self, dataframe):
        columns = getattr(self, self.valuation)
        index = set(dataframe.columns) - ({"scenario"} | set(columns))
        dataframe = dataframe.pivot(index=list(index), columns="scenario")
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        return dataframe

    def format(self, dataframe, *args, **kwargs):
        columns = getattr(self, self.valuation)
        scenarios = list(map(str, self.scenarios))
        columns = list(product(columns, scenarios))
        index = set(dataframe.columns) - set(columns)
        dataframe = dataframe.set_index(list(index), drop=True, inplace=False)
        dataframe = dataframe.stack("scenario").reset_index(drop=False, inplace=False)
        return dataframe


class ValuationCalculator(Sizing):
    def __init__(self, *args, valuation, **kwargs):
        super().__init__(*args, **kwargs)
        calculations = {variables["scenario"]: calculation for variables, calculation in ODict(list(ValuationCalculation)).items() if variables["valuation"] is kwargs["valuation"]}
        self.__calculations = {scenario: calculation(*args, **kwargs) for scenario, calculation in calculations.items()}
        self.__variables = lambda **mapping: {key: xr.Variable(key, [value]).squeeze(key) for key, value in mapping.items()}
        self.__axes = ValuationAxes(*args, valuation=valuation, **kwargs)
        self.__logger = __logger__

    def calculate(self, contract, strategies, *args, **kwargs):
        assert isinstance(contract, Contract) and all([isinstance(dataset, xr.Dataset) for dataset in strategies])
        valuations = ODict(list(self.valuations(strategies, *args, **kwargs)))
        valuations = ODict(list(self.flatten(valuations, *args, **kwargs)))
        valuations = pd.concat(list(valuations.values()), axis=0) if bool(valuations) else pd.DataFrame()
        valuations = self.axes.parse(valuations) if bool(valuations) else pd.DataFrame(columns=self.axes.header)
        size = self.size(valuations)
        string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
        self.logger.info(string)
        return valuations

    def valuations(self, strategies, *args, **kwargs):
        if not bool(strategies): return
        for scenario, calculation in self.calculations.items():
            valuations = [calculation(dataset, *args, **kwargs) for dataset in strategies]
            assert all([isinstance(dataset, xr.Dataset) for dataset in valuations])
            if not bool(valuations): continue
            variables = self.variables(valuation=self.axes.valuation, scenario=scenario)
            valuations = [dataset.assign_coords(variables).expand_dims("scenario") for dataset in valuations]
            yield scenario, valuations

    def flatten(self, valuations, *args, **kwargs):
        if not bool(valuations): return
        for scenario, datasets in valuations.items():
            datasets = [dataset.drop_vars(self.axes.security, errors="ignore") for dataset in datasets]
            datasets = [dataset.expand_dims(list(set(iter(dataset.coords)) - set(iter(dataset.dims)))) for dataset in datasets]
            dataframes = [dataset.to_dataframe().dropna(how="all", inplace=False) for dataset in datasets]
            dataframes = [dataframe.reset_index(drop=False, inplace=False) for dataframe in dataframes]
            if not bool(dataframes) or all([bool(dataframe.empty) for dataframe in dataframes]): continue
            dataframe = pd.concat(dataframes, axis=0).reset_index(drop=True, inplace=False)
            yield scenario, dataframe

    @property
    def calculations(self): return self.__calculations
    @property
    def variables(self): return self.__variables
    @property
    def logger(self): return self.__logger
    @property
    def axes(self): return self.__axes



