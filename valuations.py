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
from collections import OrderedDict as ODict

from finance.variables import Variables, Contract
from support.calculations import Variable, Equation, Calculation
from support.meta import ParametersMeta
from support.pipelines import Processor
from support.filtering import Filter
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationFiles", "ValuationFilter", "ValuationCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Axes:
    securities = list(map(str, Variables.Securities))
    options = list(map(str, Variables.Securities.Options))
    stocks = list(map(str, Variables.Securities.Stocks))
    stacking = {Variables.Valuations.ARBITRAGE: {"apy", "npv", "cost"}}
    arbitrage = ["apy", "npv", "cost", "size", "tau", "underlying", "current"]
    security = ["instrument", "option", "position"]
    contract = ["ticker", "expire"]

class Parameters(metaclass=ParametersMeta):
    types = {"ticker": str, "apy": np.float32, "npv": np.float32, "cost": np.float32, "size": np.float32, "tau": np.int32} | {security: np.float32 for security in Axes.securities}
    parsers = {"strategy": Variables.Strategies, "valuation": Variables.Valuations, "scenario": Variables.Scenarios}
    formatters = {"strategy": int, "valuation": int, "scenario": int}
    dates = {"current": "%Y%m%d-%H%M", "expire": "%Y%m%d"}
    filename = lambda variable: "_".join([str(variable.ticker).upper(), str(variable.expire.strftime("%Y%m%d"))])
    datatype = pd.DataFrame

class Headers:
    arbitrage = Axes.contract + ["valuation", "scenario", "strategy"] + Axes.options + Axes.arbitrage


class ArbitrageFile(File, variable=Variables.Valuations.ARBITRAGE, header=Headers.arbitrage, **dict(Parameters)):
    def format(self, dataframe, *args, **kwargs):
        index = [column for column in dataframe.columns if column[0] not in Axes.stacking[Variables.Valuations.ARBITRAGE]]
        dataframe = dataframe.set_index(list(index), drop=True, inplace=False)
        dataframe = dataframe.stack("scenario").reset_index(drop=False, inplace=False)
        dataframe = dataframe.rename(columns={column: column[0] for column in index})
        return dataframe

    def parse(self, dataframe, *args, **kwargs):
        index = set(dataframe.columns) - ({"scenario"} | Axes.stacking[Variables.Valuations.ARBITRAGE])
        dataframe = dataframe.pivot(index=list(index), columns="scenario")
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        return dataframe

class ValuationFiles(object):
    Arbitrage = ArbitrageFile


class ValuationFilter(Filter, variable=Variables.Querys.CONTRACT):
    def __init__(self, *args, valuation, **kwargs):
        super().__init__(*args, **kwargs)
        self.__valuation = valuation

    def processor(self, contents, *args, **kwargs):
        parameters = dict(variable=contents[self.variable])
        valuations = list(self.calculate(contents, *args, **parameters, **kwargs))
        if not bool(valuations): return
        yield contents | ODict(valuations)

    def calculate(self, contents, *args, **kwargs):
        valuations = contents.get(self.valuation, None)
        if valuations is None: return
        if bool(valuations.empty): return
        valuations = self.filter(valuations, *args, **kwargs)
        valuations = valuations.reset_index(drop=True, inplace=False)
        if bool(valuations.empty): return
        yield self.valuation, valuations

    @property
    def valuation(self): return self.__valuation


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


class ValuationCalculator(Processor, title="Calculated", variable=Variables.Querys.CONTRACT):
    def __init__(self, *args, valuation, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        calculations = {variables["scenario"]: calculation for variables, calculation in ODict(list(ValuationCalculation)).items() if variables["valuation"] is valuation}
        self.__calculations = {scenario: calculation(*args, **kwargs) for scenario, calculation in calculations.items()}
        self.__variables = lambda **mapping: {key: xr.Variable(key, [value]).squeeze(key) for key, value in mapping.items()}
        self.__valuation = valuation

    def processor(self, contents, *args, **kwargs):
        strategies = contents[Variables.Datasets.STRATEGY]
        assert isinstance(strategies, list) and all([isinstance(dataset, xr.Dataset) for dataset in strategies])
        valuations = list(self.calculate(strategies, *args, **kwargs))
        valuations = list(self.parser(valuations, *args, **kwargs))
        valuations = list(self.valuations(valuations, *args, **kwargs))
        if not bool(valuations): return
        yield contents | ODict(valuations)

    def calculate(self, strategies, *args, **kwargs):
        for scenario, calculation in self.calculations.items():
            valuations = [calculation(dataset, *args, **kwargs) for dataset in strategies]
            if not bool(valuations): continue
            variables = self.variables(valuation=self.valuation, scenario=scenario)
            valuations = [dataset.assign_coords(variables).expand_dims("scenario") for dataset in valuations]
            yield valuations

    def valuations(self, valuations, *args, **kwargs):
        if not bool(valuations): return
        valuations = pd.concat(valuations, axis=0)
        index = set(valuations.columns) - ({"scenario"} | Axes.stacking[self.valuation])
        valuations = valuations.pivot(index=list(index), columns="scenario")
        valuations = valuations.reset_index(drop=False, inplace=False)
        yield self.valuation, valuations

    @staticmethod
    def parser(valuations, *args, **kwargs):
        if not bool(valuations): return
        for datasets in valuations:
            datasets = [dataset.drop_vars(Axes.security, errors="ignore") for dataset in datasets]
            datasets = [dataset.expand_dims(list(set(iter(dataset.coords)) - set(iter(dataset.dims)))) for dataset in datasets]
            dataframes = [dataset.to_dataframe().dropna(how="all", inplace=False) for dataset in datasets]
            dataframes = [dataframe.reset_index(drop=False, inplace=False) for dataframe in dataframes]
            if not bool(dataframes) or all([bool(dataframe.empty) for dataframe in dataframes]): continue
            dataframe = pd.concat(dataframes, axis=0).reset_index(drop=True, inplace=False)
            yield dataframe

    @property
    def calculations(self): return self.__calculations
    @property
    def variables(self): return self.__variables
    @property
    def valuation(self): return self.__valuation




