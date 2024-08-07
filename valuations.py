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

from finance.variables import Variables
from support.calculations import Variable, Equation, Calculation
from support.dispatchers import kwargsdispatcher
from support.pipelines import Processor
from support.filtering import Filter
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationFiles", "ValuationFilter", "ValuationCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


valuation_dates = {"current": "%Y%m%d-%H%M", "expire": "%Y%m%d"}
valuation_parsers = {"strategy": Variables.Strategies, "valuation": Variables.Valuations, "scenario": Variables.Scenarios}
valuation_formatters = {"strategy": int, "valuation": int, "scenario": int}
valuation_types = {"ticker": str, "apy": np.float32, "npv": np.float32, "cost": np.float32, "size": np.float32, "tau": np.int32} | {security: np.float32 for security in list(map(str, Variables.Securities))}
valuation_filename = lambda query: "_".join([str(query.ticker).upper(), str(query.expire.strftime("%Y%m%d"))])
valuation_formatter = lambda self, *, results, elapsed, **kw: f"{str(self.title)}: {repr(self)}|{str(results[Variables.Querys.CONTRACT])}[{elapsed:.02f}s]"
valuation_parameters = dict(datatype=pd.DataFrame, filename=valuation_filename, dates=valuation_dates, parsers=valuation_parsers, formatters=valuation_formatters, types=valuation_types)
valuation_index = ["ticker", "expire", "strategy", "valuation", "scenario"] + list(map(str, Variables.Securities.Options))
valuation_columns = ["current", "apy", "npv", "cost", "size", "tau", "underlying"]


class ArbitrageFile(File, variable=Variables.Valuations.ARBITRAGE, header=valuation_index + valuation_columns, **valuation_parameters): pass
class ValuationFiles(object): Arbitrage = ArbitrageFile


class ValuationFilter(Filter, variables=[Variables.Valuations.ARBITRAGE], formatter=valuation_formatter):
    @kwargsdispatcher("variable")
    def filter(self, dataframe, *args, variable, **kwargs): raise ValueError(variable)
    @filter.register.value(Variables.Valuations.ARBITRAGE)
    def arbitrage(self, dataframe, *args, **kwargs):
        variable = Variables.Scenarios.MINIMUM
        index = set(dataframe.columns) - ({"scenario"} | set(valuation_columns))
        dataframe = dataframe.pivot(index=list(index), columns="scenario")
        dataframe = super().filter(dataframe, *args, stack=[variable], **kwargs)
        dataframe = dataframe.stack("scenario")
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        return dataframe


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


class ValuationCalculator(Processor, formatter=valuation_formatter):
    def __init__(self, *args, valuation, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        calculations = {variables["scenario"]: calculation for variables, calculation in ODict(list(ValuationCalculation)).items() if variables["valuation"] is kwargs.get(calculation, Variables.Valuations.ARBITRAGE)}
        self.__calculations = {scenario: calculation(*args, **kwargs) for scenario, calculation in calculations.items()}
        self.__valuation = valuation

    def processor(self, contents, *args, **kwargs):
        strategies = contents[Variables.Datasets.STRATEGY]
        assert isinstance(strategies, list) and all([isinstance(dataset, xr.Dataset) for dataset in strategies])
        valuations = ODict(list(self.calculate(strategies, *args, **kwargs)))
        valuations = ODict(list(self.flatten(valuations, *args, **kwargs)))
        if not bool(valuations):
            return
        valuations = pd.concat(list(valuations.values()), axis=0)
        valuations = {self.valuation: valuations}
        yield contents | valuations

    def calculate(self, valuations, *args, **kwargs):
        function = lambda **mapping: {key: xr.Variable(key, [value]).squeeze(key) for key, value in mapping.items()}
        for scenario, calculation in self.calculations.items():
            variables = function(valuation=self.valuation, scenario=scenario)
            datasets = [calculation(dataset, *args, **kwargs) for dataset in valuations]
            datasets = [dataset.assign_coords(variables).expand_dims("scenario") for dataset in datasets]
            yield scenario, datasets

    @staticmethod
    def flatten(options, *args, **kwargs):
        for scenario, datasets in options.items():
            datasets = [dataset.drop_vars(["instrument", "option", "position"], errors="ignore") for dataset in datasets]
            datasets = [dataset.expand_dims(list(set(iter(dataset.coords)) - set(iter(dataset.dims)))) for dataset in datasets]
            dataframes = [dataset.to_dataframe().dropna(how="all", inplace=False) for dataset in datasets]
            dataframes = [dataframe.reset_index(drop=False, inplace=False) for dataframe in dataframes]
            if not bool(dataframes) or all([bool(dataframe.empty) for dataframe in dataframes]):
                continue
            dataframe = pd.concat(dataframes, axis=0)
            yield scenario, dataframe

    @property
    def calculations(self): return self.__calculations
    @property
    def valuation(self): return self.__valuation




