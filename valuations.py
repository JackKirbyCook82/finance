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

from finance.variables import Querys, Variables, Securities
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


arbitrage_index = {security: np.float32 for security in list(map(str, Securities))} | {"strategy": str, "valuation": str, "scenario": str, "ticker": str, "expire": np.datetime64}
arbitrage_columns = {"current": np.datetime64, "apy": np.float32, "npv": np.float32, "cost": np.float32, "size": np.float32, "underlying": np.float32}


class ArbitrageFile(File, variable=Variables.Valuations.ARBITRAGE, query=Querys.Contract, datatype=pd.DataFrame, header=arbitrage_index | arbitrage_columns):
    pass


class ValuationFilter(Filter, variables=[Variables.Valuations.ARBITRAGE], query=Querys.Contract):
    @kwargsdispatcher("variable")
    def filter(self, dataframe, *args, variable, **kwargs): raise ValueError(variable)
    @filter.register.value(Variables.Valuations.ARBITRAGE)
    def arbitrage(self, dataframe, *args, **kwargs):
        columns = set(arbitrage_columns.keys())
        index = set(dataframe.columns) - ({"scenario"} | set(columns))
        dataframe = dataframe.pivot(columns="scenario", index=index)
        dataframe = super().filter(dataframe, *args, stack=["minimum"], **kwargs)
        dataframe = dataframe.stack("scenario")
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        return dataframe


class ValuationEquation(Equation): pass
class ArbitrageEquation(ValuationEquation):
    tau = Variable("tau", "tau", np.int32, function=lambda ti, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(ti, "ns"), "D") / np.timedelta64(1, "D"))
    inc = Variable("inc", "income", np.float32, function=lambda vi, vτ: + np.maximum(vi, 0) + np.maximum(vτ, 0))
    exp = Variable("exp", "cost", np.float32, function=lambda vi, vτ: - np.minimum(vi, 0) - np.minimum(vτ, 0))
    npv = Variable("npv", "npv", np.float32, function=lambda inc, exp, tau, ρ: np.divide(inc, np.power(1 + ρ, tau / 365)) - exp)
    irr = Variable("irr", "irr", np.float32, function=lambda inc, exp, tau: np.power(np.divide(inc, exp), np.power(tau, -1)) - 1)
    apy = Variable("apy", "apy", np.float32, function=lambda irr, tau: np.power(irr + 1, np.power(tau / 365, -1)) - 1)
    tτ = Variable("tτ", "expire", np.datetime64, position=0, locator="expire")
    ti = Variable("ti", "date", np.datetime64, position=0, locator="date")
    vi = Variable("vi", "spot", np.float32, position=0, locator="spot")
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
        yield strategies["underlying"]
        yield strategies["current"]
        yield strategies["size"]

class MinimumArbitrageCalculation(ArbitrageCalculation, scenario=Variables.Scenarios.MINIMUM, equation=MinimumArbitrageEquation): pass
class MaximumArbitrageCalculation(ArbitrageCalculation, scenario=Variables.Scenarios.MAXIMUM, equation=MaximumArbitrageEquation): pass


class ValuationCalculator(Processor):
    def __init__(self, *args, name=None, **kwargs):
        assert kwargs["calculation"] in list(Variables.Valuations)
        super().__init__(*args, name=name, **kwargs)
        calculations = {variables["scenario"]: calculation for variables, calculation in ODict(list(ValuationCalculation)).items() if variables["valuation"] is kwargs["calculation"]}
        self.__calculations = {str(scenario.name).lower(): calculation(*args, **kwargs) for scenario, calculation in calculations.items()}
        self.__variables = lambda **mapping: {key: xr.Variable(key, [value]).squeeze(key) for key, value in mapping.items()}
        self.__calculation = str(kwargs["calculation"].name).lower()

    def execute(self, contents, *args, **kwargs):
        strategies = contents["strategies"]
        assert isinstance(strategies, list) and all([isinstance(dataset, xr.Dataset) for dataset in strategies])
        valuations = ODict(list(self.calculate(strategies, *args, **kwargs)))
        valuations = ODict(list(self.flatten(valuations, *args, **kwargs)))
        valuations = {self.calculation: pd.concat(list(valuations.values()), axis=0)}
        if bool(valuations["valuations"].empty):
            return
        yield contents | valuations

    def calculate(self, strategies, *args, **kwargs):
        for scenario, calculation in self.calculations.items():
            variables = self.variables(valuation=self.calculation, scenario=scenario)
            datasets = [calculation(dataset, *args, **kwargs) for dataset in strategies]
            datasets = [dataset.assign_coords(variables).expand_dims("scenario") for dataset in datasets]
            yield scenario, datasets

    @staticmethod
    def flatten(valuations, *args, **kwargs):
        for scenario, datasets in valuations.items():
            datasets = [dataset.drop_vars(["instrument", "position"], errors="ignore") for dataset in datasets]
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
    def calculation(self): return self.__calculation
    @property
    def variables(self): return self.__variables


class ValuationFiles(object):
    Arbitrage = ArbitrageFile




