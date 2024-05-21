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

from support.calculations import Variable, Equation, Calculation, Calculator
from support.dispatchers import kwargsdispatcher
from support.query import Data, Header, Query
from support.filtering import Filter
from support.files import Files

from finance.variables import Securities, Valuations, Scenarios

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ArbitrageFile", "ValuationFilter", "ValuationCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


arbitrage_index = {option: str for option in list(map(str, Securities.Options))} | {"strategy": str, "valuation": str, "scenario": str, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
arbitrage_columns = {"apy": np.float32, "npv": np.float32, "cost": np.float32, "size": np.float32, "underlying": np.float32}
arbitrage_header = Header(pd.DataFrame, index=list(arbitrage_index.keys()), columns=list(arbitrage_columns.keys()))
valuations_headers = ODict(list(dict(arbitrage=arbitrage_header).items()))


class ArbitrageFile(Files.Dataframe, variable="arbitrage", index=arbitrage_index, columns=arbitrage_columns):
    pass


class ValuationFilter(Filter):
    @Query()
    def execute(self, contents, *args, **kwargs):
        valuations = {key: contents[key] for key in valuations_headers.keys() if key in contents.keys()}
        valuations = ODict(list(self.calculate(valuations, *args, contract=contents["contract"], **kwargs)))
        yield valuations

    def calculate(self, valuations, *args, contract, **kwargs):
        for valuation, dataframe in valuations.items():
            variable = str(valuation.name).lower()
            prior = self.size(dataframe)
            dataframe = dataframe.reset_index(drop=False, inplace=False)
            dataframe = self.pivot(dataframe, *args, valuation=valuation, **kwargs)
            post = self.size(dataframe)
            __logger__.info(f"Filter: {repr(self)}|{str(contract)}|{str(variable)}[{prior:.0f}|{post:.0f}]")
            yield valuation, dataframe

    @kwargsdispatcher("valuation")
    def filter(self, dataframe, *args, valuation, **kwargs): raise ValueError(valuation)
    @filter.register.value(str(Valuations.ARBITRAGE.name).lower())
    def arbitrage(self, dataframe, *args, **kwargs):
        scenario = str(self.scenario.name).lower()
        index = set(dataframe.columns) - ({"scenario"} | set(self.columns))
        dataframe = dataframe.pivot(columns="scenario", index=index)
        mask = self.mask(dataframe, variable=scenario)
        dataframe = self.where(dataframe, mask)
        dataframe = dataframe.stack("scenario")
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        return dataframe


class ValuationEquation(Equation): pass
class ArbitrageEquation(ValuationEquation):
    tau = Variable("tau", "tau", np.int32, function=lambda to, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))
    inc = Variable("inc", "income", np.float32, function=lambda vo, vτ: + np.maximum(vo, 0) + np.maximum(vτ, 0))
    exp = Variable("exp", "cost", np.float32, function=lambda vo, vτ: - np.minimum(vo, 0) - np.minimum(vτ, 0))
    npv = Variable("npv", "npv", np.float32, function=lambda inc, exp, tau, ρ: np.divide(inc, np.power(1 + ρ, tau / 365)) - exp)
    irr = Variable("irr", "irr", np.float32, function=lambda inc, exp, tau: np.power(np.divide(inc, exp), np.power(tau, -1)) - 1)
    apy = Variable("apy", "apy", np.float32, function=lambda irr, tau: np.power(irr + 1, np.power(tau / 365, -1)) - 1)
    tτ = Variable("tτ", "expire", np.datetime64, position=0, locator="expire")
    to = Variable("to", "date", np.datetime64, position=0, locator="date")
    vo = Variable("vo", "spot", np.float32, position=0, locator="spot")
    ρ = Variable("ρ", "discount", np.float32, position="discount")

class MinimumArbitrageEquation(ArbitrageEquation):
    vτ = Variable("vτ", "minimum", np.float32, position=0, locator="minimum")

class MaximumArbitrageEquation(ArbitrageEquation):
    vτ = Variable("vτ", "maximum", np.float32, position=0, locator="maximum")


class ValuationCalculation(Calculation, ABC, fields=["valuation", "scenario"]): pass
class ArbitrageCalculation(ValuationCalculation, ABC, valuation=Valuations.ARBITRAGE):
    def execute(self, strategies, *args, discount, **kwargs):
        equation = self.equation(*args, **kwargs)
        yield equation.npv(strategies, discount=discount)
        yield equation.apy(strategies, discount=discount)
        yield equation.exp(strategies, discount=discount)
        yield strategies["underlying"]
        yield strategies["size"]

class MinimumArbitrageCalculation(ArbitrageCalculation, scenario=Scenarios.MINIMUM, equation=MinimumArbitrageEquation): pass
class MaximumArbitrageCalculation(ArbitrageCalculation, scenario=Scenarios.MAXIMUM, equation=MaximumArbitrageEquation): pass


class ValuationCalculator(Data, Calculator, calculations=ValuationCalculation):
    @Query()
    def execute(self, strategies, *args, **kwargs):
        assert isinstance(strategies, list) and all([isinstance(dataset, xr.Dataset) for dataset in strategies])
        valuations = ODict(list(self.calculate(strategies, *args, **kwargs)))
        valuations = {valuation: [self.flatten(dataset) for dataset in datasets] for valuation, datasets in valuations.items()}
        valuations = {valuation: pd.concat(dataframe, axis=1) for valuation, dataframe in valuations.items()}
        yield valuations

    def calculate(self, strategies, *args, **kwargs):
        function = lambda key, value: {key: xr.Variable(key, [value]).squeeze(key)}
        for variables, calculations in self.calculations.items():
            variable = str(variables["valuation"]).lower()
            variables = lambda scenario: function("valuation", variable) | function("scenario", str(scenario.name).lower())
            results = {scenario: [calculation(dataset, *args, **kwargs) for dataset in strategies] for scenario, calculation in calculations.items()}
            results = [dataset.assign_coords(variables(scenario)).expand_dims("scenario") for scenario, datasets in results.items() for dataset in datasets]
            yield variable, results

    @staticmethod
    def flatten(dataset):
        dataset = dataset.drop_vars(["instrument", "position"], errors="ignore")
        dataset = dataset.expand_dims(list(set(iter(dataset.coords)) - set(iter(dataset.dims))))
        dataframe = dataset.to_dataframe().dropna(how="all", inplace=False)
        return dataframe



