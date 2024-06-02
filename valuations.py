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
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from finance.variables import Contract, Securities, Valuations, Scenarios
from support.calculations import Variable, Equation, Calculation
from support.files import FileDirectory, FileQuery, FileData
from support.dispatchers import kwargsdispatcher
from support.pipelines import Processor
from support.filtering import Filter

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ArbitrageFile", "ValuationFilter", "ValuationCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


Header = ntuple("Header", "index columns")
securities_index = {option: str for option in list(map(str, Securities))}
arbitrage_index = {"strategy": str, "valuation": str, "scenario": str, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
arbitrage_columns = {"apy": np.float32, "npv": np.float32, "cost": np.float32, "size": np.float32, "underlying": np.float32}
valuation_headers = {"arbitrage": Header(securities_index | arbitrage_columns, arbitrage_columns)}
arbitrage_data = FileData.Dataframe(header=securities_index | arbitrage_index | arbitrage_columns)
contract_query = FileQuery("contract", Contract.tostring, Contract.fromstring)


class ArbitrageFile(FileDirectory, variable="arbitrage", query=contract_query, data=arbitrage_data):
    pass


class ValuationFilter(Filter):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__valuations = list(valuation_headers.keys())

    def execute(self, contents, *args, **kwargs):
        contract, valuations = str(contents["contract"]), {valuation: contents[valuation] for valuation in self.valuations if valuation in contents.keys()}
        valuations = ODict(list(self.calculate(valuations, *args, contract=contract, **kwargs)))
        yield contents | dict(valuations)

    def calculate(self, valuations, *args, contract, **kwargs):
        for valuation, dataframe in valuations.items():
            prior = self.size(dataframe)

            print(dataframe)
            print(dataframe.columns)
            print(dataframe.index)
            raise Exception()

            dataframe = self.filter(dataframe, *args, valuation=valuation, **kwargs)
            post = self.size(dataframe)
            __logger__.info(f"Filter: {repr(self)}|{contract}|{valuation}[{prior:.0f}|{post:.0f}]")
            yield valuation, dataframe

    @kwargsdispatcher("valuation")
    def filter(self, dataframe, *args, valuation, **kwargs): raise ValueError(valuation)
    @filter.register.value(str(Valuations.ARBITRAGE.name).lower())
    def arbitrage(self, dataframe, *args, **kwargs):
        index = set(dataframe.columns) - ({"scenario"} | set(self.columns))
        dataframe = dataframe.pivot(columns="scenario", index=index)
        mask = self.mask(dataframe, variable="minimum")
        dataframe = self.where(dataframe, mask)
        dataframe = dataframe.stack("scenario")
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        return dataframe

    @property
    def valuations(self): return self.__valuations


class ValuationHeader(Processor):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__valuations = dict(valuation_headers)

    def execute(self, contents, *args, **kwargs):
        valuations = {valuation: contents[valuation] for valuation in self.valuations if valuation in contents.keys()}
        valuations = ODict(list(self.calculate(valuations, *args, **kwargs)))
        yield contents | dict(valuations)

    def calculate(self, valuations, *args, **kwargs):
        for valuation, dataframe in valuations.items():
            header = self.valuations[valuation]
            index = [value for value in header.index if value in dataframe.columns]
            dataframe = dataframe.set_index(index, drop=True, inplace=False)
            dataframe = dataframe[header.columns]
            yield valuation, dataframe

    @property
    def valuations(self): return self.__valuations


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


class ValuationCalculator(Processor):
    def __init__(self, *args, valuation, name=None, **kwargs):
        assert valuation in list(Valuations)
        super().__init__(*args, name=name, **kwargs)
        calculations = {variables["scenario"]: calculation for variables, calculation in ODict(list(ValuationCalculation)).items() if variables["valuation"] is valuation}
        valuation = str(valuation.name).lower()
        self.__calculations = {str(scenario.name).lower(): calculation(*args, **kwargs) for scenario, calculation in calculations.items()}
        self.__variables = lambda **mapping: {key: xr.Variable(key, [value]).squeeze(key) for key, value in mapping.items()}
        self.__valuation = valuation

    def execute(self, contents, *args, **kwargs):
        strategies = contents["strategies"]
        assert isinstance(strategies, list) and all([isinstance(dataset, xr.Dataset) for dataset in strategies])
        valuations = ODict(list(self.calculate(strategies, *args, **kwargs)))
        valuations = ODict(list(self.flatten(valuations, *args, **kwargs)))
        valuations = {self.valuation: pd.concat(list(valuations.values()), axis=0)}
        yield contents | valuations

    def calculate(self, strategies, *args, **kwargs):
        for scenario, calculation in self.calculations.items():
            variables = self.variables(valuation=self.valuation, scenario=scenario)
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
            dataframe = pd.concat(dataframes, axis=0)
            yield scenario, dataframe

    @property
    def calculations(self): return self.__calculations
    @property
    def variables(self): return self.__variables
    @property
    def valuation(self): return self.__valuation



