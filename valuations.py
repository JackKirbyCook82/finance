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
from functools import reduce
from itertools import product
from collections import namedtuple as ntuple

from finance.variables import Variables, Querys
from support.meta import ParametersMeta, RegistryMeta
from support.calculations import Calculation, Equation, Variable
from support.tables import Writer, Reader, Table, View
from support.mixins import Function, Emptying, Sizing, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationCalculator", "ValuationReader", "ValuationWriter", "ValuationTable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class ValuationFormatting(metaclass=ParametersMeta):
    order = ["ticker", "expire", "valuation", "strategy"] + list(map(str, Variables.Securities.Options)) + ["apy", "npv", "cost"] + ["size", "status"]
    formats = {"apy": lambda column: f"{column * 100:.02f}%" if np.isfinite(column) else "InF", "status": lambda status: str(status), "size": lambda size: f"{size:.0f}"}
    numbers = lambda column: f"{column:.02f}"


class ValuationView(View, ABC, datatype=pd.DataFrame, **dict(ValuationFormatting)): pass
class ValuationTable(Table, ABC, datatype=pd.DataFrame, viewtype=ValuationView):
    def obsolete(self, contract, *args, **kwargs):
        assert isinstance(contract, Querys.Contract)
        contract = lambda table: [table[key] == value for key, value in contract.items()]
        status = lambda table: table["status"] == Variables.Status.PROSPECT
        function = lambda table: reduce(lambda x, y: x & y, contract(table) + status(table))
        return self.extract(function)

    def rejected(self, *args, tenure=None, **kwargs):
        timeout = lambda table: (pd.to_datetime("now") - table["current"]) >= tenure if (tenure is not None) else False
        rejected = lambda table: table["status"] == Variables.Status.REJECTED
        abandoned = lambda table: table["status"] == Variables.Status.ABANDONED
        function = lambda table: rejected(table) | abandoned(table) | timeout(table)
        return self.extract(function)

    def accepted(self, *args, **kwargs):
        function = lambda table: table["status"] == Variables.Status.ACCEPTED
        return self.extract(function)


class ValuationHeader(object):
    def __init__(self, *args, valuation, **kwargs):
        assert valuation in list(Variables.Valuations)
        valuations = {Variables.Valuations.ARBITRAGE: ["apy", "npv", "cost"]}
        stacked = lambda cols: list(product(cols, list(Variables.Scenarios)))
        unstacked = lambda cols: list(product(cols, [""]))
        columns = stacked(valuations[valuation]) + unstacked(["underlying", "size", "current"] + ["status", "priority"])
        index = unstacked(["valuation", "strategy"] + list(Variables.Contract) + list(map(str, Variables.Securities.Options)))
        self.__columns = columns
        self.__index = index

    @property
    def columns(self): return self.__columns
    @property
    def index(self): return self.__index


class ValuationEquation(Equation, ABC, metaclass=RegistryMeta):
    tau = Variable("tau", np.int32, function=lambda to, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))
    inc = Variable("income", np.float32, function=lambda vo, vτ: + np.maximum(vo, 0) + np.maximum(vτ, 0))
    exp = Variable("cost", np.float32, function=lambda vo, vτ: - np.minimum(vo, 0) - np.minimum(vτ, 0))

    xo = Variable("underlying", np.float32, locator="underlying")
    tτ = Variable("expire", np.datetime64, locator="expire")
    to = Variable("current", np.datetime64, locator="current")
    vo = Variable("spot", np.float32, locator="spot")
    ρ = Variable("discount", np.float32, locator="discount")

class ArbitrageCalculation(ValuationEquation, ABC):
    irr = Variable("irr", np.float32, function=lambda inc, exp, tau: np.power(np.divide(inc, exp), np.power(tau, -1)) - 1)
    npv = Variable("npv", np.float32, function=lambda inc, exp, tau, ρ: np.divide(inc, np.power(1 + ρ, tau / 365)) - exp)
    apy = Variable("apy", np.float32, function=lambda irr, tau: np.power(irr + 1, np.power(tau / 365, -1)) - 1)

class MinimumArbitrageCalculation(ArbitrageCalculation, register=(Variables.Valuations.ARBITRAGE, Variables.Scenarios.MINIMUM)):
    vτ = Variable("minimum", np.float32, locator="minimum")

class MaximumArbitrageCalculation(ArbitrageCalculation, register=(Variables.Valuations.ARBITRAGE, Variables.Scenarios.MAXIMUM)):
    vτ = Variable("maximum", np.float32, locator="maximum")


class ValuationCalculation(Calculation, ABC, metaclass=RegistryMeta): pass
class ArbitrageCalculation(ValuationCalculation, ABC):
    def execute(self, strategies, *args, discount, **kwargs):
        with self.equation(strategies, discount=discount) as equation:
            yield equation["underlying"]
            yield equation["size"]
            yield equation["current"]
            yield equation.exp()
            yield equation.npv()
            yield equation.apy()

class MinimumArbitrageCalculation(ArbitrageCalculation, equation=MinimumArbitrageCalculation, register=(Variables.Valuations.ARBITRAGE, Variables.Scenarios.MINIMUM)): pass
class MaximumArbitrageCalculation(ArbitrageCalculation, equation=MaximumArbitrageCalculation, register=(Variables.Valuations.ARBITRAGE, Variables.Scenarios.MAXIMUM)): pass


class ValuationCalculator(Function, Logging, Sizing, Emptying):
    def __init__(self, *args, **kwargs):
        Function.__init__(self, *args, **kwargs)
        Logging.__init__(self, *args, **kwargs)
        Identity = ntuple("Identity", "valuation scenario")
        calculations = {Identity(*identity): calculation for identity, calculation in dict(ValuationCalculation).items()}
        calculations = {identity.scenario: calculation for identity, calculation in calculations.items() if identity.valuation == kwargs["valuation"]}
        self.__calculations = {scenario: calculation(*args, **kwargs) for scenario, calculation in calculations.items()}
        self.__header = ValuationHeader(*args, **kwargs)

    def execute(self, source, *args, **kwargs):
        assert isinstance(source, tuple)
        contract, strategies = source
        assert isinstance(contract, Querys.Contract) and isinstance(strategies, (list, xr.Dataset))
        assert all([isinstance(dataset, xr.Dataset) for dataset in strategies]) if isinstance(strategies, list) else True
        if self.empty(strategies): return
        strategies in list(self.strategies(strategies))
        for valuations in self.calculate(strategies, *args, **kwargs):
            size = self.size(valuations)
            string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
            self.logger.info(string)
            if self.empty(valuations): continue
            yield valuations

    def calculate(self, strategies, *args, **kwargs):
        isinstance(strategies, (list, xr.Dataset))
        assert all([isinstance(dataset, xr.Dataset) for dataset in strategies]) if isinstance(strategies, list) else True
        for dataframe in list(strategies):
            scenarios = dict(self.scenarios(dataframe, *args, **kwargs))
            valuations = dict(self.valuations(scenarios, *args, **kwargs))
            valuations = pd.concat(list(valuations.values()), axis=0)
            if self.empty(valuations): return
            valuations = self.pivot(valuations, *args, **kwargs)
            return valuations

    def scenarios(self, strategies, *args, **kwargs):
        assert isinstance(strategies, xr.Dataset)
        function = lambda mapping: {key: xr.Variable(key, [value]).squeeze(key) for key, value in mapping.items()}
        for scenario, calculation in self.calculations.items():
            valuations = calculation(strategies, *args, **kwargs)
            assert isinstance(valuations, xr.Dataset)
            coordinates = dict(valuation=self.valuation, scenario=scenario)
            coordinates = function(coordinates)
            valuations = valuations.assign_coords(coordinates).expand_dims("scenario")
            yield scenario, valuations

    def pivot(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        index = set(dataframe.columns) - ({"scenario"} | set(self.header.stacking))
        dataframe = dataframe.pivot(index=list(index), columns="scenario")
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        return dataframe

    @staticmethod
    def strategies(strategies):
        isinstance(strategies, (list, xr.Dataset))
        assert all([isinstance(dataset, xr.Dataset) for dataset in strategies]) if isinstance(strategies, list) else True
        strategies = [strategies] if isinstance(strategies, xr.Dataset) else strategies
        yield from iter(strategies)

    @staticmethod
    def valuations(scenarios, *args, **kwargs):
        assert isinstance(scenarios, dict)
        for scenario, dataset in scenarios.items():
            dataset = dataset.drop_vars(list(Variables.Security), errors="ignore")
            dataset = dataset.expand_dims(list(set(iter(dataset.coords)) - set(iter(dataset.dims))))
            dataframe = dataset.to_dataframe().dropna(how="all", inplace=False)
            dataframe = dataframe.reset_index(drop=False, inplace=False)
            yield scenario, dataframe

    @property
    def calculations(self): return self.__calculations
    @property
    def valuation(self): return self.__valuation
    @property
    def header(self): return self.__header


class ValuationReader(Reader):
    def read(self, *args, **kwargs):
        rejected = self.rejected(*args, **kwargs)
        accepted = self.accepted(*args, **kwargs)
        return accepted

    def rejected(self, *args, **kwargs):
        rejected = self.table.rejected(*args, **kwargs)
        size = self.size(rejected)
        string = f"Rejected: {repr(self)}[{size:.0f}]"
        self.logger.info(string)
        return rejected

    def accepted(self, *args, **kwargs):
        accepted = self.table.accepted(*args, **kwargs)
        size = self.size(accepted)
        string = f"Accepted: {repr(self)}[{size:.0f}]"
        self.logger.info(string)
        return accepted


class ValuationWriter(Writer):
    def __init__(self, *args, priority, status=Variables.Status.PROSPECT, **kwargs):
        assert callable(priority) and status == Variables.Status.PROSPECT
        Writer.__init__(self, *args, **kwargs)
        self.__header = ValuationHeader(*args, **kwargs)
        self.__priority = priority
        self.__status = status

    def write(self, contract, valuations, *args, **kwargs):
        assert isinstance(contract, Querys.Contract) and isinstance(valuations, pd.DataFrame)
        obsolete = self.table.obsolete(contract, *args, **kwargs)
        size = self.size(obsolete)
        string = f"Obsolete: {repr(self)}|{str(contract)}[{size:.0f}]"
        self.logger.info(string)
        valuations = self.valuations(valuations, *args, **kwargs)
        if bool(valuations.empty): return
        valuations = self.portfolio(valuations, *args, **kwargs)
        valuations = self.prospect(valuations, *args, **kwargs)
        self.table.combine(valuations)
        self.table.unique(self.header.index)
        self.table.sort("priority", reverse=True)

    def valuations(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.Dataframe)
        if not bool(self): return valuations
        overlap = self.table.dataframe.merge(valuations, on=self.header.index, how="inner", suffixes=("_", ""))[self.header.columns]
        valuations = pd.concat([valuations, overlap], axis=0)
        valuations = valuations.drop_duplicates(self.header.index, keep="last", inplace=False)
        return valuations

    def prioritize(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        valuations["priority"] = valuations.apply(self.priority, axis=1)
        parameters = dict(ascending=False, inplace=False, ignore_index=False)
        valuations = valuations.sort_values("priority", axis=0, **parameters)
        return valuations

    def prospect(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        if "status" not in valuations.columns.levels[0]: valuations["status"] = np.NaN
        function = lambda status: self.status if np.isnan(status) else status
        valuations["status"] = valuations["status"].apply(function)
        return valuations

    @property
    def priority(self): return self.__priority
    @property
    def status(self): return self.__status
    @property
    def header(self): return self.__header



