# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Valuation Objects
@author: Jack Kirby Cook

"""

import types
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
    formats = dict(status=lambda status: str(status), size=lambda size: f"{size:.0f}")
    formats["apy"] = lambda column: (f"{column * 100:.02f}%" if column < 10 else "EsV") if np.isfinite(column) else "InF"
    numbers = lambda column: f"{column:.02f}"


class ValuationHeader(object):
    def __iter__(self): return iter(self.index + self.columns)
    def __init__(self, *args, valuation, columns=[], **kwargs):
        assert valuation in list(Variables.Valuations) and isinstance(columns, list)
        valuations = {Variables.Valuations.ARBITRAGE: ["apy", "npv", "cost"]}
        stacked = lambda cols: list(product(cols, list(Variables.Scenarios)))
        unstacked = lambda cols: list(product(cols, [""]))
        columns = stacked(valuations[valuation]) + unstacked(["underlying", "size", "current"] + list(columns))
        index = unstacked(["valuation", "strategy"] + list(Querys.Contract) + list(map(str, Variables.Securities.Options)))
        self.__stacking = valuations[valuation]
        self.__valuation = valuation
        self.__columns = columns
        self.__index = index

    @property
    def valuation(self): return self.__valuation
    @property
    def stacking(self): return self.__stacking
    @property
    def columns(self): return self.__columns
    @property
    def index(self): return self.__index


class ValuationView(View, ABC, datatype=pd.DataFrame, **dict(ValuationFormatting)): pass
class ValuationTable(Table, ABC, datatype=pd.DataFrame):
    def __init__(self, *args, **kwargs):
        header = ValuationHeader(*args, columns=["priority", "status"], **kwargs)
        table = pd.DataFrame(columns=list(header))
        view = ValuationView(*args, **kwargs)
        parameters = dict(table=table, view=view)
        Table.__init__(self, *args, **parameters, **kwargs)

    def obsolete(self, contract, *args, **kwargs):
        assert isinstance(contract, Querys.Contract)
        primary = lambda table: [table[:, key] == value for key, value in contract.items()]
        secondary = lambda table: table[:, "status"] == Variables.Status.PROSPECT
        function = lambda table: reduce(lambda x, y: x & y, primary(table) + [secondary(table)])
        return self.extract(function)

    def rejected(self, *args, tenure=None, **kwargs):
        timeout = lambda table: (pd.to_datetime("now") - table[:, "current"]) >= tenure if (tenure is not None) else False
        rejected = lambda table: table[:, "status"] == Variables.Status.REJECTED
        abandoned = lambda table: table[:, "status"] == Variables.Status.ABANDONED
        function = lambda table: rejected(table) | abandoned(table) | timeout(table)
        return self.extract(function)

    def accepted(self, *args, **kwargs):
        function = lambda table: table[:, "status"] == Variables.Status.ACCEPTED
        return self.extract(function)


class ValuationEquation(Equation, ABC):
    tau = Variable("tau", "tau", np.int32, xr.DataArray, vectorize=True, function=lambda to, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))
    inc = Variable("inc", "income", np.float32, xr.DataArray, vectorize=True, function=lambda vo, vτ: + np.maximum(vo, 0) + np.maximum(vτ, 0))
    exp = Variable("exp", "cost", np.float32, xr.DataArray, vectorize=True, function=lambda vo, vτ: - np.minimum(vo, 0) - np.minimum(vτ, 0))

    xo = Variable("xo", "underlying", np.float32, xr.DataArray, locator="underlying")
    tτ = Variable("tτ", "expire", np.datetime64, xr.DataArray, locator="expire")
    to = Variable("to", "current", np.datetime64, xr.DataArray, locator="current")
    vo = Variable("vo", "spot", np.float32, xr.DataArray, locator="spot")
    ρ = Variable("ρ", "discount", np.float32, types.NoneType, locator="discount")

class ArbitrageEquation(ValuationEquation, ABC):
    irr = Variable("irr", "irr", np.float32, xr.DataArray, vectorize=True, function=lambda inc, exp, tau: np.power(np.divide(inc, exp), np.power(tau, -1)) - 1)
    npv = Variable("npv", "npv", np.float32, xr.DataArray, vectorize=True, function=lambda inc, exp, tau, ρ: np.divide(inc, np.power(1 + ρ, tau / 365)) - exp)
    apy = Variable("apy", "apy", np.float32, xr.DataArray, vectorize=True, function=lambda irr, tau: np.power(irr + 1, np.power(tau / 365, -1)) - 1)

class MinimumArbitrageEquation(ArbitrageEquation):
    vτ = Variable("vτ", "minimum", np.float32, xr.DataArray, locator="minimum")

class MaximumArbitrageEquation(ArbitrageEquation):
    vτ = Variable("vτ", "maximum", np.float32, xr.DataArray, locator="maximum")


class ValuationCalculation(Calculation, ABC, metaclass=RegistryMeta): pass
class ArbitrageCalculation(ValuationCalculation, ABC):
    def execute(self, strategies, *args, discount, **kwargs):
        with self.equation(strategies, discount=discount) as equation:
            yield strategies["underlying"]
            yield strategies["size"]
            yield strategies["current"]
            yield equation.exp()
            yield equation.npv()
            yield equation.apy()

class MinimumArbitrageCalculation(ArbitrageCalculation, equation=MinimumArbitrageEquation, register=(Variables.Valuations.ARBITRAGE, Variables.Scenarios.MINIMUM)): pass
class MaximumArbitrageCalculation(ArbitrageCalculation, equation=MaximumArbitrageEquation, register=(Variables.Valuations.ARBITRAGE, Variables.Scenarios.MAXIMUM)): pass


class ValuationCalculator(Function, Logging, Sizing, Emptying):
    def __init__(self, *args, **kwargs):
        Function.__init__(self, *args, **kwargs)
        Logging.__init__(self, *args, **kwargs)
        Identity = ntuple("Identity", "valuation scenario")
        calculations = {Identity(*identity): calculation for identity, calculation in dict(ValuationCalculation).items()}
        calculations = {identity.scenario: calculation for identity, calculation in calculations.items() if identity.valuation == kwargs["valuation"]}
        self.__calculations = {scenario: calculation(*args, **kwargs) for scenario, calculation in calculations.items()}
        self.__header = ValuationHeader(*args, columns=[], **kwargs)

    def execute(self, source, *args, **kwargs):
        assert isinstance(source, tuple)
        contract, strategies = source
        assert isinstance(contract, Querys.Contract) and isinstance(strategies, (list, xr.Dataset))
        assert all([isinstance(dataset, xr.Dataset) for dataset in strategies]) if isinstance(strategies, list) else True
        if self.empty(strategies["size"]): return
        strategies = list(self.strategies(strategies))
        for valuations in self.calculate(strategies, *args, **kwargs):
            size = self.size(valuations)
            valuations = valuations.reindex(columns=list(self.header), fill_value=np.NaN)
            string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
            self.logger.info(string)
            if self.empty(valuations): continue
            yield valuations

    def calculate(self, strategies, *args, **kwargs):
        assert isinstance(strategies, (list, xr.Dataset))
        assert all([isinstance(dataset, xr.Dataset) for dataset in strategies]) if isinstance(strategies, list) else True
        for dataset in list(strategies):
            scenarios = dict(self.scenarios(dataset, *args, **kwargs))
            valuations = dict(self.valuations(scenarios, *args, **kwargs))
            valuations = pd.concat(list(valuations.values()), axis=0)
            if self.empty(valuations): return
            valuations = self.pivot(valuations, *args, **kwargs)
            yield valuations

    def scenarios(self, strategies, *args, **kwargs):
        assert isinstance(strategies, xr.Dataset)
        function = lambda mapping: {key: xr.Variable(key, [value]).squeeze(key) for key, value in mapping.items()}
        for scenario, calculation in self.calculations.items():
            valuations = calculation(strategies, *args, **kwargs)
            assert isinstance(valuations, xr.Dataset)
            coordinates = dict(valuation=self.header.valuation, scenario=scenario)
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
        assert isinstance(strategies, (list, xr.Dataset))
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
        self.__header = ValuationHeader(*args, columns=["status", "priority"], **kwargs)
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
        valuations = self.prioritize(valuations, *args, **kwargs)
        valuations = self.prospect(valuations, *args, **kwargs)
        self.table.combine(valuations)
        self.table.unique(self.header.index)
        self.table.sort("priority", reverse=True)
        self.table.reset()

    def valuations(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        if not bool(self.table): return valuations
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



