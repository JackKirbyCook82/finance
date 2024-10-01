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
from itertools import product, count

from finance.variables import Variables, Contract
from support.calculations import Variable, Equation, Calculation
from support.meta import RegistryMeta, ParametersMeta
from support.mixins import Empty, Sizing, Logging
from support.tables import Table, View
from support.filtering import Filter

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationFilter", "ValuationCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class ValuationFormatting(metaclass=ParametersMeta):
    order = ["ticker", "expire", "valuation", "strategy"] + list(map(str, Variables.Securities.Options)) + ["apy", "npv", "cost"] + ["size", "status"]
    formats = {"apy": lambda column: f"{column * 100:.02f}%" if np.isfinite(column) else "InF", "identity": lambda identity: f"{identity:.0f}", "status": lambda status: str(status), "size": lambda size: f"{size:.02f}"}
    numbers = lambda column: f"{column:.02f}"


class ValuationVariables(object):
    axes = {Variables.Querys.CONTRACT: ["ticker", "expire"], Variables.Datasets.SECURITY: ["instrument", "option", "position"]}
    axes.update({Variables.Instruments.STOCK: list(Variables.Securities.Stocks), Variables.Instruments.OPTION: list(Variables.Securities.Options)})
    data = {Variables.Valuations.ARBITRAGE: ["apy", "npv", "cost"]}

    def __init__(self, *args, valuation, **kwargs):
        stacked = list(product(self.data[valuation], list(Variables.Scenarios)))
        unstacked = list(product(["current", "size", "tau", "underlying"], [""]))
        self.contract = self.axes[Variables.Querys.CONTRACT]
        self.security = self.axes[Variables.Datasets.SECURITY]
        self.options = list(map(str, self.axes[Variables.Instruments.OPTIONS]))
        self.index = list(product(["identity"] + self.contract + self.options + ["valuation", "strategy"], [""]))
        self.header = self.index + stacked + unstacked
        self.stacking = self.data[valuation]


class ValuationView(View, ABC, datatype=pd.DataFrame, **dict(ValuationFormatting)): pass
class ValuationTable(Table, ABC, datatype=pd.DataFrame, view=ValuationView): pass
class ArbitrageTable(ValuationTable, ABC, variable=Variable.Valuations.ARBITRAGE): pass


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


class ValuationCalculation(Calculation, ABC, metaclass=RegistryMeta): pass
class ArbitrageCalculation(ValuationCalculation, ABC):
    def execute(self, strategies, *args, discount, **kwargs):
        equation = self.equation(*args, **kwargs)
        yield equation.npv(strategies, discount=discount)
        yield equation.apy(strategies, discount=discount)
        yield equation.exp(strategies, discount=discount)
        yield equation.tau(strategies)
        yield strategies["underlying"]
        yield strategies["current"]
        yield strategies["size"]

class MinimumArbitrageCalculation(ArbitrageCalculation, equation=MinimumArbitrageEquation, register=(Variables.Valuations.ARBITRAGE, Variables.Scenarios.MINIMUM)): pass
class MaximumArbitrageCalculation(ArbitrageCalculation, equation=MaximumArbitrageEquation, register=(Variables.Valuations.ARBTIRAGE, Variables.Scenarios.MAXIMUM)): pass


class ValuationFilter(Sizing, Empty, Logging, Filter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__variables = ValuationVariables(*args, **kwargs)

    def execute(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        for contract, dataframe in self.contracts(valuations, *args, **kwargs):
            if bool(dataframe.empty): continue
            prior = len(dataframe.dropna(how="all", inplace=False).index)
            dataframe = self.filter(dataframe, *args, **kwargs)
            assert isinstance(dataframe, pd.DataFrame)
            dataframe = dataframe.reset_index(drop=True, inplace=False)
            post = len(dataframe.dropna(how="all", inplace=False).index)
            string = f"Filtered: {repr(self)}|{str(contract)}[{prior:.0f}|{post:.0f}]"
            self.logger.info(string)
            yield dataframe

    def contracts(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        for (ticker, expire), dataframe in valuations.groupby(self.variables.contract):
            contract = Contract(ticker, expire)
            yield contract, dataframe

    @property
    def variables(self): return self.__variables


class ValuationCalculator(Sizing, Empty, Logging):
    def __init__(self, *args, **kwargs):
        calculations = dict(ValuationCalculation).items()
        calculations = {scenario: calculation for (valuation, scenario), calculation in calculations if valuation == kwargs["valuation"]}
        super().__init__(*args, **kwargs)
        self.__calculations = {scenario: calculation(*args, **kwargs) for scenario, calculation in calculations.items()}
        self.__variables = ValuationVariables(*args, **kwargs)
        self.__valuation = kwargs["valuation"]
        self.__identifier = count(start=1, step=1)

    def __call__(self, strategies, *args, **kwargs):
        assert isinstance(strategies, xr.Dataset)
        for contract, dataset in self.contracts(strategies):
            valuations = self.execute(dataset, *args, **kwargs)
            size = self.size(valuations)
            string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
            self.logger.info(string)
            if bool(valuations.empty): continue
            yield valuations

    def contracts(self, strategies):
        assert isinstance(strategies, xr.Dataset)
        for (ticker, expire), dataset in strategies.groupby(self.variables.contract):
            if not bool(np.count_nonzero(~np.isnan(dataset["size"].values))): continue
            contract = Contract(ticker, expire)
            yield contract, dataset

    def execute(self, strategies, *args, **kwargs):
        assert isinstance(strategies, xr.Dataset)
        valuations = self.calculate(strategies, *args, **kwargs)
        valuations["identity"] = [next(self.identifier) for _ in range(len(valuations))]
        if not bool(valuations): return pd.DataFrame(columns=self.variables.header)
        return self.pivot(valuations, *args, **kwargs)

    def calculate(self, strategies, *args, **kwargs):
        assert isinstance(strategies, xr.Dataset)
        scenarios = dict(self.scenarios(strategies, *args, **kwargs))
        valuations = dict(self.valuations(scenarios, *args, **kwargs))
        return valuations

    def scenarios(self, strategies, *args, **kwargs):
        assert isinstance(strategies, xr.Dataset)
        function = lambda mapping: {key: xr.Variable(key, [value]).squeeze(key) for key, value in mapping.items()}
        for scenario, calculation in self.calculations.items():
            valuations = calculation(strategies, *args, **kwargs)
            assert isinstance(valuations, xr.Dataset)
            coordinates = function(valuation=self.valuation, scenario=scenario)
            valuations = valuations.assign_coords(coordinates).expand_dims("scenario")
            yield scenario, valuations

    def valuations(self, scenarios, *args, **kwargs):
        assert isinstance(scenarios, dict)
        for scenario, dataset in scenarios.items():
            dataset = dataset.drop_vars(self.variables.security, errors="ignore")
            dataset = dataset.expand_dims(list(set(iter(dataset.coords)) - set(iter(dataset.dims))))
            dataframe = dataset.to_dataframe().dropna(how="all", inplace=False)
            dataframe = dataframe.reset_index(drop=False, inplace=False)
            yield scenario, dataframe

    def pivot(self, dataframe, *args, **kwargs):
        assert isinstance(dataframe, pd.DataFrame)
        index = set(dataframe.columns) - ({"scenario"} | set(self.variables.stacking))
        dataframe = dataframe.pivot(index=list(index), columns="scenario")
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        return dataframe

    @property
    def calculations(self): return self.__calculations
    @property
    def variables(self): return self.__variables
    @property
    def valuation(self): return self.__valuation
    @property
    def identifier(self): return self.__identifier


class ValuationWriter(Sizing, Empty, Logging):
    def __init__(self, *args, table, priority, **kwargs):
        super().__init__(*args, **kwargs)
        self.__variables = ValuationVariables(*args, **kwargs)
        self.__status = Variables.Status.PROSPECT
        self.__identity = count(1, step=1)
        self.__priority = priority
        self.__table = table

    def __call__(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        if bool(valuations.empty): return
        for contract, dataframe in self.contracts(valuations):
            with self.table.mutex:
                self.obsolete(contract, *args, **kwargs)
                dataframe = self.valuations(dataframe, *args, **kwargs)
                dataframe = self.prioritize(dataframe, *args, **kwargs)
                dataframe = self.identify(dataframe, *args, **kwargs)
                dataframe = self.prospect(dataframe, *args, **kwargs)
                self.execute(dataframe, *args, **kwargs)

    def contracts(self, exposures):
        assert isinstance(exposures, pd.DataFrame)
        for (ticker, expire), dataframe in exposures.groupby(self.variables.contract):
            if bool(dataframe.empty): continue
            contract = Contract(ticker, expire)
            yield contract, dataframe

    def obsolete(self, contract, *args, **kwargs):
        ticker = lambda table: table["ticker"] == contract.ticker
        expire = lambda table: table["expire"] == contract.expire
        status = lambda table: table["status"] == Variables.Status.PROSPECT
        obsolete = lambda table: ticker(table) & expire(table) & status(table)
        self.table.remove(obsolete)

    def valuations(self, valuations, *args, **kwargs):
        if not bool(self): return valuations
        index, columns = list(self.variables.index), list(self.variables.columns)
        overlap = self.dataframe.merge(valuations, on=index, how="inner", suffixes=("_", ""))[columns]
        valuations = pd.concat([valuations, overlap], axis=0)
        valuations = valuations.drop_duplicates(index, keep="last", inplace=False)
        return valuations

    def prioritize(self, valuations, *args, **kwargs):
        valuations["priority"] = valuations.apply(self.priority, axis=1)
        parameters = dict(ascending=False, inplace=False, ignore_index=False)
        valuations = valuations.sort_values("priority", axis=0, **parameters)
        return valuations

    def identify(self, valuations, *args, **kwargs):
        if "identity" not in valuations.columns.levels[0]: valuations["identity"] = np.NaN
        function = lambda tag: next(self.identity) if np.isnan(tag) else tag
        valuations["identity"] = valuations["identity"].apply(function)
        valuations = valuations.set_index("identity", drop=False, inplace=False)
        return valuations

    def prospect(self, valuations, *args, **kwargs):
        if "status" not in valuations.columns.levels[0]: valuations["status"] = np.NaN
        function = lambda status: self.status if np.isnan(status) else status
        valuations["status"] = valuations["status"].apply(function)
        return valuations

    def execute(self, valuations, *args, **kwargs):
        index = list(self.variables.index)
        self.table.combine(valuations)
        self.table.unique(index)
        self.table.sort("priority", reverse=True)

    @property
    def priority(self): return self.__priority
    @property
    def identity(self): return self.__identity
    @property
    def status(self): return self.__status
    @property
    def variables(self): return self.__variables
    @property
    def table(self): return self.__table


class ValuationReader(Sizing, Empty, Logging):
    def __init__(self, *args, table, **kwargs):
        super().__init__(*args, **kwargs)
        self.__variables = ValuationVariables(*args, **kwargs)
        self.__table = table

    def __call__(self, *args, **kwargs):
        if not bool(self.table): return
        with self.table.mutex:
            self.obsolete(*args, **kwargs)
            valuations = self.execute(*args, **kwargs)
            if bool(valuations.empty): return
            for contract, dataframe in self.contracts(valuations):
                string = f"Accepted: {repr(self)}|{str(contract)}|{len(dataframe):.0f}"
                self.logger.info(string)
                if bool(dataframe.empty): continue
                yield dataframe

    def obsolete(self, *args, tenure=None, **kwargs):
        rejected = lambda table: table["status"] == Variables.Status.REJECTED
        abandoned = lambda table: table["status"] == Variables.Status.ABANDONED
        timeout = lambda table: (pd.to_datetime("now") - table["current"]) >= tenure if (tenure is not None) else False
        obsolete = lambda table: rejected(table) | abandoned(table) | timeout(table)
        self.table.remove(obsolete)
        dataframe = obsolete.dropna(how="all", inplace=False)
        string = f"Rejected: {repr(self)}|{len(dataframe):.0f}"
        self.logger.info(string)

    def execute(self, *args, **kwargs):
        if not bool(self.table): return pd.DataFrame(columns=self.variables.header)
        accepted = lambda table: table["status"] == Variables.Status.ACCEPTED
        valuations = self.table.extract(accepted)
        return valuations

    def contracts(self, valuations):
        assert isinstance(valuations, pd.DataFrame)
        for (ticker, expire), dataframe in valuations.groupby(self.variables.contract):
            if bool(dataframe.empty): continue
            contract = Contract(ticker, expire)
            yield contract, dataframe

    @property
    def variables(self): return self.__variables
    @property
    def table(self): return self.__table





