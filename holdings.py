# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Holdings Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
from itertools import product, count
from collections import namedtuple as ntuple

from finance.variables import Variables, Contract
from support.tables import Tables, Views
from support.pipelines import Producer, Consumer
from support.meta import ParametersMeta
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["HoldingFiles", "ValuationTable", "ValuationReader", "ValuationWriter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class ValuationAxes(object, metaclass=ParametersMeta):
    options = list(map(str, Variables.Securities.Options))
    stocks = list(map(str, Variables.Securities.Stocks))
    securities = list(map(str, Variables.Securities))
    scenarios = list(Variables.Scenarios)
    arbitrage = ["apy", "npv", "cost"]
    contract = ["ticker", "expire"]

    def __init__(self, *args, valuation, **kwargs):
        valuation = str(valuation).lower()
        index = self.contract + ["valuation", "strategy"] + self.options
        unstacked = ["current", "size", "tau", "underlying"] + ["status", "priority"]
        stacked = list(product(getattr(self, valuation), self.scenarios))
        unstacked = list(product(unstacked, [""]))
        index = list(product(index, [""]))
        self.valuation = str(valuation).lower()
        self.header = list(index) + list(stacked) + list(unstacked)
        self.columns = list(stacked) + list(unstacked)
        self.unstacked = list(unstacked)
        self.stacked = list(stacked)
        self.index = list(index)

class HoldingAxes(object, metaclass=ParametersMeta):
    security = ["instrument", "option", "position"]
    contract = ["ticker", "expire"]

    def __init__(self, *args, **kwargs):
        self.header = self.contract + self.security + ["strike", "quantity"]
        self.index = self.contract + self.security + ["strike"]
        self.columns = ["quantity"]


class HoldingParameters(object, metaclass=ParametersMeta):
    filename = lambda contract: "_".join([str(contract.ticker).upper(), str(contract.expire.strftime("%Y%m%d"))])
    parsers = {"instrument": Variables.Instruments, "option": Variables.Options, "position": Variables.Positions}
    formatters = {"instrument": int, "option": int, "position": int}
    types = {"ticker": str, "strike": np.float32, "quantity": np.int32}
    dates = {"expire": "%Y%m%d"}
    datatype = pd.DataFrame

class ValuationFormatting(object, metaclass=ParametersMeta):
    order = ["ticker", "expire", "valuation", "strategy"] + ValuationAxes.options + [(lead, lag) for lead, lag in product(ValuationAxes.arbitrage, ValuationAxes.scenarios)] + ["size", "status"]
    formats = {(lead, lag): lambda column: f"{column:.02f}" for lead, lag in product(ValuationAxes.arbitrage[1:], ValuationAxes.scenarios)}
    formats.update({(lead, lag): lambda column: f"{column * 100:.02f}%" if np.isfinite(column) else "InF" for lead, lag in product(ValuationAxes.arbitrage[0], ValuationAxes.scenarios)})
    formats.update({"priority": lambda priority: f"{priority * 100:.02f}%" if np.isfinite(priority) else "InF"})
    formats.update({"identity": lambda identity: f"{identity:.0f}", "status": lambda status: str(status), "size": lambda size: f"{size:.02f}"})
    numbers = lambda column: f"{column:.02f}"


class ValuationView(Views.Dataframe, rows=20, columns=30, width=250, **dict(ValuationFormatting)): pass
class ValuationTable(Tables.Dataframe, view=ValuationView, axes=ValuationAxes): pass


class HoldingFile(File, variable=Variables.Datasets.HOLDINGS, **dict(HoldingParameters)): pass
class HoldingFiles(object): Holding = HoldingFile


class ValuationWriter(Consumer, variable=Variables.Querys.CONTRACT):
    def __init__(self, *args, table, priority, **kwargs):
        super().__init__(*args, **kwargs)
        self.__axes = ValuationAxes(*args, **kwargs)
        self.__status = Variables.Status.PROSPECT
        self.__identity = count(1, step=1)
        self.__priority = priority
        self.__table = table

    def consumer(self, contents, *args, **kwargs):
        contract, valuations = contents[Variables.Querys.CONTRACT], contents[self.axes.valuation]
        assert isinstance(valuations, pd.DataFrame)
        if bool(valuations.empty): return
        with self.table.mutex:
            self.obsolete(contract, *args, **kwargs)
            valuations = self.valuations(valuations, *args, **kwargs)
            valuations = self.prioritize(valuations, *args, **kwargs)
            valuations = self.identify(valuations, *args, **kwargs)
            valuations = self.prospect(valuations, *args, **kwargs)
            self.write(valuations, *args, **kwargs)

    def obsolete(self, contract, *args, **kwargs):
        if not bool(self.table): return
        ticker = lambda table: table["ticker"] == contract.ticker
        expire = lambda table: table["expire"] == contract.expire
        status = lambda table: table["status"] == Variables.Status.PROSPECT
        obsolete = lambda table: ticker(table) & expire(table) & status(table)
        self.table.remove(obsolete)

    def valuations(self, valuations, *args, **kwargs):
        if not bool(self.table): return valuations
        existing = self.table.dataframe
        index = list(self.axes.index)
        overlap = existing.merge(valuations, on=index, how="inner", suffixes=("_", ""))[existing.columns]
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

    def write(self, valuations, *args, **kwargs):
        index = list(self.axes.index)
        self.table.combine(valuations)
        self.table.unique(index)
        self.table.sort("priority", reverse=True)

    @property
    def table(self): return self.__table
    @property
    def priority(self): return self.__priority
    @property
    def identity(self): return self.__identity
    @property
    def status(self): return self.__status
    @property
    def axes(self): return self.__axes


class ValuationReader(Producer, variable=Variables.Querys.CONTRACT):
    def __init__(self, *args, table, **kwargs):
        super().__init__(*args, **kwargs)
        Axes = ntuple("Axes", "valuation holding")
        valuation = ValuationAxes(*args, **kwargs)
        holding = HoldingAxes(*args, **kwargs)
        self.__axes = Axes(valuation, holding)
        self.__table = table

    def producer(self, *args, **kwargs):
        if not bool(self.table): return
        with self.table.mutex:
            self.obsolete(*args, **kwargs)
            valuations = self.read(*args, **kwargs)
        if bool(valuations.empty): return
        valuations = self.valuations(valuations, *args, **kwargs)
        valuations = self.stocks(valuations, *args, **kwargs)
        holdings = self.holdings(valuations, *args, **kwargs)
        for (ticker, expire), dataframe in self.groupings(holdings, *args, **kwargs):
            contract = Contract(ticker, expire)
            holdings = {Variables.Querys.CONTRACT: contract, Variables.Datasets.HOLDINGS: dataframe}
            yield dict(holdings)

    def read(self, *args, **kwargs):
        if not bool(self.table): return pd.DataFrame(columns=self.table.columns)
        accepted = lambda table: table["status"] == Variables.Status.ACCEPTED
        valuations = self.table.extract(accepted)
        return valuations

    def obsolete(self, *args, tenure=None, **kwargs):
        if not bool(self.table): return
        rejected = lambda table: table["status"] == Variables.Status.REJECTED
        abandoned = lambda table: table["status"] == Variables.Status.ABANDONED
        timeout = lambda table: (pd.to_datetime("now") - table["current"]) >= tenure if (tenure is not None) else False
        obsolete = lambda table: rejected(table) | abandoned(table) | timeout(table)
        self.table.remove(obsolete)

    def valuations(self, valuations, *args, **kwargs):
        stacked = [column[0] for column in self.axes.valuation.stacked]
        index = set(valuations.columns) - ({"scenario"} | set(stacked))
        valuations = valuations[list(index)].droplevel("scenario", axis=1)
        return valuations

    def stocks(self, valuations, *args, **kwargs):
        stocks = list(self.axes.valuation.stocks)
        strategy = lambda cols: list(map(str, cols["strategy"].stocks))
        function = lambda cols: {stock: cols["underlying"] if stock in strategy(cols) else np.NaN for stock in stocks}
        stocks = valuations.apply(function, axis=1, result_type="expand")
        valuations = pd.concat([valuations, stocks], axis=1)
        return valuations

    def holdings(self, valuations, *args, **kwargs):
        securities, header = list(self.axes.valuation.securities), list(self.axes.holding.header)
        valuations = valuations[[column for column in header if column in valuations.columns] + securities]
        contracts = [column for column in valuations.columns if column not in securities]
        holdings = valuations.melt(id_vars=contracts, value_vars=securities, var_name="security", value_name="strike")
        holdings = holdings.where(holdings["strike"].notna()).dropna(how="all", inplace=False)
        holdings["security"] = holdings["security"].apply(Variables.Securities)
        holdings["instrument"] = holdings["security"].apply(lambda security: security.instrument)
        holdings["option"] = holdings["security"].apply(lambda security: security.option)
        holdings["position"] = holdings["security"].apply(lambda security: security.position)
        holdings = holdings.assign(quantity=1)
        return holdings[header]

    def groupings(self, holdings, *args, **kwargs):
        header = list(self.axes.holding.header)
        holdings = holdings.groupby(header, as_index=False, dropna=False, sort=False).sum()
        for (ticker, expire), dataframe in iter(holdings.groupby(["ticker", "expire"])):
            yield (ticker, expire), dataframe

    @property
    def table(self): return self.__table
    @property
    def axes(self): return self.__axes




