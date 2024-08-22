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
from support.meta import ParametersMeta
from support.pipelines import Producer, Consumer
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["HoldingFiles", "HoldingTable", "HoldingReader", "HoldingWriter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Parameters(metaclass=ParametersMeta):
    parsers = {"instrument": Variables.Instruments, "option": Variables.Options, "position": Variables.Positions}
    formatters = {"instrument": int, "option": int, "position": int}
    types = {"ticker": str, "strike": np.float32, "quantity": np.int32}
    dates = {"expire": "%Y%m%d"}
    filename = lambda variable: "_".join([str(variable.ticker).upper(), str(variable.expire.strftime("%Y%m%d"))])
    datatype = pd.DataFrame

class Axes:
    stacking = {Variables.Valuations.ARBITRAGE: {"apy", "npv", "cost"}}
    options = list(map(str, Variables.Securities.Options))
    stocks = list(map(str, Variables.Securities.Stocks))
    securities = list(map(str, Variables.Securities))
    contract = ["ticker", "expire"]

class Headers:
    holdings = ["ticker", "expire", "strike", "instrument", "option", "position", "quantity"]

class Formatting(metaclass=ParametersMeta):
    order = ["ticker", "expire", "valuation", "strategy"] + Axes.options
    order = order + [(lead, lag) for lead, lag in product(["apy", "npv", "cost"], Variables.Scenarios)] + ["size", "status"]
    numbers = lambda column: f"{column:.02f}"
    formats = {(lead, lag): lambda column: f"{column:.02f}" for lead, lag in product(["npv", "cost"], Variables.Scenarios)}
    formats.update({(lead, lag): lambda column: f"{column * 100:.02f}%" if np.isfinite(column) else "InF" for lead, lag in product(["apy"], Variables.Scenarios)})
    formats.update({"priority": lambda priority: f"{priority * 100:.02f}%" if np.isfinite(priority) else "InF"})
    formats.update({"status": lambda status: str(status), "size": lambda size: f"{size:.02f}"})


class HoldingView(Views.Dataframe, rows=20, columns=30, width=250, **dict(Formatting)): pass
class HoldingTable(Tables.Dataframe, datatype=pd.DataFrame, tableview=HoldingView): pass
class HoldingFile(File, variable=Variables.Datasets.HOLDINGS, header=Headers.holdings, **dict(Parameters)): pass
class HoldingFiles(object): Holding = HoldingFile


class HoldingWriter(Consumer, title="Consumed", variable=Variables.Querys.CONTRACT):
    def __init__(self, *args, destination, priority, valuation, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__identity = count(1, step=1)
        self.__destination = destination
        self.__valuation = valuation
        self.__priority = priority

    def consumer(self, contents, *args, **kwargs):
        contract, valuations = contents[Variables.Querys.CONTRACT], contents[self.valuation]
        assert isinstance(valuations, pd.DataFrame)
        if bool(valuations.empty): return
        self.destination.mutex.acquire()
        blocking = self.blocking(contract, *args, **kwargs)
        if bool(blocking): return
        valuations = self.market(valuations, *args, **kwargs)
        valuations = self.prioritize(valuations, *args, **kwargs)
        valuations = self.identify(valuations, *args, **kwargs)
        if bool(valuations.empty): return
        self.write(valuations, *args, contract=contract, **kwargs)
        self.destination.mutex.release()

    def blocking(self, contract, *args, **kwargs):
        if not bool(self.destination): return False
        Columns = ntuple("Columns", "ticker expire status")
        dataframe = self.destination.table
        columns = [self.column(column, self.destination.table) for column in Columns._fields]
        columns = Columns(*columns)
        ticker = dataframe[columns.ticker] == contract.ticker
        expire = dataframe[columns.expire] == contract.expire
        series = dataframe.where(ticker & expire)[columns.status]
        return any(series == Variables.Status.PENDING)

    def market(self, valuations, *args, tenure=None, **kwargs):
        if tenure is None: return valuations
        column = self.column("current", valuations)
        current = (pd.to_datetime("now") - valuations[column]) <= self.tenure
        valuations = valuations.where(current).dropna(how="all", inplace=False)
        return valuations

    def prioritize(self, valuations, *args, **kwargs):
        column = self.column("priority", valuations)
        valuations[column] = valuations.apply(self.priority, axis=1)
        valuations = valuations.sort_values(column, axis=0, ascending=False, inplace=False, ignore_index=False)
        return valuations

    def identify(self, valuations, *args, **kwargs):
        tags = lambda size: [next(self.identity) for _ in range(size)]
        valuations = valuations.assign(status=Variables.Status.PROSPECT, tag=tags(len(valuations)))
        return valuations

    def write(self, valuations, *args, contract, tenure=None, **kwargs):
        Columns = ntuple("Columns", "ticker expire current")
        columns = [self.column(column, self.destination.table) for column in Columns._fields]
        columns = Columns(*columns)
        obsolete = lambda dataframe: (pd.to_datetime("now") - dataframe[columns.current]) >= tenure
        function = lambda dataframe: (dataframe[columns.ticker] == contract.ticker) & (dataframe[columns.expire] == contract.expire)
        valuations = valuations.set_index("tag", drop=False, inplace=False)
        self.destination.remove(function)
        self.destination.concat(valuations)
        self.destination.unique(Axes.contract + ["strategy"] + Axes.options)
        if tenure is not None: self.destination.remove(obsolete)
        self.destination.sort("priority", reverse=True)

    @staticmethod
    def column(column, dataframe):
        if isinstance(dataframe.columns, pd.MultiIndex):
            column = tuple([column]) if not isinstance(column, tuple) else column
            length = dataframe.columns.nlevels - len(column)
            column = column + tuple([""]) * length
        return column

    @property
    def destination(self): return self.__destination
    @property
    def valuation(self): return self.__valuation
    @property
    def priority(self): return self.__priority
    @property
    def identity(self): return self.__identity


class HoldingReader(Producer, title="Produced", variable=Variables.Querys.CONTRACT):
    def __init__(self, *args, source, valuation, **kwargs):
        super().__init__(*args, **kwargs)
        self.__source = source
        self.__valuation = valuation

    def producer(self, *args, **kwargs):
        valuations = self.read(*args, **kwargs)
        if bool(valuations.empty): return
        valuations = self.parse(valuations, *args, **kwargs)
        valuations = self.stocks(valuations, *args, **kwargs)
        holdings = self.holdings(valuations, *args, **kwargs)
        for (ticker, expire), dataframe in self.groupings(holdings, *args, **kwargs):
            contract = Contract(ticker, expire)
            holdings = {Variables.Querys.CONTRACT: contract, Variables.Datasets.HOLDINGS: dataframe}
            yield dict(holdings)

    def read(self, *args, **kwargs):
        if not bool(self.source): return pd.DataFrame()
        self.source.mutex.acquire()
        column = self.column("status", self.source.table)
        accepted = self.source.table[column] == Variables.Status.ACCEPTED
        accepted = self.source.table.where(accepted).dropna(how="all", inplace=False)
        self.source.remove(lambda dataframe: dataframe[column] == Variables.Status.ACCEPTED)
        self.source.remove(lambda dataframe: dataframe[column] == Variables.Status.REJECTED)
        self.source.remove(lambda dataframe: dataframe[column] == Variables.Status.ABANDONED)
        self.source.mutex.release()
        return accepted

    def parse(self, valuations, *args, **kwargs):
        columns = set(valuations.columns) - ({"scenario"} | Axes.stacking[self.valuation])
        valuations = valuations[list(columns)].droplevel("scenario", axis=1)
        return valuations

    @staticmethod
    def stocks(valuations, *args, **kwargs):
        strategy = lambda cols: list(map(str, cols["strategy"].stocks))
        function = lambda cols: {stock: cols["underlying"] if stock in strategy(cols) else np.NaN for stock in Axes.stocks}
        stocks = valuations.apply(function, axis=1, result_type="expand")
        valuations = pd.concat([valuations, stocks], axis=1)
        return valuations

    @staticmethod
    def holdings(valuations, *args, **kwargs):
        valuations = valuations[[column for column in Headers.holdings if column in valuations.columns] + Axes.securities]
        contracts = [column for column in valuations.columns if column not in Axes.securities]
        holdings = valuations.melt(id_vars=contracts, value_vars=Axes.securities, var_name="security", value_name="strike")
        holdings = holdings.where(holdings["strike"].notna()).dropna(how="all", inplace=False)
        holdings["security"] = holdings["security"].apply(Variables.Securities)
        holdings["instrument"] = holdings["security"].apply(lambda security: security.instrument)
        holdings["option"] = holdings["security"].apply(lambda security: security.option)
        holdings["position"] = holdings["security"].apply(lambda security: security.position)
        holdings = holdings.assign(quantity=1)
        return holdings[Headers.holdings]

    @staticmethod
    def groupings(holdings, *args, **kwargs):
        holdings = holdings.groupby(Headers.holdings, as_index=False, dropna=False, sort=False).sum()
        for (ticker, expire), dataframe in iter(holdings.groupby(["ticker", "expire"])):
            yield (ticker, expire), dataframe

    @staticmethod
    def column(column, dataframe):
        if isinstance(dataframe.columns, pd.MultiIndex):
            column = tuple([column]) if not isinstance(column, tuple) else column
            length = dataframe.columns.nlevels - len(column)
            column = column + tuple([""]) * length
        return column

    @property
    def source(self): return self.__source
    @property
    def valuation(self): return self.__valuation



