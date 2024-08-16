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

from finance.variables import Variables, Contract
from finance.operations import Operations
from support.tables import Tables, Views
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["HoldingFiles", "HoldingTable", "HoldingReader", "HoldingWriter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


holdings_dates = {"expire": "%Y%m%d"}
holdings_parsers = {"instrument": Variables.Instruments, "option": Variables.Options, "position": Variables.Positions}
holdings_formatters = {"instrument": int, "option": int, "position": int}
holdings_types = {"ticker": str, "strike": np.float32, "quantity": np.int32}
holdings_filename = lambda query: "_".join([str(query.ticker).upper(), str(query.expire.strftime("%Y%m%d"))])
holdings_parameters = dict(datatype=pd.DataFrame, filename=holdings_filename, dates=holdings_dates, parsers=holdings_parsers, formatters=holdings_formatters, types=holdings_types)
holdings_header = ["ticker", "expire", "strike", "instrument", "option", "position", "quantity"]

valuation_index = ["ticker", "expire", "strategy"] + list(map(str, Variables.Securities.Options))
valuation_stacking = {Variables.Valuations.ARBITRAGE: {"apy", "npv", "cost"}}
valuation_order = ["ticker", "expire", "valuation", "strategy"] + list(map(str, Variables.Securities.Options))
valuation_order = valuation_order + [(lead, lag) for lead, lag in product(["apy", "npv", "cost"], Variables.Scenarios)] + ["size", "status"]
valuation_formats = {(lead, lag): lambda column: f"{column:.02f}" for lead, lag in product(["npv", "cost"], Variables.Scenarios)}
valuation_formats.update({(lead, lag): lambda column: f"{column * 100:.02f}%" if np.isfinite(column) else "InF" for lead, lag in product(["apy"], Variables.Scenarios)})
valuation_formats.update({"priority": lambda priority: f"{priority * 100:.02f}%" if np.isfinite(priority) else "InF"})
valuation_formats.update({"status": lambda status: str(status), "size": lambda size: f"{size:.02f}"})
valuation_numbers = lambda column: f"{column:.02f}"


class HoldingView(Views.Dataframe, rows=20, columns=30, width=250, formats=valuation_formats, numbers=valuation_numbers, order=valuation_order): pass
class HoldingTable(Tables.Dataframe, datatype=pd.DataFrame, tableview=HoldingView): pass
class HoldingFile(File, variable=Variables.Datasets.HOLDINGS, header=holdings_header, **holdings_parameters): pass
class HoldingFiles(object): Holding = HoldingFile


class HoldingWriter(Operations.Consumer):
    def __init__(self, *args, destination, priority, valuation, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__identity = count(1, step=1)
        self.__destination = destination
        self.__valuation = valuation
        self.__priority = priority

    def consumer(self, contents, *args, **kwargs):
        contract, valuations = contents[Variables.Querys.CONTRACT], contents[self.valuation]
        assert isinstance(valuations, pd.DataFrame)
        if bool(valuations.empty):
            return
        valuations = self.parse(valuations, *args, **kwargs)
        valuations = self.compare(valuations, *args, **kwargs)
        valuations = self.market(valuations, *args, **kwargs)
        valuations = self.prioritize(valuations, *args, **kwargs)
        valuations = self.identify(valuations, *args, **kwargs)
        if bool(valuations.empty):
            return
        self.write(valuations, *args, **kwargs)

    def parse(self, valuations, *args, **kwargs):
        index = set(valuations.columns) - ({"scenario"} | valuation_stacking[self.valuation])
        valuations = valuations.pivot(index=list(index), columns="scenario")
        valuations = valuations.reset_index(drop=False, inplace=False)
        return valuations

    def compare(self, valuations, *args, **kwargs):
        if not bool(self.destination):
            return valuations
        overlap = pd.merge(valuations, self.destination.table, how="inner", on=valuation_index, suffixes=("", "|expired"))
        overlap = overlap[self.destination.columns]
        index = [(value, "") for value in valuation_index]
        valuations = pd.concat([valuations, overlap], axis=0) .drop_duplicates(subset=index, inplace=False, keep="last")
        return valuations

    def market(self, valuations, *args, tenure=None, **kwargs):
        if tenure is not None:
            current = (pd.to_datetime("now") - valuations["current"]) <= self.tenure
            valuations = valuations.where(current).dropna(how="all", inplace=False)
        return valuations

    def prioritize(self, valuations, *args, **kwargs):
        valuations["priority"] = valuations.apply(self.priority, axis=1)
        valuations = valuations.sort_values("priority", axis=0, ascending=False, inplace=False, ignore_index=False)
        valuations = valuations.where(valuations["priority"] > 0).dropna(axis=0, how="all")
        return valuations

    def identify(self, valuations, *args, **kwargs):
        if not bool(self.destination):
            valuations["status"] = np.NaN
            valuations["tag"] = np.NaN
        function = lambda tag: next(self.identity) if np.isnan(tag) else tag
        valuations["status"] = valuations["status"].fillna(Variables.Status.PROSPECT)
        valuations["tag"] = valuations["tag"].apply(function)
        return valuations

    def write(self, valuations, *args, tenure=None, **kwargs):
        function = lambda dataframe: (pd.to_datetime("now") - dataframe["current"]) <= tenure
        with self.destination.mutex:
            valuations = valuations.set_index("tag", drop=False, inplace=False)
            self.destination.concat(valuations)
            self.destination.unique(valuation_index)
            if tenure is not None:
                self.destination.where(function)
            self.destination.sort("priority", reverse=True)

    @property
    def destination(self): return self.__destination
    @property
    def valuation(self): return self.__valuation
    @property
    def priority(self): return self.__priority
    @property
    def identity(self): return self.__identity


class HoldingReader(Operations.Producer):
    def __init__(self, *args, source, valuation, **kwargs):
        super().__init__(*args, **kwargs)
        self.__source = source
        self.__valuation = valuation

    def producer(self, *args, **kwargs):
        valuations = self.read(*args, **kwargs)
        assert isinstance(valuations, pd.DataFrame)
        if bool(valuations.empty):
            return
        valuations = self.parse(valuations, *args, **kwargs)
        valuations = self.stocks(valuations, *args, **kwargs)
        holdings = self.holdings(valuations, *args, **kwargs)
        for (ticker, expire), dataframe in self.groupings(holdings, *args, **kwargs):
            contract = Contract(ticker, expire)
            holdings = {Variables.Querys.CONTRACT: contract, Variables.Datasets.HOLDINGS: dataframe}
            yield dict(holdings)

    def read(self, *args, **kwargs):
        if not bool(self.source):
            return pd.DataFrame()
        function = lambda status: lambda dataframe: dataframe["status"] == status
        self.source.remove(function(Variables.Status.ABANDONED))
        self.source.remove(function(Variables.Status.REJECTED))
        accepted = self.source.remove(function(Variables.Status.ACCEPTED))
        return accepted

    def parse(self, valuations, *args, **kwargs):
        columns = set(valuations.columns) - ({"scenario"} | valuation_stacking[self.valuation])
        valuations = valuations[list(columns)].droplevel("scenario", axis=1)
        return valuations

    @staticmethod
    def stocks(valuations, *args, **kwargs):
        stocks = list(map(str, Variables.Securities.Stocks))
        strategy = lambda cols: list(map(str, cols["strategy"].stocks))
        function = lambda cols: {stock: cols["underlying"] if stock in strategy(cols) else np.NaN for stock in stocks}
        stocks = valuations.apply(function, axis=1, result_type="expand")
        valuations = pd.concat([valuations, stocks], axis=1)
        return valuations

    @staticmethod
    def holdings(valuations, *args, **kwargs):
        columns = [column for column in holdings_header if column in valuations.columns] + list(map(str, Variables.Securities))
        valuations = valuations[columns]
        securities = list(map(str, Variables.Securities))
        contracts = [column for column in valuations.columns if column not in securities]
        dataframe = valuations.melt(id_vars=contracts, value_vars=securities, var_name="security", value_name="strike")
        dataframe = dataframe.where(dataframe["strike"].notna()).dropna(how="all", inplace=False)
        dataframe["security"] = dataframe["security"].apply(Variables.Securities)
        dataframe["instrument"] = dataframe["security"].apply(lambda security: security.instrument)
        dataframe["option"] = dataframe["security"].apply(lambda security: security.option)
        dataframe["position"] = dataframe["security"].apply(lambda security: security.position)
        dataframe["quantity"] = 1
        return dataframe[holdings_header]

    @staticmethod
    def groupings(holdings, *args, **kwargs):
        holdings = holdings.groupby(holdings_header, as_index=False, dropna=False, sort=False).sum()
        for (ticker, expire), dataframe in iter(holdings.groupby(["ticker", "expire"])):
            yield (ticker, expire), dataframe

    @property
    def source(self): return self.__source
    @property
    def valuation(self): return self.__valuation



