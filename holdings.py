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
from support.mixins import Mixin
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
    arbitrage = ["apy", "npv", "cost", "size", "tau", "underlying", "current"]
    security = ["instrument", "option", "position"]
    contract = ["ticker", "expire"]

class Headers:
    arbitrage = Axes.contract + ["valuation", "scenario", "strategy"] + Axes.options + Axes.arbitrage + ["status", "priority"]
    arbitrage = [(column, "scenario" if column in Axes.stacking[Variables.Valuations.ARBITRAGE] else "") for column in arbitrage]
    holdings = Axes.contract + Axes.security + ["strike", "quantity"]

class Formatting(metaclass=ParametersMeta):
    order = ["ticker", "expire", "valuation", "strategy"] + Axes.options
    order = order + [(lead, lag) for lead, lag in product(["apy", "npv", "cost"], Variables.Scenarios)] + ["size", "status"]
    numbers = lambda column: f"{column:.02f}"
    formats = {(lead, lag): lambda column: f"{column:.02f}" for lead, lag in product(["npv", "cost"], Variables.Scenarios)}
    formats.update({(lead, lag): lambda column: f"{column * 100:.02f}%" if np.isfinite(column) else "InF" for lead, lag in product(["apy"], Variables.Scenarios)})
    formats.update({"priority": lambda priority: f"{priority * 100:.02f}%" if np.isfinite(priority) else "InF"})
    formats.update({"status": lambda status: str(status), "size": lambda size: f"{size:.02f}"})


class HoldingView(Views.Dataframe, rows=20, columns=30, width=250, **dict(Formatting)): pass
class HoldingTable(Tables.Dataframe, tableview=HoldingView): pass
class HoldingFile(File, variable=Variables.Datasets.HOLDINGS, header=Headers.holdings, **dict(Parameters)): pass
class HoldingFiles(object): Holding = HoldingFile


class HoldingMixin(Mixin):
    def __init__(self, *args, datatable, valuation, **kwargs):
        super().__init__(*args, **kwargs)
        self.__datatable = datatable
        self.__valuation = valuation

#    def stack(self, column):
#        if isinstance(self.datatable.columns, pd.MultiIndex):
#            column = tuple([column]) if not isinstance(column, tuple) else column
#            length = self.datatable.columns.nlevels - len(column)
#            column = column + tuple([""]) * length
#        return column

    @property
    def datatable(self): return self.__datatable
    @property
    def valuation(self): return self.__valuation


class HoldingWriter(HoldingMixin, Consumer, variable=Variables.Querys.CONTRACT):
    def __init__(self, *args, priority, **kwargs):
        super().__init__(*args, **kwargs)
        self.__identity = count(1, step=1)
        self.__priority = priority

    def consumer(self, contents, *args, **kwargs):
        contract, valuations = contents[Variables.Querys.CONTRACT], contents[self.valuation]
        assert isinstance(valuations, pd.DataFrame)
        if bool(valuations.empty): return
        with self.datatable.mutex:
            pass

#            datatable = self.obsolete(contract, *args, **kwargs)
#            datatable = self.parser(valuations, *args, **kwargs)
#            valuations = self.valuations(valuations, *args, **kwargs)
#            valuations = self.prioritize(valuations, *args, **kwargs)
#            valuations = self.identify(valuations, *args, **kwargs)
#            valuations = self.prospect(valuations, *args, **kwargs)
#            self.write(valuations, *args, **kwargs)

    def obsolete(self, contract, *args, **kwargs):
        pass

#        if not bool(self.datatable): return
#        columns = Axes.contract + ["status"]
#        Columns = ntuple("Columns", columns)
#        columns = Columns(*list(map(self.stack, columns)))
#        ticker = lambda dataframe: dataframe[columns.ticker] == contract.ticker
#        expire = lambda dataframe: dataframe[columns.expire] == contract.expire
#        status = lambda dataframe: dataframe[columns.status] == Variables.Status.PROSPECT
#        function = lambda dataframe: ticker(dataframe) & expire(dataframe) & status(dataframe)
#        self.datatable.remove(function)

    def parser(self, valuations, *args, **kwargs):
        pass

#        valuation = str(self.valuation).lower()
#        columns = getattr(Headers, valuation, [])
#        assert valuations.columns == columns
#        datatable = HoldingTable(*args, contents=valuations, columns=columns, **kwargs)
#        return datatable

    def valuations(self, datatable, *args, **kwargs):
        pass

#        if not bool(self.datatable): return datatable
#        columns = Axes.contract + ["strategy"] + Axes.options
#        overlap = self.datatable.overlap(datatable, on=columns).right
#        datatable = datatable.concat(overlap)
#        datatable = datatable.unique(on=columns, reverse=False)
#        return datatable

    def prioritize(self, datatable, *args, **kwargs):
        pass

#        priority = self.stack("priority")
#        valuations[priority] = valuations.apply(self.priority, axis=1)
#        parameters = dict(ascending=False, inplace=False, ignore_index=False)
#        valuations = valuations.sort_values(priority, axis=0, **parameters)
#        return valuations

    def identify(self, valuations, *args, **kwargs):
        pass

#        identity = self.stack("identity")
#        valuations = valuations.assign(identity=np.NaN) if not bool(self.datatable) else valuations
#        function = lambda tag: next(self.identity) if np.isnan(tag) else tag
#        valuations[identity] = valuations[identity].apply(function)
#        return valuations

    def prospect(self, valuations, *args, **kwargs):
        pass

#        status = self.stack("status")
#        valuations = valuations.assign(status=np.NaN) if not bool(self.datatable) else valuations
#        valuations[status] = valuations[status].fillna(Variables.Status.PROSPECT)
#        return valuations

    def write(self, valuations, *args, **kwargs):
        pass

#        valuations = valuations.set_index("identity", drop=False, inplace=False)
#        self.datatable.concat(valuations)
#        self.datatable.unique(Axes.contract + ["strategy"] + Axes.options)
#        self.datatable.sort("priority", reverse=True)

    @property
    def priority(self): return self.__priority
    @property
    def identity(self): return self.__identity


class HoldingReader(HoldingMixin, Producer, variable=Variables.Querys.CONTRACT):
    def producer(self, *args, **kwargs):
        with self.datatable.mutex:
            pass

#           self.obsolete(*args, **kwargs)
#            self.cleaner(*args, **kwargs)
#            valuations = self.read(*args, **kwargs)
#        if bool(valuations.empty): return
#        valuations = self.parse(valuations, *args, **kwargs)
#        valuations = self.stocks(valuations, *args, **kwargs)
#        holdings = self.holdings(valuations, *args, **kwargs)
#        for (ticker, expire), dataframe in self.groupings(holdings, *args, **kwargs):
#            contract = Contract(ticker, expire)
#            holdings = {Variables.Querys.CONTRACT: contract, Variables.Datasets.HOLDINGS: dataframe}
#            yield dict(holdings)

    def read(self, *args, **kwargs):
        pass

#        if not bool(self.datatable): return pd.DataFrame()
#        status = self.stack("status")
#        accepted = lambda dataframe: dataframe[status] == Variables.Status.ACCEPTED
#        valuations = self.datatable.table.where(accepted).dropna(how="all", inplace=False)
#        self.datatable.remove(accepted)
#        return valuations

    def obsolete(self, *args, tenure=None, **kwargs):
        pass

#        if not bool(self.datatable): return
#        if tenure is None: return
#        current = self.stack("current")
#        obsolete = lambda dataframe: (pd.to_datetime("now") - dataframe[current]) >= tenure
#        self.datatable.remove(obsolete)

    def cleaner(self, *args, **kwargs):
        pass

#        if not bool(self.datatable): return
#        status = self.stack("status")
#        rejected = lambda dataframe: dataframe[status] == Variables.Status.REJECTED
#        abandoned = lambda dataframe: dataframe[status] == Variables.Status.ABANDONED
#        self.datatable.remove(rejected)
#        self.datatable.remove(abandoned)

    def parse(self, valuations, *args, **kwargs):
        pass

#        columns = set(valuations.columns) - ({"scenario"} | Axes.stacking[self.valuation])
#        valuations = valuations[list(columns)].droplevel("scenario", axis=1)
#        return valuations

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




