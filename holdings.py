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
    filename = lambda contract: "_".join([str(contract.ticker).upper(), str(contract.expire.strftime("%Y%m%d"))])
    datatype = pd.DataFrame

class Axes:
    options = list(map(str, Variables.Securities.Options))
    stocks = list(map(str, Variables.Securities.Stocks))
    securities = list(map(str, Variables.Securities))
    scenarios = list(map(str, Variables.Scenarios))
    valuation = ["current", "size", "tau", "underlying"]
    arbitrage = ["apy", "npv", "cost"]
    security = ["instrument", "option", "position"]
    contract = ["ticker", "expire"]

class Headers:
    holdings = Axes.contract + Axes.security + ["strike", "quantity"]
    arbitrage = Axes.contract + ["valuation", "strategy"] + Axes.options + list(product(Axes.arbitrage, Axes.scenarios)) + Axes.valuation + ["status", "priority"]


class Formatting(metaclass=ParametersMeta):
    order = ["ticker", "expire", "valuation", "strategy"] + Axes.options + [(lead, lag) for lead, lag in product(Axes.arbitrage, Axes.scenarios)] + ["size", "status"]
    formats = {(lead, lag): lambda column: f"{column:.02f}" for lead, lag in product(Axes.arbitrage[1:], Axes.scenarios)}
    formats.update({(lead, lag): lambda column: f"{column * 100:.02f}%" if np.isfinite(column) else "InF" for lead, lag in product(Axes.arbitrage[0], Axes.scenarios)})
    formats.update({"priority": lambda priority: f"{priority * 100:.02f}%" if np.isfinite(priority) else "InF"})
    formats.update({"status": lambda status: str(status), "size": lambda size: f"{size:.02f}"})
    numbers = lambda column: f"{column:.02f}"


class HoldingView(Views.Dataframe, rows=20, columns=30, width=250, **dict(Formatting)): pass
class HoldingTable(Tables.Dataframe, tableview=HoldingView, tableaxes=Headers.arbitrage): pass
class HoldingFile(File, variable=Variables.Datasets.HOLDINGS, header=Headers.holdings, **dict(Parameters)): pass
class HoldingFiles(object): Holding = HoldingFile


class HoldingMixin(Mixin):
    def __init__(self, *args, datatable, valuation, **kwargs):
        super().__init__(*args, **kwargs)
        self.__stacking = getattr(Axes, str(valuation).lower())
        self.__datatable = datatable
        self.__valuation = valuation

    @property
    def datatable(self): return self.__datatable
    @property
    def valuation(self): return self.__valuation
    @property
    def stacking(self): return self.__stacking


class HoldingWriter(HoldingMixin, Consumer, variable=Variables.Querys.CONTRACT):
    def __init__(self, *args, priority, **kwargs):
        super().__init__(*args, **kwargs)
        self.__status = Variables.Status.PROSPECT
        self.__identity = count(1, step=1)
        self.__priority = priority

    def consumer(self, contents, *args, **kwargs):
        contract, valuations = contents[Variables.Querys.CONTRACT], contents[self.valuation]
        assert isinstance(valuations, pd.DataFrame)
        if bool(valuations.empty): return
        with self.datatable.mutex:
            self.obsolete(contract, *args, **kwargs)
            valuations = self.valuations(valuations, *args, **kwargs)
            valuations = self.prioritize(valuations, *args, **kwargs)
            valuations = self.identify(valuations, *args, **kwargs)
            valuations = self.prospect(valuations, *args, **kwargs)
            self.write(valuations, *args, **kwargs)

    def obsolete(self, contract, *args, **kwargs):
        if not bool(self.datatable): return
        ticker = lambda dataframe: dataframe["ticker"] == contract.ticker
        expire = lambda dataframe: dataframe["expire"] == contract.expire
        status = lambda dataframe: dataframe["status"] == Variables.Status.PROSPECT
        obsolete = lambda dataframe: ticker(dataframe) & expire(dataframe) & status(dataframe)
        self.datatable.remove(obsolete)

    def valuations(self, valuations, *args, **kwargs):
        if not bool(self.datatable): return valuations
        columns = Axes.contract + ["strategy"] + Axes.options

#        overlap = self.datatable.inner(valuations, on=columns).right

        valuations = pd.concat([valuations, overlap], axis=0)
        valuations = valuations.drop_duplicates(columns, keep="last", inplace=False)
        return valuations

    def prioritize(self, valuations, *args, **kwargs):
        valuations["priority"] = valuations.apply(self.priority, axis=1)
        parameters = dict(ascending=False, inplace=False, ignore_index=False)
        valuations = valuations.sort_values("priority", axis=0, **parameters)
        return valuations

    def identify(self, valuations, *args, **kwargs):
        if "identity" not in valuations.columns.levels[0]: valuations["identity"] = np.NaN
        function = lambda tag: next(self.identity) if np.isnan(tag) else tag
        valuations["identity"] = valuations.xs("identity", level=0, axis=1).apply(function)
        return valuations

    def prospect(self, valuations, *args, **kwargs):
        if "status" not in valuations.columns.levels[0]: valuations["status"] = np.NaN
        function = lambda status: Variables.Status.PROSPECT if np.isnan(self.status) else status
        valuations["status"] = valuations.xs("status", level=0, axis=1).apply(function)
        return valuations

    def write(self, valuations, *args, **kwargs):
        valuations = valuations.set_index("identity", drop=False, inplace=False)
        columns = Axes.contract + ["strategy"] + Axes.options
        self.datatable.concat(valuations)
        self.datatable.unique(columns)
        self.datatable.sort("priority", reverse=True)

    @property
    def priority(self): return self.__priority
    @property
    def identity(self): return self.__identity
    @property
    def status(self): return self.__status


class HoldingReader(HoldingMixin, Producer, variable=Variables.Querys.CONTRACT):
    def producer(self, *args, **kwargs):
        with self.datatable.mutex:
            self.obsolete(*args, **kwargs)
            valuations = self.read(*args, **kwargs)
            valuations = self.valuation(valuations, *args, **kwargs)
        if bool(valuations.empty): return
        valuations = self.parse(valuations, *args, **kwargs)
        valuations = self.stocks(valuations, *args, **kwargs)
        holdings = self.holdings(valuations, *args, **kwargs)
        for (ticker, expire), dataframe in self.groupings(holdings, *args, **kwargs):
            contract = Contract(ticker, expire)
            holdings = {Variables.Querys.CONTRACT: contract, Variables.Datasets.HOLDINGS: dataframe}
            yield dict(holdings)

    def read(self, *args, **kwargs):
        if not bool(self.datatable): return pd.DataFrame(columns=self.datatable.columns)
        accepted = lambda dataframe: dataframe["status"] == Variables.Status.ACCEPTED
        valuations = self.datatable.extract(accepted)
        return valuations

    def obsolete(self, *args, tenure=None, **kwargs):
        if not bool(self.datatable): return
        rejected = lambda dataframe: dataframe["status"] == Variables.Status.REJECTED
        abandoned = lambda dataframe: dataframe["status"] == Variables.Status.ABANDONED
        timeout = lambda dataframe: (pd.to_datetime("now") - dataframe["current"]) >= tenure if (tenure is not None) else False
        obsolete = lambda dataframe: rejected(dataframe) | abandoned(dataframe) | timeout(dataframe)
        self.datatable.remove(obsolete)

    def valuations(self, valuations, *args, **kwargs):
        index = set(valuations.columns) - ({"scenario"} | self.stacking)
        valuations = valuations[list(index)].droplevel("scenario", axis=1)
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




