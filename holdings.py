# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Holdings Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
from abc import ABC
from itertools import product

from finance.variables import Variables, Contract
from support.pipelines import Producer, Consumer
from support.tables import Tables, Options
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
holdings_index = ["ticker", "expire", "strategy", "valuation", "scenario"]
holdings_columns = ["current", "apy", "npv", "cost", "size", "underlying"]
holdings_stacking = {Variables.Valuations.ARBITRAGE: {"apy", "npv", "cost"}}

holdings_formats = {(lead, lag): lambda column: f"{column:.02f}" for lead, lag in product(["npv", "cost"], list(map(lambda scenario: str(scenario), Variables.Scenarios)))}
holdings_formats.update({(lead, lag): lambda column: f"{column * 100:.02f}%" for lead, lag in product(["apy"], list(map(lambda scenario: str(scenario), Variables.Scenarios)))})
holdings_formats.update({("priority", ""): lambda priority: f"{priority:.02f}", ("status", ""): lambda status: str(status)})
holdings_options = Options.Dataframe(rows=20, columns=25, width=1000, formats=holdings_formats, numbers=lambda column: f"{column:.02f}")


class HoldingFile(File, variable=Variables.Datasets.HOLDINGS, header=holdings_header, **holdings_parameters): pass
class HoldingFiles(object): Holding = HoldingFile


class HoldingTable(Tables.Dataframe, datatype=pd.DataFrame, options=holdings_options):
    pass


class HoldingWriter(Consumer, ABC):
    def __init__(self, *args, destination, liquidity, priority, valuation, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__destination = destination
        self.__valuation = valuation
        self.__liquidity = liquidity
        self.__priority = priority

    def execute(self, contents, *args, **kwargs):
        dataframe = contents[self.valuation]
        assert isinstance(dataframe, pd.DataFrame)
        if bool(dataframe.empty):
            return
        dataframe = self.parse(dataframe, *args, **kwargs)
        dataframe = self.market(dataframe, *args, **kwargs)
        dataframe = self.prioritize(dataframe, *args, **kwargs)
        if bool(dataframe.empty):
            return
        with self.destination.mutex:
            self.write(dataframe)

    def write(self, dataframe):
        pass

#        dataframe["status"] = self.destination["status"] if bool(self.destination) else np.NaN
#        dataframe["status"] = dataframe["status"].fillna(Variables.Status.PROSPECT)
#        self.destination.update(dataframe)
#        self.destination.sort("priority", reverse=True)

    def parse(self, dataframe, *args, **kwargs):
        pass

#        index = set(dataframe.columns) - ({"scenario"} | holdings_stacking[self.valuation])
#        dataframe = dataframe.pivot(index=list(index), columns="scenario")
#        dataframe = dataframe.reset_index(drop=False, inplace=False)
#        index = list(holdings_index) + list(map(str, Variables.Securities.Options))
#        index = [column for column in index if column != "scenario"]
#        dataframe = dataframe.set_index(list(index), drop=True, inplace=False)
#        return dataframe

    def market(self, dataframe, *args, tenure=None, **kwargs):
        if tenure is not None:
            current = (pd.to_datetime("now") - dataframe["current"]) <= tenure
            dataframe = dataframe.where(current).dropna(how="all", inplace=False)
        dataframe["liquidity"] = dataframe.apply(self.liquidity, axis=1)
        return dataframe

    def prioritize(self, dataframe, *args, **kwargs):
        dataframe["priority"] = dataframe.apply(self.priority, axis=1)
        dataframe = dataframe.sort_values("priority", axis=0, ascending=False, inplace=False, ignore_index=False)
        dataframe = dataframe.where(dataframe["priority"] > 0).dropna(axis=0, how="all")
        return dataframe

    @property
    def destination(self): return self.__destination
    @property
    def valuation(self): return self.__valuation
    @property
    def liquidity(self): return self.__liquidity
    @property
    def priority(self): return self.__priority


class HoldingReader(Producer, ABC):
    def __init__(self, *args, source, **kwargs):
        super().__init__(*args, **kwargs)
        self.__source = source

    def execute(self, *args, **kwargs):
        with self.source.mutex:
            dataframe = self.read()
        assert isinstance(dataframe, pd.DataFrame)
        if bool(dataframe.empty):
            return
        dataframe = self.parse(dataframe, *args, **kwargs)
        contract = self.contract(dataframe, *args, **kwargs)
        options = self.options(dataframe, *args, **kwargs)
        stocks = self.stocks(dataframe, *args, **kwargs)
        securities = pd.concat([contract, options, stocks, dataframe["quantity"]], axis=1, ignore_index=False)
        holdings = self.holdings(securities, *args, **kwargs)
        index = [column for column in holdings_header if column != "quantity"]
        holdings = holdings.groupby(index, as_index=False, dropna=False, sort=False)["quantity"].sum()
        for (ticker, expire), dataframe in iter(holdings.groupby(["ticker", "expire"])):
            contract = Contract(ticker, expire)
            holdings = {Variables.Querys.CONTRACT: contract, Variables.Datasets.HOLDINGS: dataframe}
            yield holdings

    def read(self):
        pass

#        if not bool(self.source):
#            return pd.DataFrame()
#        mask = self.source["status"] == Variables.Status.PURCHASED
#        dataframe = self.source.where(mask)
#        self.source.remove(dataframe)
#        dataframe["quantity"] = 1
#        return dataframe

    @staticmethod
    def parse(dataframe, *args, **kwargs):
        pass

#        dataframe = dataframe.droplevel("scenario", axis=1)
#        dataframe = dataframe[["underlying", "quantity"]]
#        dataframe = dataframe.reset_index(drop=False, inplace=False)
#        return dataframe

    @staticmethod
    def contract(dataframe, *args, **kwargs): return dataframe[["ticker", "expire"]]
    @staticmethod
    def options(dataframe, *args, **kwargs): return dataframe[list(map(str, Variables.Securities.Options))]
    @staticmethod
    def stocks(dataframe, *args, **kwargs):
        securities = list(map(str, Variables.Securities.Stocks))
        strategies = lambda cols: list(map(str, cols["strategy"].stocks))
        underlying = lambda cols: np.round(cols["underlying"], decimals=2)
        function = lambda cols: [underlying(cols) if column in strategies(cols) else np.NaN for column in securities]
        dataframe = dataframe.apply(function, axis=1, result_type="expand")
        dataframe.columns = securities
        dataframe = dataframe.dropna(how="all", axis=1)
        return dataframe

    @staticmethod
    def holdings(securities, *args, **kwargs):
        columns = [security for security in list(map(str, Variables.Securities)) if security in securities.columns]
        contracts = [column for column in securities.columns if column not in columns]
        dataframe = securities.melt(id_vars=contracts, value_vars=columns, var_name="security", value_name="strike")
        dataframe = dataframe.where(dataframe["strike"].notna()).dropna(how="all", inplace=False)
        dataframe["security"] = dataframe["security"].apply(Variables.Securities)
        dataframe["instrument"] = dataframe["security"].apply(lambda security: security.instrument)
        dataframe["option"] = dataframe["security"].apply(lambda security: security.option)
        dataframe["position"] = dataframe["security"].apply(lambda security: security.position)
        return dataframe

    @property
    def source(self): return self.__source




