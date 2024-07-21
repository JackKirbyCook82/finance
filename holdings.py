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
valuation_index = ["ticker", "expire", "strategy", "valuation"] + list(map(str, Variables.Securities))
valuation_stacking = {Variables.Valuations.ARBITRAGE: {"apy", "npv", "cost"}}

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
        self.__identify = np.arange(0, dtype=np.int32)
        self.__destination = destination
        self.__valuation = valuation
        self.__liquidity = liquidity
        self.__priority = priority

    def execute(self, contents, *args, **kwargs):
        contract, valuations = contents[Variables.Querys.CONTRACT], contents[self.valuation]
        assert isinstance(valuations, pd.DataFrame)
        if bool(valuations.empty):
            return
        if self.blocking(contract, *args, **kwargs):
            return
        valuations = self.parse(valuations, *args, **kwargs)
        valuations = self.market(valuations, *args, **kwargs)
        valuations = self.prioritize(valuations, *args, **kwargs)
        valuations = self.status(valuations, *args, **kwargs)
        valuations = self.tagging(valuations, *args, **kwargs)
        if bool(valuations.empty):
            return
        with self.destination.mutex:
            self.write(valuations, *args, **kwargs)

    def blocking(self, contract, *args, **kwargs):
        if not bool(self.destination):
            return False
        ticker = self.destination[:, "ticker"] == contract.ticker
        expire = self.destination[:, "expire"] == contract.expire
        prospect = self.destination[:, "status"] != Variables.Status.PROSPECT
        purchased = self.destination[:, "status"] != Variables.Status.PURCHASED
        blocking = (prospect & purchased)
        return (ticker & expire & blocking).any()

    def parse(self, valuations, *args, **kwargs):
        index = set(valuations.columns) - ({"scenario"} | valuation_stacking[self.valuation])
        valuations = valuations.pivot(index=list(index), columns="scenario")
        valuations = valuations.reset_index(drop=False, inplace=False)
        return valuations

    def market(self, valuations, *args, tenure=None, **kwargs):
        if tenure is not None:
            current = (pd.to_datetime("now") - valuations["current"]) <= tenure
            valuations = valuations.where(current).dropna(how="all", inplace=False)
        valuations["liquidity"] = valuations.apply(self.liquidity, axis=1)
        return valuations

    def prioritize(self, valuations, *args, **kwargs):
        valuations["priority"] = valuations.apply(self.priority, axis=1)
        valuations = valuations.sort_values("priority", axis=0, ascending=False, inplace=False, ignore_index=False)
        valuations = valuations.where(valuations["priority"] > 0).dropna(axis=0, how="all")
        return valuations

    def status(self, valuations, *args, **kwargs):
        valuations["status"] = self.destination[:, "status"] if bool(self.destination) else np.NaN
        valuations["status"] = valuations["status"].fillna(Variables.Status.PROSPECT)
        return valuations

    def tagging(self, valuations, *args, **kwargs):
        function = lambda tag: next(self.identify) if pd.isna(tag) else tag
        valuations["tag"] = self.destination[:, "tag"] if bool(self.destination) else np.NaN
        valuations["tag"] = valuations["tag"].apply(function)
        return valuations

    def write(self, valuations, *args, **kwargs):
        self.destination.concat(valuations, duplicates=valuation_index)
        self.destination.sort("priority", reverse=True)

    @property
    def destination(self): return self.__destination
    @property
    def valuation(self): return self.__valuation
    @property
    def liquidity(self): return self.__liquidity
    @property
    def priority(self): return self.__priority
    @property
    def identify(self): return self.__identify


class HoldingReader(Producer, ABC):
    def __init__(self, *args, source, **kwargs):
        super().__init__(*args, **kwargs)
        self.__source = source

    def execute(self, *args, **kwargs):
        with self.source.mutex:
            valuations = self.read(*args, **kwargs)
        assert isinstance(valuations, pd.DataFrame)
        if bool(valuations.empty):
            return
        valuations = self.parse(valuations, *args, **kwargs)
        holdings = self.holdings(valuations, *args, **kwargs)
        for (ticker, expire), dataframe in self.groupings(holdings, *args, **kwargs):
            contract = Contract(ticker, expire)
            holdings = {Variables.Querys.CONTRACT: contract, Variables.Datasets.HOLDINGS: dataframe}
            yield holdings

    def read(self, *args, **kwargs):
        if not bool(self.source):
            return pd.DataFrame()
        mask = self.source[:, "status"] == Variables.Status.PURCHASED
        dataframe = self.source.where(mask)
        self.source.remove(dataframe)
        dataframe["quantity"] = 1
        return dataframe

    @staticmethod
    def parse(valuations, *args, **kwargs):
        valuations = valuations.droplevel("scenario", axis=1)
        columns = [column for column in holdings_header if column in valuations.columns]
        valuations = valuations[columns + list(Variables.Securities)]
        return valuations

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

    @staticmethod
    def groupings(holdings, *args, **kwargs):
        index = [column for column in holdings_header if column != "quantity"]
        holdings = holdings.groupby(index, as_index=False, dropna=False, sort=False)["quantity"].sum()
        for (ticker, expire), dataframe in iter(holdings.groupby(["ticker", "expire"])):
            yield (ticker, expire), dataframe

    @property
    def source(self): return self.__source




