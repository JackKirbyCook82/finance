# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Target Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
from abc import ABC
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.processes import Writer
from support.tables import DataframeTable

from finance.variables import Securities, Strategies

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["AcquisitionWriter", "DivestitureWriter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


VALUES = {"apy": np.float32, "npv": np.float32, "cost": np.float32, "size": np.int32, "tau": np.int32}
SCOPE = {"strategy": str, "valuation": str, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
INDEX = {option: str for option in list(map(str, Securities.Options))}
COLUMNS = {"scenario": str}

Header = ntuple("Header", "index columns scope values")
Contract = ntuple("Contract", "ticker expire")
Option = ntuple("Content", "instrument position strike")
Holding = ntuple("Holding", "contract option")


class TargetHoldings(tuple):
    def __new__(cls, records):
        assert isinstance(records, list)
        assert all([isinstance(record, dict) for record in records])
        function = lambda record: record["strike"]
        records = sorted(records, key=function, reverse=False)
        holdings = [Holding(record["contract"], record["option"]) for record in records]
        return super().__new__(cls, holdings)

    def __bool__(self): pass
    def __add__(self, other): pass
    def __sub__(self, other): pass

    @property
    def ceiling(self): pass
    @property
    def floor(self): pass

    @property
    def bull(self): pass
    @property
    def bear(self): pass

    @property
    def instruments(self): pass
    @property
    def positions(self): pass
    @property
    def strikes(self): pass


class TargetTable(DataframeTable, INDEX | COLUMNS | SCOPE | VALUES):
    def write(self, dataframe, *args, **kwargs):
        super().write(dataframe, *args, **kwargs)
        self.table.sort_values("priority", axis=0, ascending=False, inplace=True, ignore_index=False)


class TargetWriter(Writer, ABC):
    def __init__(self, *args, table, **kwargs):
        assert isinstance(table, TargetTable)
        super().__init__(*args, **kwargs)
        self.__table = table

    def __init__(self, *args, valuation, priority, header={}, **kwargs):
        market = Header(list(INDEX.keys()), list(COLUMNS.keys()), list(SCOPE.keys()), list(VALUES.keys()))
        super().__init__(*args, **kwargs)
        assert callable(priority)
        self.header = header | {"market": market}
        self.priority = priority
        self.valuation = valuation

    def market(self, valuations, *args, liquidity=None, apy=None, **kwargs):
        function = lambda cols: (np.min(cols.values) * liquidity).apply(np.floor).astype(np.int32)
        header = self.header["market"]._asdict()
        liquidity = liquidity if liquidity is not None else 1
        apy = apy if apy is not None else 0
        mask = valuations["valuation"] == str(self.valuation.name).lower()
        valuations = valuations.where(mask).dropna(axis=0, how="all")
        scenarios = set(valuations["scenario"].values)
        valuations = self.pivot(valuations, *args, **header, **kwargs)
        columns = [(scenario, "size") for scenario in scenarios]
        valuations["liquidity"] = valuations[columns].apply(function)
        mask = valuations["liquidity"] > liquidity & valuations["apy"] > apy
        valuations = valuations.where(mask).dropna(axis=0, how="all")
        valuations = valuations.reset_index(drop=True, inplace=False)
        return valuations

    def prioritize(self, market, *args, **kwargs):
        market["priority"] = market.apply(self.priority)
        market = market.sort_values("priority", axis=0, ascending=False, inplace=False, ignore_index=False)
        mask = market["priority"] > 0
        market = market.where(mask).dropna(axis=0, how="all")
        market = market.reset_index(drop=True, inplace=False)
        return market

    def write(self, dataframe, *args, **kwargs):
        self.table.write(dataframe, *args, **kwargs)
        size = self.size(dataframe["liquidity"])
        __logger__.info("Wrote: {}[{:.0f}]".format(repr(self), size))

    @property
    def table(self): return self.__table


class AcquisitionWriter(TargetWriter):
    def execute(self, query, *args, **kwargs):
        valuations = query.valuations
        assert isinstance(valuations, pd.DataFrame)
        market = self.market(valuations, *args, **kwargs)
        market = self.prioritize(market, *args, **kwargs)
        if bool(market.empty):
            return
        self.write(market, *args, **kwargs)


class DivestitureWriter(TargetWriter):
    def __init__(self, *args, **kwargs):
        portfolio = Header(["security", "position", "strike", "ticker", "expire", "date"], ["quantity"], None, None)
        header = {"portfolio": portfolio}
        super().__init__(*args, header=header, **kwargs)

    def execute(self, query, *args, **kwargs):
        valuations, holdings = query.valuations, query.holdings
        assert isinstance(valuations, pd.DataFrame) and isinstance(holdings, pd.DataFrame)
        market = self.market(valuations, *args, **kwargs)
        market = self.prioritize(market, *args, **kwargs)
        if bool(market.empty):
            return
        portfolio = self.portfolio(holdings, *args, **kwargs)
        market = self.closures(market, portfolio, *args, **kwargs)
        if bool(market.empty):
            return
        self.write(market, *args, **kwargs)

    def portfolio(self, holdings, *args, **kwargs):
        index = self.header["portfolio"].index
        columns = self.header["portfolio"].columns
        holdings = holdings.set_index(index, drop=True, inplace=False)
        holdings = holdings.loc[holdings.index.repeat(holdings[columns])]
        holdings = holdings.reset_index(drop=True, inplace=False)[index]
        return holdings

    def closures(self, market, portfolio, *args, **kwargs):
        divesting = self.divesting(portfolio)
        function = lambda series: bool(divesting - self.acquiring(series))
        mask = market.apply(function)
        market = market.where(mask).dropna(axis=0, how="all")
        market = market.reset_index(drop=True, inplace=False)
        return market

    @staticmethod
    def acquiring(market):
        assert isinstance(market, pd.Series)
        acquisitions = market.to_dict(into=ODict)
        contract = Contract(acquisitions["ticker"], acquisitions["expire"])
        securities = Strategies[acquisitions["strategy"]].securities
        options = {security: market.get(str(security), np.NaN) for security in securities}
        options = {security: strike for security, strike in options.items() if not np.isnan(strike)}
        options = [Option(security.instrument, security.position, strike) for security, strike in options.items()]
        acquisitions = [dict(contract=contract, option=option) for option in options]
        return TargetHoldings(acquisitions)

    @staticmethod
    def divesting(portfolio):
        assert isinstance(portfolio, pd.DataFrame)
        divestitures = list(portfolio.to_dict("records"))
        contracts = [Contract(divestiture["ticker"], divestiture["expire"]) for divestiture in divestitures]
        options = [Option(divestiture["security"], divestiture["position"], divestiture["strike"]) for divestiture in divestitures]
        divestitures = [dict(contract=contract, option=option) for contract, option in zip(contracts, options)]
        return TargetHoldings(divestitures)



