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

from support.processes import Writer

from finance.variables import Securities, Strategies

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["AcquisitionWriter", "DivestitureWriter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


# VALUES = {"apy": np.float32, "npv": np.float32, "cost": np.float32, "size": np.int32, "tau": np.int32}
# SCOPE = {"strategy": str, "valuation": str, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
# INDEX = {option: str for option in list(map(str, Securities.Options))}
# COLUMNS = {"scenario": str}

Contract = ntuple("Contract", "ticker expire")
Option = ntuple("Content", "instrument position strike")
Holding = ntuple("Holding", "contract option")


class TargetHoldings(tuple):
    def __new__(cls, records):
        assert isinstance(records, list)
        assert all([isinstance(record, dict) for record in records])
        function = lambda record: record["strike"]
        records = sorted(records, key=function, reverse=False)
        contracts = [Contract(record["ticker"], record["expire"]) for record in records]
        options = [Option(record["instrument"], record["position"], record["strike"]) for record in records]
        holdings = [Holding(contract, option) for contract, option in zip(contracts, options)]
        holdings = sorted(holdings, key=function, reverse=False)
        return super().__new__(cls, holdings)


class TargetWriter(Writer, ABC):
    def __init__(self, *args, valuation, priority, **kwargs):
        super().__init__(*args, **kwargs)
        assert callable(priority)
        self.priority = priority
        self.valuation = valuation

    def write(self, dataframe, *args, **kwargs):
        pass

    def market(self, dataframe, *args, liquidity=None, apy=None, **kwargs):
        scope = ["strategy", "valuation", "ticker", "expire", "date"]
        values = ["apy", "tau", "npy", "cost", "size"]
        index = list(map(str, Securities.Options))
        header = dict(index=index, columns=["scenario"], scope=scope, values=values)
        function = lambda cols: (np.min(cols.values) * liquidity).apply(np.floor).astype(np.int32)
        liquidity = liquidity if liquidity is not None else 1
        apy = apy if apy is not None else 0
        mask = dataframe["valuation"] == str(self.valuation.name).lower()
        dataframe = dataframe.where(mask).dropna(axis=0, how="all")
        scenarios = set(dataframe["scenario"].values)
        dataframe = self.pivot(dataframe, *args, **header, **kwargs)
        columns = [(scenario, "size") for scenario in scenarios]
        dataframe["liquidity"] = dataframe[columns].apply(function)
        mask = dataframe["liquidity"] > liquidity & dataframe["apy"] > apy
        dataframe = dataframe.where(mask).dropna(axis=0, how="all")
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe

    def prioritize(self, dataframe, *args, **kwargs):
        dataframe["priority"] = dataframe.apply(self.priority)
        dataframe = dataframe.sort_values("priority", axis=0, ascending=False, inplace=False, ignore_index=False)
        dataframe = dataframe.where(dataframe["priority"] > 0).dropna(axis=0, how="all")
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe


class AcquisitionWriter(TargetWriter):
    def execute(self, query, *args, **kwargs):
        valuations = query.valuations
        assert isinstance(valuations, pd.DataFrame)
        market = self.market(valuations, *args, **kwargs)
        market = self.prioritize(market, *args, **kwargs)
        if bool(market.empty):
            return


class DivestitureWriter(TargetWriter):
    def execute(self, query, *args, **kwargs):
        valuations, holdings = query.valuations, query.holdings
        assert isinstance(valuations, pd.DataFrame) and isinstance(holdings, pd.DataFrame)
        market = self.market(valuations, *args, **kwargs)
        market = self.prioritize(market, *args, **kwargs)
        if bool(market.empty):
            return
        portfolio = self.portfolio(holdings, *args, **kwargs)
        holdings = self.holdings(portfolio, *args, **kwargs)
        market = self.qualify(market, *args, holdings=holdings, **kwargs)
        if bool(market.empty):
            return


    @staticmethod
    def portfolio(dataframe, *args, **kwargs):
        index = ["instrument", "position", "strike", "ticker", "expire", "date"]
        dataframe = dataframe.set_index(index, drop=True, inplace=False)
        dataframe = dataframe.loc[dataframe.index.repeat(dataframe["quantity"])]
        dataframe = dataframe.reset_index(drop=True, inplace=False)[index]
        return dataframe

    @staticmethod
    def holdings(dataframe, *args, **kwargs):
        holdings = dataframe.to_dict("records")
        holdings = TargetHoldings(holdings)
        return holdings

    @staticmethod
    def qualify(dataframe, *args, holdings, **kwargs):
        pass


#    @staticmethod
#    def prospects(dataframe, *args, **kwargs):
#        for record in dataframe.to_dict("records"):
#            contract = Contract(record["ticker"], record["expire"])
#            securities = {security: record.get(str(security), np.NaN) for security in Strategies[str(record["strategy"])].securities}
#            options = [Option(security.instrument, security.position, strike) for security, strike in securities.items() if not np.isnan(strike)]
#            holdings = [Holding(contract, option) for option in options]
#            yield holdings



