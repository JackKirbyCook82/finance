# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Market Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
from collections import OrderedDict as ODict
from collections import namedtuple as ntuple

from support.pipelines import Processor
from support.dispatchers import argsdispatcher

from finance.securities import Securities
from finance.valuations import Valuations

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["MarketCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


class MarketQuery(ntuple("Query", "current ticker expire valuation market")):
    def __call__(self, funds):
        cumulative = (self.market["cost"] * self.market["size"]).cumsum()
        market = cumulative.where(cumulative > funds)
        return MarketQuery(self.current, self.ticker, self.expire, self.valuation, market)

#    def curve(self, stop, *args, start=0, steps=10, **kwargs):
#        curve = pd.concat([self(funds).results for funds in reversed(range(start, stop, steps))], axis=0)
#        return CurveQuery(self.current, self.ticker, self.expire, self.valuation, curve)

    @property
    def results(self): return {"apy": self.apy, "npv": self.npv, "cost": self.cost, "size": self.size, "tau-": self.market["tau"].min(), "tau+": self.market["tau"].max()}
    @property
    def weights(self): return (self.market["cost"] / self.market["cost"].sum()) * (self.market["size"] / self.market["size"].sum())
    @property
    def cost(self): return self.market["cost"] @ self.market["size"]
    @property
    def tau(self): return self.market["tau"].min(), self.market["tau"].max()
    @property
    def apy(self): return self.market["apy"] @ self.weights
    @property
    def npv(self): return self.market["npv"] @ self.market["size"]


# class CurveQuery(ntuple("Query", "current ticker expire valuation curve")):
#     pass


class MarketCalculator(Processor):
    def execute(self, query, *args, **kwargs):
        valuations = {str(valuation): dataframe for valuation, dataframe in query.valuations.items()}
        valuations = {valuation: self.parser(valuation, dataframe, *args, **kwargs) for valuation, dataframe in valuations.items()}
        securities = {str(security): dataframe for security, dataframe in query.securities.items() if security in list(Securities.Options)}
        securities = {security: self.parser(security, dataframe, *args, **kwargs) for security, dataframe in securities.items()}
        securities = pd.concat(list(securities.values()), axis=1)
        for valuation, dataframe in valuations.items():
            dataframe = dataframe.sort_values(["apy", "npv"], axis=0, ascending=False, inplace=False, ignore_index=False)
            market = list(self.market(securities, dataframe))
            if not bool(market):
                continue
            market = pd.DataFrame.from_records(market)
            market = market.where(market["size"] > 0)
            market = market.dropna(axis=0, how="all")
            if bool(market.empty):
                continue
            yield MarketQuery(query.current, query.ticker, query.expire, valuation, market)

    @staticmethod
    def market(supply, demand):
        purchased = pd.DataFrame(0, index=supply.index, columns=supply.columns, dtype=np.int16)
        for row in demand.itertuples(index=True):
            available = supply - purchased
            position = list(zip(demand.index.names, row.Index))
            options = list(map(str, Securities.Options))
            index = ODict([(key, value) for key, value in iter(position) if key not in list(options)])
            strikes = ODict([(key, value) for key, value in iter(position) if key in list(options)])
            columns = ODict([(column, getattr(row, column)) for column in demand.columns])
            locators = [(tuple([*index.values(), strike]), option) for option, strike in strikes.items() if not np.isnan(strike)]
            sizes = [available.loc[indx, cols] for (indx, cols) in locators]
            size = np.min([row.size, *sizes])
            for (indx, cols) in locators:
                purchased.loc[indx, cols] = purchased.loc[indx, cols] + size
            yield index | strikes | columns | {"size": size}

    @argsdispatcher(index=0)
    def parser(self, dataframe, *args, key, **kwargs): raise ValueError(str(key))

    @parser.register.value(*list(map(str, Securities.Options)))
    def option(self, security, dataframe, *args, **kwargs):
        dataframe = dataframe.drop_duplicates(subset=["ticker", "date", "expire", "strike"], keep="last", inplace=False)
        dataframe = dataframe.set_index(["ticker", "date", "expire", "strike"], inplace=False, drop=True)
        return dataframe["size"].rename(str(security))

    @parser.register.value(*list(map(str, Valuations)))
    def valuation(self, valuation, dataframe, *args, **kwargs):
        index = ["ticker", "date", "expire"] + list(map(str, Securities.Options))
        subset = ["strategy", "ticker", "date", "expire"] + list(map(str, Securities.Options))
        columns = ["strategy", "apy", "npv", "cost", "tau", "size"]
        dataframe = dataframe.drop_duplicates(subset=subset, keep="last", inplace=False)
        dataframe = dataframe.set_index(index, inplace=False, drop=True)
        return dataframe[columns]



