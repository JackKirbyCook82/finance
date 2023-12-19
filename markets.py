# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Market Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
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

    def curve(self, stop, *args, start=0, steps=10, **kwargs):
        curve = pd.concat([self(funds).results for funds in reversed(range(start, stop, steps))], axis=0)
        return CurveQuery(self.current, self.ticker, self.expire, self.valuation, curve)

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


class CurveQuery(ntuple("Query", "current ticker expire valuation curve")):
    pass


class MarketCalculator(Processor):
    def execute(self, query, *args, **kwargs):
        valuations = {str(valuation): dataframe for valuation, dataframe in query.valuations.items()}
        valuations = {valuation: self.parser(valuation, dataframe, *args, **kwargs) for valuation, dataframe in valuations.items()}
        securities = {str(security): dataframe for security, dataframe in query.securities.items() if security in list(Securities.Options)}
        securities = {security: self.parser(security, dataframe, *args, **kwargs) for security, dataframe in securities.items()}
        securities = pd.concat(list(securities.values()), axis=1)
        for valuation, dataframe in valuations.items():

            print(valuation)
            print(dataframe)
            print(securities)
            raise Exception()

            demand = dataframe.sort_values(["apy", "npv"], axis=0, ascending=False, inplace=False, ignore_index=True)
            market = list(self.market(supply, demand))
            market = pd.DataFrame.from_records(market)
            market = dataframe.where(market["size"] > 0)
            market = market.dropna(axis=0, how="all")
            yield MarketQuery(query.current, query.ticker, query.expire, valuation, market)

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

    @staticmethod
    def market(supply, demand):
        options = [str(option) for option in list(Securities.Options)]
        for row in demand.itertuples():
            row = row.to_dict()
            strikes = [row[option] for option in options]
            sizes = [supply.loc[strike, option] for strike, option in zip(strikes, options)]
            size = np.min([row.pop("size")] + sizes)
            for strike, option in zip(strikes, options):
                supply.at[strike, option] = supply.loc[strike, option] - size
            row = row | {"size": size}
            yield row



