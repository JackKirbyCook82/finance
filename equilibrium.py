# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 2023
@name:   Equilibrium Objects
@author: Jack Kirby Cook

"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime as Datetime
from collections import OrderedDict as ODict
from collections import namedtuple as ntuple

from support.pipelines import Calculator, Loader, Table

from finance.securities import Securities
from finance.valuations import Valuations

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SupplyDemandLoader", "EquilibriumCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)
INDEX = ["ticker", "date", "expire"] + list(map(str, Securities.Options))
COLUMNS = ["strategy", "apy", "npv", "cost", "tau", "size"]


class SupplyDemandQuery(ntuple("Query", "current ticker expire supply demand")): pass
class EquilibriumQuery(ntuple("Query", "current ticker expire equilibrium")):
    def __str__(self): return "{}|{}, {:.0f}".format(self.ticker, self.expire.strftime("%Y-%m-%d"), len(self.equilibrium.index))


class SupplyDemandFile(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        datetypes = {str(Valuations.Arbitrage.Minimum): ["date", "expire"], str(Valuations.Arbitrage.Maximum): ["date", "expire"], str(Valuations.Arbitrage.Current): ["date", "expire"]}
        datetypes.update({str(Securities.Option.Put.Long): ["date", "expire"], str(Securities.Option.Put.Short): ["date", "expire"]})
        datetypes.update({str(Securities.Option.Call.Long): ["date", "expire"], str(Securities.Option.Call.Short): ["date", "expire"]})
        options = {"strike": np.float32, "price": np.float32, "size": np.int64}
        valuation = {"npv": np.float32, "apy": np.float32, "cost": np.float32, "size": np.int64, "tau": np.int16}
        datatypes = {str(Valuations.Arbitrage.Minimum): valuation, str(Valuations.Arbitrage.Maximum): valuation, str(Valuations.Arbitrage.Current): valuation}
        datatypes.update({str(Securities.Option.Put.Long): options, str(Securities.Option.Put.Short): options})
        datatypes.update({str(Securities.Option.Call.Long): options, str(Securities.Option.Call.Short): options})
        self.__datetypes = datetypes
        self.__datatypes = datatypes

    @property
    def datetypes(self): return self.__datetypes
    @property
    def datatypes(self): return self.__datatypes


class SupplyDemandLoader(SupplyDemandFile, Loader):
    def execute(self, *args, tickers=None, expires=None, dates=None, **kwargs):
        TickerExpire = ntuple("TickerExpire", "ticker expire")
        function = lambda foldername: Datetime.strptime(foldername, "%Y%m%d_%H%M%S")
        datatypes = lambda key: self.datatypes[str(key)]
        datetypes = lambda key: self.datetypes[str(key)]
        reader = lambda key, file: self.read(file=file, filetype=pd.DataFrame, datatypes=datatypes(key), datetypes=datetypes(key))
        for current_name in sorted(os.listdir(self.repository), key=function, reverse=False):
            current = function(current_name)
            if dates is not None and current.date() not in dates:
                continue
            current_folder = os.path.join(self.repository, current_name)
            for ticker_expire_name in os.listdir(current_folder):
                ticker_expire = TickerExpire(*str(ticker_expire_name).split("_"))
                ticker = str(ticker_expire.ticker).upper()
                if tickers is not None and ticker not in tickers:
                    continue
                expire = Datetime.strptime(os.path.splitext(ticker_expire.expire)[0], "%Y%m%d").date()
                if expires is not None and expire not in expires:
                    continue
                ticker_expire_folder = os.path.join(current_folder, ticker_expire_name)
                with self.locks[ticker_expire_folder]:
                    filenames = {valuation: str(valuation).replace("|", "_") + ".csv" for valuation in list(Valuations)}
                    files = {valuation: os.path.join(ticker_expire_folder, filename) for valuation, filename in filenames.items()}
                    valuations = {valuation: reader(valuation, file) for valuation, file in files.items() if os.path.isfile(file)}
                    if not bool(valuations) or all([dataframe.empty for dataframe in valuations.values()]):
                        continue
                    filenames = {option: str(option).replace("|", "_") + ".csv" for option in list(Securities.Options)}
                    files = {option: os.path.join(ticker_expire_folder, filename) for option, filename in filenames.items()}
                    options = {option: reader(option, file) for option, file in files.items()}
                    yield SupplyDemandQuery(current, ticker, expire, options, valuations)


class EquilibriumCalculator(Calculator):
    def __init__(self, *args, name, valuation=Valuations.Arbitrage.Minimum, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        Variables = ntuple("Variables", "supply demand")
        self.__index = Variables(["ticker", "date", "expire", "strike"], INDEX)
        self.__columns = Variables(["size"], COLUMNS)
        self.__valuation = valuation

    def execute(self, query, *args, **kwargs):
        if bool(query.demand.empty):
            return
        options, valuations = query.supply, query.demand[self.valuation]
        supply = self.supply(options, *args, **kwargs)
        demand = self.demand(valuations, *args, **kwargs)
        equilibrium = self.equilibrium(supply, demand, *args, *kwargs)
        equilibrium = pd.DataFrame.from_records(list(equilibrium))
        if bool(equilibrium.empty):
            return
        equilibrium = self.parser(equilibrium, *args, **kwargs)
        if bool(equilibrium.empty):
            return
        query = EquilibriumQuery(query.current, query.ticker, query.expire, equilibrium)
        LOGGER.info("Equilibrium: {}[{}]".format(repr(self), str(query)))
        yield query

    def supply(self, options, *args, **kwargs):
        options = {str(option): dataframe for option, dataframe in options.items()}
        for option, dataframe in options.items():
            dataframe.drop_duplicates(subset=["ticker", "date", "expire", "strike"], keep="last", inplace=True)
            dataframe.set_index(self.index.supply, drop=True, inplace=True)
        options = pd.concat([dataframe[self.columns.supply].rename(option) for option, dataframe in options.items()], axis=1)
        return options

    def demand(self, valuations, *args, apy=None, **kwargs):
        subset = ["strategy", "ticker", "date", "expire"] + list(map(str, Securities.Options))
        valuations = valuations.where(valuations["apy"] >= apy) if apy is not None else valuations
        valuations = valuations.dropna(axis=0, how="all")
        valuations = valuations.drop_duplicates(subset=subset, keep="last", inplace=False)
        valuations = valuations.sort_values("apy", axis=0, ascending=False, inplace=False, ignore_index=False)
        valuations = valuations.set_index(self.index.demand, inplace=False, drop=True)
        return valuations[self.columns.demand]

    @staticmethod
    def equilibrium(supply, demand, *args, **kwargs):
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

    @staticmethod
    def parser(equilibrium, *args, **kwargs):
        equilibrium = equilibrium.where(equilibrium["size"] > 0)
        equilibrium = equilibrium.dropna(axis=0, how="all")
        equilibrium = equilibrium.reset_index(drop=True, inplace=False)
        return equilibrium

    @property
    def valuation(self): return self.__valuation
    @property
    def columns(self): return self.__columns
    @property
    def index(self): return self.__index


class EquilibriumTable(Table, index=INDEX, columns=COLUMNS):
    def __str__(self): return "{:.2f}%|${:.0f}".format(self.apy * 100, self.cost)

    def execute(self, content, *args, **kwargs):
        equilibrium = content.equilibrium if isinstance(content, EquilibriumTable) else content
        assert isinstance(equilibrium, pd.DataFrame)
        if bool(equilibrium.empty):
            return
        with self.mutex:
            equilibrium = equilibrium[self.index + self.columns]
            equilibrium = pd.concat([self.table, equilibrium], axis=0)
            equilibrium = equilibrium.reset_index(drop=True, inplace=False)
            self.table = equilibrium
            LOGGER.info("Equilibrium: {}[{}]".format(repr(self), str(self)))

    @property
    def weights(self):
        cost = self.table["cost"] / self.table["cost"].sum()
        size = self.table["size"] / self.table["size"].sum()
        weights = cost * size
        weights = weights / weights.sum()
        return weights

    @property
    def apy(self): return self.table["apy"] @ self.weights
    @property
    def npv(self): return self.table["npv"] @ self.table["size"]
    @property
    def cost(self): return self.table["cost"] @ self.table["size"]
    @property
    def size(self): return self.table["size"].sum()
    @property
    def tau(self): return self.table["tau"].min(), self.table["tau"].max()










