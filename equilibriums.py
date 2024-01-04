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

from support.pipelines import Processor, Reader, Writer
from support.tables import DataframeTable
from support.files import DataframeFile

from finance.securities import Securities
from finance.valuations import Valuations

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SupplyDemandFile", "SupplyDemandReader", "EquilibriumCalculator", "EquilibriumWriter", "EquilibriumTable"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)
INDEX = ["ticker", "date", "expire"] + list(map(str, Securities.Options))
COLUMNS = ["strategy", "apy", "npv", "cost", "tau", "size"]


class SupplyDemandQuery(ntuple("Query", "current ticker expire supply demand")): pass
class EquilibriumQuery(ntuple("Query", "current ticker expire equilibrium")):
    def __str__(self): return "{}|{}, {:.0f}".format(self.ticker, self.expire.strftime("%Y-%m-%d"), len(self.equilibrium.index))


class SupplyDemandFile(DataframeFile):
    def __init__(self, *args, **kwargs):
        datatypes = {"strike": np.float32, "price": np.float32, "size": np.int64, "npv": np.float32, "apy": np.float32, "cost": np.float32, "size": np.int64, "tau": np.int16}
        datetypes = ["date", "expire"]
        parameters = dict(datatypes=datatypes, datetypes=datetypes)
        super().__init__(*args, **parameters, **kwargs)


class SupplyDemandReader(Reader):
    def execute(self, *args, tickers=None, expires=None, dates=None, **kwargs):
        TickerExpire = ntuple("TickerExpire", "ticker expire")
        function = lambda foldername: Datetime.strptime(foldername, "%Y%m%d_%H%M%S")
        current_folders = self.source.folders()
        for current_name in sorted(current_folders, key=function, reverse=False):
            current = function(current_name)
            if dates is not None and current.date() not in dates:
                continue
            ticker_expire_folders = self.source.folders(current_name)
            for ticker_expire_name in ticker_expire_folders:
                ticker_expire = TickerExpire(*str(ticker_expire_name).split("_"))
                ticker = str(ticker_expire.ticker).upper()
                if tickers is not None and ticker not in tickers:
                    continue
                expire = Datetime.strptime(os.path.splitext(ticker_expire.expire)[0], "%Y%m%d").date()
                if expires is not None and expire not in expires:
                    continue
                filenames = {valuation: str(valuation).replace("|", "_") + ".csv" for valuation in list(Valuations)}
                files = {valuation: self.source.file(current_name, ticker_expire_name, filename) for valuation, filename in filenames.items()}
                valuations = {valuation: self.source.read(file=file) for valuation, file in files.items() if os.path.isfile(file)}
                if not bool(valuations) or all([dataframe.empty for dataframe in valuations.values()]):
                    continue
                filenames = {option: str(option).replace("|", "_") + ".csv" for option in list(Securities.Options)}
                files = {option: self.source.file(current_name, ticker_expire_name, filename) for option, filename in filenames.items()}
                options = {option: self.source.read(file=file) for option, file in files.items()}
                yield SupplyDemandQuery(current, ticker, expire, options, valuations)


class EquilibriumCalculator(Processor):
    def __init__(self, *args, valuation=Valuations.Arbitrage.Minimum, **kwargs):
        super().__init__(*args, **kwargs)
        self.valuation = valuation

    def execute(self, query, *args, **kwargs):
        if not bool(query.demand) or not bool(query.supply):
            return
        options, valuations = query.supply, query.demand[self.valuation]
        supply = self.supply(options, *args, **kwargs)
        demand = self.demand(valuations, *args, **kwargs)
        equilibrium = self.equilibrium(supply, demand, *args, *kwargs)
        equilibrium = pd.DataFrame.from_records(list(equilibrium))
        if bool(equilibrium.empty):
            return
        equilibrium = self.format(equilibrium, *args, **kwargs)
        equilibrium = self.parser(equilibrium, *args, **kwargs)
        if bool(equilibrium.empty):
            return
        query = EquilibriumQuery(query.current, query.ticker, query.expire, equilibrium)
        LOGGER.info("Equilibrium: {}[{}]".format(repr(self), str(query)))
        yield query

    @staticmethod
    def supply(options, *args, liquidity=None, **kwargs):
        liquidity = liquidity if liquidity is not None else 1
        options = {str(option): dataframe for option, dataframe in options.items()}
        assert set([str(option) for option in options.keys()]) == set([str(option) for option in list(Securities.Options)])
        for option, dataframe in options.items():
            dataframe.drop_duplicates(subset=["ticker", "date", "expire", "strike"], keep="last", inplace=True)
            dataframe.set_index(["ticker", "date", "expire", "strike"], drop=True, inplace=True)
            dataframe["size"] = (dataframe["size"] * liquidity).apply(np.floor).astype(np.int32)
        options = pd.concat([dataframe["size"].rename(option) for option, dataframe in options.items()], axis=1)
        return options

    @staticmethod
    def demand(valuations, *args, liquidity=None, apy=None, **kwargs):
        liquidity = liquidity if liquidity is not None else 1
        subset = ["strategy", "ticker", "date", "expire"] + list(map(str, Securities.Options))
        for option in list(Securities.Options):
            if str(option) not in valuations.columns:
                valuations[str(option)] = np.NaN
        valuations = valuations.where(valuations["apy"] >= apy) if apy is not None else valuations
        valuations = valuations.dropna(axis=0, how="all")
        valuations = valuations.drop_duplicates(subset=subset, keep="last", inplace=False)
        valuations = valuations.sort_values("apy", axis=0, ascending=False, inplace=False, ignore_index=False)
        valuations = valuations.set_index(INDEX, inplace=False, drop=True)[COLUMNS]
        valuations["size"] = (valuations["size"] * liquidity).apply(np.floor).astype(np.int32)
        return valuations

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
    def format(dataframe, *args, **kwargs):
        dataframe["apy"] = dataframe["apy"].round(2)
        dataframe["npv"] = dataframe["npv"].round(2)
        dataframe["tau"] = dataframe["tau"].astype(np.int32)
        dataframe["size"] = dataframe["size"].apply(np.floor).astype(np.int32)
        return dataframe

    @staticmethod
    def parser(equilibrium, *args, **kwargs):
        equilibrium = equilibrium.where(equilibrium["size"] > 0)
        equilibrium = equilibrium.dropna(axis=0, how="all")
        equilibrium = equilibrium.reset_index(drop=True, inplace=False)
        return equilibrium


class EquilibriumWriter(Writer):
    def execute(self, content, *args, **kwargs):
        equilibrium = content.equilibrium if isinstance(content, EquilibriumQuery) else content
        assert isinstance(equilibrium, pd.DataFrame)
        if bool(equilibrium.empty):
            return
        self.write(equilibrium[INDEX + COLUMNS], *args, **kwargs)
        LOGGER.info("Equilibrium: {}[{}]".format(repr(self), str(self.destination)))

    @staticmethod
    def format(dataframe, *args, **kwargs):
        dataframe["tau"] = dataframe["tau"].astype(np.int32)
        dataframe["size"] = dataframe["size"].apply(np.floor).astype(np.int32)
        return dataframe


class EquilibriumTable(DataframeTable):
    def __str__(self): return "{:,.02f}%, ${:,.0f}|${:,.0f}".format(self.apy * 100, self.npv, self.cost)

    @staticmethod
    def parser(dataframe, *args, **kwargs):
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe

    @staticmethod
    def format(dataframe, *args, **kwargs):
        dataframe["tau"] = dataframe["tau"].astype(np.int32)
        dataframe["size"] = dataframe["size"].apply(np.floor).astype(np.int32)
        return dataframe

    @property
    def apy(self): return self.table["apy"] @ self.weights
    @property
    def weights(self): return (self.table["cost"] * self.table["size"]) / (self.table["cost"] @ self.table["size"])
    @property
    def tau(self): return self.table["tau"].min(), self.table["tau"].max()
    @property
    def npv(self): return self.table["npv"] @ self.table["size"]
    @property
    def cost(self): return self.table["cost"] @ self.table["size"]
    @property
    def size(self): return self.table["size"].sum()



