# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Valuation Objects
@author: Jack Kirby Cook

"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from enum import IntEnum
from datetime import datetime as Datetime
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.pipelines import Calculator, Processor, Saver, Loader
from support.calculations import Calculation, equation, source, constant
from support.dispatchers import typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Valuation", "Valuations", "Calculations"]
__all__ += ["ValuationSaver", "ValuationLoader", "ValuationFilter", "ValuationParser", "ValuationCalculator", "ValuationAnalysis"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


Basis = IntEnum("Basis", ["ARBITRAGE"], start=1)
Scenario = IntEnum("Scenario", ["CURRENT", "MINIMUM", "MAXIMUM"], start=1)
class Valuation(ntuple("Valuation", "basis scenario")):
    def __str__(self): return "|".join([str(value.name).lower() for value in self if bool(value)])
    def __int__(self): return int(self.basis) * 10 + int(self.scenario) * 1

CurrentArbitrage = Valuation(Basis.ARBITRAGE, Scenario.CURRENT)
MinimumArbitrage = Valuation(Basis.ARBITRAGE, Scenario.MINIMUM)
MaximumArbitrage = Valuation(Basis.ARBITRAGE, Scenario.MAXIMUM)

class ValuationsMeta(type):
    def __iter__(cls): return iter([MinimumArbitrage, MaximumArbitrage])
    def __getitem__(cls, indexkey): return cls.retrieve(indexkey)

    @typedispatcher
    def retrieve(cls, indexkey): raise TypeError(type(indexkey).__name__)
    @retrieve.register(int)
    def integer(cls, index): return {int(valuation): valuation for valuation in iter(cls)}[index]
    @retrieve.register(str)
    def string(cls, string): return {int(valuation): valuation for valuation in iter(cls)}[str(string).lower()]

    class Arbitrage:
        Current = CurrentArbitrage
        Minimum = MinimumArbitrage
        Maximum = MaximumArbitrage

class Valuations(object, metaclass=ValuationsMeta): pass
class ValuationQuery(ntuple("Query", "current ticker expire strategy valuation data")): pass


class ValuationCalculation(Calculation):
    inc = equation("inc", "income", np.float32, domain=("Λ.vo", "Λ.vτ"), function=lambda vo, vτ: + np.maximum(vo, 0) + np.maximum(vτ, 0))
    exp = equation("exp", "cost", np.float32, domain=("Λ.vo", "Λ.vτ"), function=lambda vo, vτ: + np.minimum(vo, 0) + np.minimum(vτ, 0))
    apy = equation("apy", "apy", np.float32, domain=("r", "Λ.τ"), function=lambda r, τ: np.power(r + 1, np.power(τ / 365, -1)) - 1)
    npv = equation("npv", "npv", np.float32, domain=("π", "Λ.τ", "ρ"), function=lambda π, τ, ρ: π * np.power(ρ / 365 + 1, τ))
    π = equation("π", "profit", np.float32, domain=("inc", "exp"), function=lambda inc, exp: inc + exp)
    r = equation("r", "return", np.float32, domain=("π", "exp"), function=lambda π, exp: π / - exp)
    Λ = source("Λ", "valuation", position=0, variables={"τ": "tau", "x": "size", "wo": "spot", "wτ": "future", "τ": "tau", "x": "size", "sμ": "underlying"})
    ρ = constant("ρ", "discount", position="discount")

    def execute(self, feed, *args, discount, **kwargs):
        yield self.inc(feed, discount=discount)
        yield self.exp(feed, discount=discount)
        yield self.npv(feed, discount=discount)
        yield self.apy(feed, discount=discount)
        yield self["Λ"].τ(feed)
        yield self["Λ"].x(feed)
        yield self["Λ"].wo(feed)
        yield self["Λ"].wτ(feed)
        yield self["Λ"].sμ(feed)


class ArbitrageCalculation(ValuationCalculation):
    Λ = source("Λ", "arbitrage", position=0, variables={"vo": "spot", "vτ": "future"})

class MinimumCalculation(ArbitrageCalculation):
    Λ = source("Λ", "minimum", position=0, variables={"vτ": "future"})

class MaximumCalculation(ArbitrageCalculation):
    Λ = source("Λ", "maximum", position=0, variables={"vτ": "future"})


class CalculationsMeta(type):
    def __iter__(cls):
        contents = {Valuations.Arbitrage.Minimum: MinimumCalculation, Valuations.Arbitrage.Maximum: MaximumCalculation}
        return ((key, value) for key, value in contents.items())

    class Arbitrage:
        Minimum = MinimumCalculation
        Maximum = MaximumCalculation

class Calculations(object, metaclass=CalculationsMeta):
    pass


class ValuationCalculator(Calculator, calculations=ODict(list(iter(Calculations)))):
    def execute(self, query, *args, **kwargs):
        assert isinstance(query.data, xr.Dataset)
        for valuation, calculation in self.calculations.items():
            dataset = calculation(query.data, *args, **kwargs)
            yield ValuationQuery(query.current, query.ticker, query.expire, query.strategy, valuation, dataset)


class ValuationParser(Processor):
    def execute(self, query, *args, **kwargs):
        assert isinstance(query.data, xr.Dataset)
        dataframe = self.parser(query.data, *args, strategy=query.strategy, valuation=query.valuation, **kwargs)
        yield ValuationQuery(query.current, query.ticker, query.expire, query.strategy, query.valuation, dataframe)

    @staticmethod
    def parser(dataset, *args, strategy, valuation, **kwargs):
        dataframe = dataset.to_dataframe()
        dataframe["strategy"] = str(strategy)
        dataframe["valuation"] = str(valuation)
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        return dataframe


class ValuationFilter(Processor):
    def execute(self, query, *args, **kwargs):
        assert isinstance(query.data, pd.DataFrame)
        dataframe = self.filter(query.data, *args, **kwargs)
        if dataframe.empty:
            return
        yield ValuationQuery(query.current, query.ticker, query.expire, query.strategy, query.valuation, dataframe)

    @staticmethod
    def filter(dataframe, *args, apy=None, **kwargs):
        dataframe = dataframe.where(dataframe["apy"] >= apy) if apy is not None else dataframe
        dataframe = dataframe.dropna(axis=0, how="all")
        return dataframe


class ValuationSaver(Saver):
    def execute(self, query, *args, **kwargs):
        assert isinstance(query.data, pd.DataFrame)
        if query.data.empty:
            return
        current_folder = os.path.join(self.repository, str(query.current.strftime("%Y%m%d_%H%M%S")))
        with self.locks[current_folder]:
            if not os.path.isdir(current_folder):
                os.mkdir(current_folder)
        ticker_expire_filename = "_".join([str(query.ticker), str(query.expire.strftime("%Y%m%d"))]) + ".csv"
        ticker_expire_file = os.path.join(current_folder, ticker_expire_filename)
        with self.locks[ticker_expire_file]:
            self.write(query.data, file=ticker_expire_file, mode="a")


class ValuationLoader(Loader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Columns = ntuple("Columns", "datetypes datatypes")
        datatypes = {"ticker": str, "volume": np.float32, "size": np.float32, "interest": np.int32}
        datatypes.update({"spot": np.float32, "future": np.float32, "tau": np.int16, "income": np.float32, "cost": np.float32, "npv": np.float32, "apy": np.float32})
        self.columns = Columns(["date", "expire"], datatypes)

    def execute(self, *args, tickers=None, expires=None, dates=None, **kwargs):
        TickerExpire = ntuple("TickerExpire", "ticker expire")
        for current_name in os.listdir(self.repository):
            current = Datetime.strptime(os.path.splitext(current_name)[0], "%Y%m%d_%H%M%S")
            if dates is not None and current.date() not in dates:
                continue
            current_folder = os.path.join(self.repository, current_name)
            for ticker_expire_filename in os.listdir(current_folder):
                ticker_expire = TickerExpire(*str(ticker_expire_filename).split(".")[0].split("_"))
                ticker = str(ticker_expire.ticker).upper()
                if tickers is not None and ticker not in tickers:
                    continue
                expire = Datetime.strptime(os.path.splitext(ticker_expire.expire)[0], "%Y%m%d").date()
                if expires is not None and expire not in expires:
                    continue
                ticker_expire_file = os.path.join(current_folder, ticker_expire_filename)
                with self.locks[ticker_expire_file]:
                    dataframe = self.read(file=ticker_expire_file, filetype=pd.DataFrame, datatypes=self.columns.datatypes, datetypes=self.columns.datetypes)
                    strategies = list(set(dataframe["strategy"].values))
                    valuations = list(set(dataframe["valuation"].values))
                    yield ValuationQuery(current, ticker, expire, strategies, valuations, dataframe)


class ValuationAnalysis(Processor):
    def __init__(self, *args, **kwargs):
        self.__dataframe = pd.DataFrame()

    def execute(self, query, *args, **kwargs):
        dataframe = query.data

        print(dataframe)
        raise Exception()

#        dataframe = pd.concat([self.dataframe, dataframe[self.columns]], axis=0)
#        dataframe = dataframe.sort_values(["apy", "npv"], axis=0, ascending=False, inplace=False, ignore_index=True)
#        self.dataframe = dataframe

    @property
    def weights(self): return (self.dataframe["cost"] / self.dataframe["cost"].sum()) * (self.dataframe["size"] / self.dataframe["size"].sum())
    @property
    def tau(self): return self.dataframe["tau"].min(), self.dataframe["tau"].max()
    @property
    def apy(self): return self.dataframe["apy"] @ self.weights
    @property
    def cost(self): return self.dataframe["cost"] @ self.dataframe["size"]
    @property
    def npv(self): return self.dataframe["npv"] @ self.dataframe["size"]

    @property
    def dataframe(self): return self.__dataframe
    @property
    def dataframe(self, dataframe): self.__dataframe = dataframe




