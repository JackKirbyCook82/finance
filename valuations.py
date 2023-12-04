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

from finance.securities import Securities
from finance.strategies import Strategies

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Valuation", "Valuations", "Calculations"]
__all__ += ["ValuationProcessor", "ValuationCalculator", "ValuationAnalysis", "ValuationSaver", "ValuationLoader"]
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

class Valuations(object, metaclass=ValuationsMeta):
    pass


class ValuationCalculation(Calculation):
    pα = source("pα", str(Securities.Option.Put.Long), position=0, variables={"w": "price", "k": "strike"})
    pβ = source("pβ", str(Securities.Option.Put.Short), position=0, variables={"w": "price", "k": "strike"})
    cα = source("cα", str(Securities.Option.Call.Long), position=0, variables={"w": "price", "k": "strike"})
    cβ = source("cβ", str(Securities.Option.Call.Short), position=0, variables={"w": "price", "k": "strike"})
    Λ = source("Λ", "valuation", position=0, variables={"wo": "spot", "wτ": "future", "τ": "tau", "q": "size", "i": "interest"}, fullname=False)
    ρ = constant("ρ", "discount", position="discount")

    inc = equation("inc", "income", np.float32, domain=("Λ.vo", "Λ.vτ"), function=lambda vo, vτ: + np.maximum(vo, 0) + np.maximum(vτ, 0))
    exp = equation("exp", "cost", np.float32, domain=("Λ.vo", "Λ.vτ"), function=lambda vo, vτ: - np.minimum(vo, 0) - np.minimum(vτ, 0))
    apy = equation("apy", "apy", np.float32, domain=("r", "Λ.τ"), function=lambda r, τ: np.power(r + 1, np.power(τ / 365, -1)) - 1)
    npv = equation("npv", "npv", np.float32, domain=("π", "Λ.τ", "ρ"), function=lambda π, τ, ρ: π * np.power(ρ / 365 + 1, τ))
    π = equation("π", "profit", np.float32, domain=("inc", "exp"), function=lambda inc, exp: inc - exp)
    r = equation("r", "return", np.float32, domain=("π", "exp"), function=lambda π, exp: π / exp)

    def execute(self, *args, feed, discount, **kwargs):
        yield self.inc(feed, discount=discount)
        yield self.exp(feed, discount=discount)
        yield self.npv(feed, discount=discount)
        yield self.apy(feed, discount=discount)
        yield self["Λ"].wo(feed)
        yield self["Λ"].wτ(feed)
        yield self["Λ"].τ(feed)
        yield self["pα"].k(feed)
        yield self["pβ"].k(feed)
        yield self["cα"].k(feed)
        yield self["cβ"].k(feed)
        yield self["pα"].w(feed)
        yield self["pβ"].w(feed)
        yield self["cα"].w(feed)
        yield self["cβ"].w(feed)

class ArbitrageCalculation(ValuationCalculation):
    Λ = source("Λ", "arbitrage", position=0, variables={"vo": "spot", "vτ": "future"}, fullname=False)

class MinimumCalculation(ArbitrageCalculation):
    Λ = source("Λ", "minimum", position=0, variables={"vτ": "future"}, fullname=False)

class MaximumCalculation(ArbitrageCalculation):
    Λ = source("Λ", "maximum", position=0, variables={"vτ": "future"}, fullname=False)

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
    def execute(self, contents, *args, **kwargs):
        current, ticker, expire, strategy, dataset = contents
        assert isinstance(dataset, xr.Dataset)
        for valuation, calculation in self.calculations.items():
            results = calculation(*args, feed=dataset, **kwargs)
            yield current, ticker, expire, strategy, valuation, results


class ValuationProcessor(Processor):
    def execute(self, contents, *args, **kwargs):
        current, ticker, expire, strategy, valuation, dataset = contents
        assert isinstance(dataset, xr.Dataset)
        dataframe = self.parser(dataset, *args, **kwargs)
        dataframe = self.filter(dataframe, *args, **kwargs)
        assert isinstance(dataframe, pd.DataFrame)
        if dataframe.empty:
            return
        yield current, ticker, expire, strategy, valuation, dataframe

    @staticmethod
    def parser(dataset, *args, **kwargs):
        dataframe = dataset.to_dataframe()
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        return dataframe

    @staticmethod
    def filter(dataframe, *args, apy=None, cost=None, size=None, interest=None, volume=None, **kwargs):
        dataframe = dataframe.where(dataframe["apy"] >= apy) if apy is not None else dataframe
        dataframe = dataframe.where(dataframe["cost"] <= cost) if cost is not None else dataframe
        dataframe = dataframe.where(dataframe["size"] >= size) if size is not None else dataframe
        dataframe = dataframe.where(dataframe["interest"] >= interest) if interest is not None else dataframe
        dataframe = dataframe.where(dataframe["volume"] >= volume) if volume is not None else dataframe
        dataframe = dataframe.dropna(axis=0, how="all")
        return dataframe


class ValuationSaver(Saver):
    def execute(self, contents, *args, **kwargs):
        current, ticker, expire, strategy, valuation, dataframe = contents
        assert isinstance(dataframe, pd.DataFrame)
        current_folder = os.path.join(self.repository, str(current.strftime("%Y%m%d_%H%M%S")))
        with self.locks[current_folder]:
            if not os.path.isdir(current_folder):
                os.mkdir(current_folder)
        valuation_folder = os.path.join(current_folder, str(valuation).replace("|", "_"))
        with self.locks[valuation_folder]:
            if not os.path.isdir(valuation_folder):
                os.mkdir(valuation_folder)
        strategy_folder = os.path.join(valuation_folder, str(strategy).replace("|", "_"))
        with self.locks[strategy_folder]:
            if not os.path.isdir(strategy_folder):
                os.mkdir(strategy_folder)
        ticker_expire_filename = "_".join([str(ticker), str(expire.strftime("%Y%m%d"))]) + ".csv"
        ticker_expire_file = os.path.join(strategy_folder, ticker_expire_filename)
        with self.locks[ticker_expire_file]:
            self.write(dataframe, file=ticker_expire_file, mode="a")


class ValuationLoader(Loader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Columns = ntuple("Columns", "datetypes datatypes")
        datetypes = ["date", "expire"]
        datatypes = {"ticker": str, "tau": np.int16}
        datatypes.update({"spot": np.float32, "minimum": np.float32, "maximum": np.float32, "current": np.float32})
        datatypes.update({"income": np.float32, "cost": np.float32, "npv": np.float32, "apy": np.float32})
        self.columns = Columns(datetypes, datatypes)

    def execute(self, *args, tickers, expires, **kwargs):
        TickerExpire = ntuple("TickerExpire", "ticker expire")
        for current_name in os.listdir(self.repository):
            current = Datetime.strptime(os.path.splitext(current_name)[0], "%Y%m%d_%H%M%S")
            current_folder = os.path.join(self.repository, current_name)
            for valuation_name in os.listdir(current_folder):
                valuation = Valuations[str(valuation_name).replace("_", "|")]
                valuation_folder = os.path.join(current_folder, valuation_name)
                for strategy_name in os.listdir(valuation_folder):
                    strategy = Strategies[str(strategy_name).replace("_", "|")]
                    strategy_folder = os.path.join(valuation_folder, strategy_name)
                    for ticker_expire_filename in os.listdir(strategy_folder):
                        ticker_expire = TickerExpire(*str(ticker_expire_filename).split(".")[0].split("_"))
                        ticker = str(ticker_expire.ticker).upper()
                        if ticker not in tickers and tickers is not None:
                            continue
                        expire = Datetime.strptime(os.path.splitext(ticker_expire.expire)[0], "%Y%m%d").date()
                        if expire not in expires and expires is not None:
                            continue
                        ticker_expire_file = os.path.join(strategy_folder, ticker_expire_filename)
                        with self.locks[ticker_expire_file]:
                            dataframe = self.read(file=ticker_expire_file, filetype=pd.DataFrame, datatypes=self.columns.datatypes, datetypes=self.columns.datetypes)
                            yield current, ticker, expire, valuation, strategy, dataframe


class ValuationWrapper(object):
    def __init__(self, dataframe, column):
        assert isinstance(dataframe, pd.DataFrame)
        self.__dataframe = dataframe[["strategy", "valuation", "ticker", "tau", "cost", "apy"]]
        self.__column = column

    def __eq__(self, value): return self.dataframe.where(self.dataframe[self.column] == value)
    def __ne__(self, value): return self.dataframe.where(self.dataframe[self.column] != value)
    def __ge__(self, value): return self.dataframe.where(self.dataframe[self.column] >= value)
    def __le__(self, value): return self.dataframe.where(self.dataframe[self.column] <= value)
    def __gt__(self, value): return self.dataframe.where(self.dataframe[self.column] > value)
    def __lt__(self, value): return self.dataframe.where(self.dataframe[self.column] < value)

    @property
    def dataframe(self): return self.__dataframe
    @property
    def column(self): return self.__column


class ValuationResults(object):
    def __bool__(self): return not self.dataframe.empty
    def __init__(self, dataframe=None):
        assert isinstance(dataframe, (pd.DataFrame, type(None)))
        columns = ["strategy", "valuation", "ticker", "tau", "cost", "apy"]
        self.__dataframe = dataframe[columns] if dataframe is not None else pd.DataFrame(columns=columns)
        self.__columns = columns

    def __call__(self, dataframe):
        assert isinstance(dataframe, (pd.DataFrame, type(None)))
        dataframe = dataframe[self.columns]
        self.dataframe = pd.concat([self.dataframe, dataframe], axis=0)

    def __getitem__(self, column):
        return ValuationWrapper(self.dataframe, column)

    @property
    def columns(self): return self.__columns
    @property
    def dataframe(self): return self.__dataframe
    @dataframe.setter
    def dataframe(self, dataframe): self.__dataframe = dataframe


class ValuationAnalysis(Processor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__results = ValuationResults()

    def execute(self, contents, *args, **kwargs):
        current, ticker, expire, strategy, valuation, dataframe = contents
        assert isinstance(dataframe, pd.DataFrame)
        dataframe["strategy"] = str(strategy)
        dataframe["valuation"] = str(valuation)
        self.results(dataframe)

    @property
    def results(self): return self.__results



