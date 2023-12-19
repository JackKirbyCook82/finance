# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Valuation Objects
@author: Jack Kirby Cook

"""

import os
import logging
import numpy as np
import pandas as pd
from enum import IntEnum
from datetime import datetime as Datetime
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.pipelines import Processor, Calculator, Saver, Loader
from support.calculations import Calculation, equation, source, constant
from support.dispatchers import typedispatcher, kwargsdispatcher

from finance.securities import Securities

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Valuation", "Valuations", "Calculations"]
__all__ += ["ValuationSaver", "ValuationLoader", "ValuationFilter", "ValuationCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


Basis = IntEnum("Basis", ["ARBITRAGE"], start=1)
Scenario = IntEnum("Scenario", ["CURRENT", "MINIMUM", "MAXIMUM"], start=1)
class Valuation(ntuple("Valuation", "basis scenario")):
    def __str__(self): return "|".join([str(value.name).lower() for value in self if bool(value)])
    def __int__(self): return int(self.basis) * 10 + int(self.scenario) * 1

    @property
    def title(self): return "|".join([str(string).title() for string in str(self).split("|")])

MinimumArbitrage = Valuation(Basis.ARBITRAGE, Scenario.MINIMUM)
MaximumArbitrage = Valuation(Basis.ARBITRAGE, Scenario.MAXIMUM)
CurrentArbitrage = Valuation(Basis.ARBITRAGE, Scenario.CURRENT)


class ValuationsMeta(type):
    def __iter__(cls): return iter([MinimumArbitrage])
    def __getitem__(cls, indexkey): return cls.retrieve(indexkey)

    @typedispatcher
    def retrieve(cls, indexkey): raise TypeError(type(indexkey).__name__)
    @retrieve.register(int)
    def integer(cls, index): return {int(valuation): valuation for valuation in iter(cls)}[index]
    @retrieve.register(str)
    def string(cls, string): return {int(valuation): valuation for valuation in iter(cls)}[str(string).lower()]

    class Arbitrage:
        Minimum = MinimumArbitrage
        Maximum = MaximumArbitrage
        Current = CurrentArbitrage

class Valuations(object, metaclass=ValuationsMeta):
    pass


class ValuationCalculation(Calculation):
    Λ = source("Λ", "valuation", position=0, variables={"τ": "tau", "wo": "spot", "wτ": "future", "sμ": "underlying", "x": "size"})
    ρ = constant("ρ", "discount", position="discount")

    inc = equation("inc", "income", np.float32, domain=("Λ.vo", "Λ.vτ"), function=lambda vo, vτ: + np.maximum(vo, 0) + np.maximum(vτ, 0))
    exp = equation("exp", "cost", np.float32, domain=("Λ.vo", "Λ.vτ"), function=lambda vo, vτ: - np.minimum(vo, 0) - np.minimum(vτ, 0))
    apy = equation("apy", "apy", np.float32, domain=("r", "Λ.τ"), function=lambda r, τ: np.power(r + 1, np.power(τ / 365, -1)) - 1)
    npv = equation("npv", "npv", np.float32, domain=("π", "Λ.τ", "ρ"), function=lambda π, τ, ρ: π * np.power(ρ / 365 + 1, τ))
    π = equation("π", "profit", np.float32, domain=("inc", "exp"), function=lambda inc, exp: inc - exp)
    r = equation("r", "return", np.float32, domain=("π", "exp"), function=lambda π, exp: π / exp)

    def execute(self, feed, *args, discount, **kwargs):
        yield self.exp(feed, discount=discount)
        yield self.npv(feed, discount=discount)
        yield self.apy(feed, discount=discount)
        yield self["Λ"].τ(feed)
        yield self["Λ"].x(feed)

class ArbitrageCalculation(ValuationCalculation):
    Λ = source("Λ", "arbitrage", position=0, variables={"vo": "spot", "vτ": "future"})

class CurrentCalculation(ArbitrageCalculation):
    Λ = source("Λ", "current", position=0, variables={"vτ": "future"})

class MinimumCalculation(ArbitrageCalculation):
    Λ = source("Λ", "minimum", position=0, variables={"vτ": "future"})

class MaximumCalculation(ArbitrageCalculation):
    Λ = source("Λ", "maximum", position=0, variables={"vτ": "future"})


class CalculationsMeta(type):
    def __iter__(cls):
        contents = {Valuations.Arbitrage.Minimum: MinimumCalculation}
        return ((key, value) for key, value in contents.items())

    class Arbitrage:
        Minimum = MinimumCalculation
        Maximum = MaximumCalculation
        Current = CurrentCalculation

class Calculations(object, metaclass=CalculationsMeta):
    pass


class ValuationQuery(ntuple("Query", "current ticker expire securities valuations")): pass
class ValuationCalculator(Calculator, calculations=ODict(list(Calculations))):
    def execute(self, query, *args, **kwargs):
        securities = query.securities if hasattr(query, "securities") else {}
        strategies = {strategy: dataset for strategy, dataset in query.strategies.items()}
        calculations = {valuation: calculation for valuation, calculation in self.calculations.items()}
        parser = lambda strategy, dataset: self.parser(dataset, *args, strategy=strategy, **kwargs)
        valuations = {valuation: {strategy: calculation(dataset, *args, **kwargs) for strategy, dataset in strategies.items()} for valuation, calculation in calculations.items()}
        valuations = {valuation: [parser(strategy, dataset) for strategy, dataset in strategies.items()] for valuation, strategies in valuations.items()}
        valuations = {valuation: pd.concat(dataframes, axis=0) for valuation, dataframes in valuations.items()}
        if not bool(valuations):
            return
        yield ValuationQuery(query.current, query.ticker, query.expire, securities, valuations)

    @staticmethod
    def parser(dataset, *args, strategy, **kwargs):
        dataframe = dataset.to_dataframe()
        dataframe["strategy"] = str(strategy)
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        return dataframe


class ValuationFilter(Processor):
    def execute(self, query, *args, **kwargs):
        securities = query.securities if hasattr(query, "securities") else {}
        valuations = {valuation: self.filter(dataframe, *args, **kwargs) for valuation, dataframe in query.valuations.items()}
        strings = {str(valuation.title): str(len(dataframe.index)) for valuation, dataframe in valuations.items()}
        string = ", ".join(["=".join([key, value]) for key, value in strings.items()])
        LOGGER.info("Filtered: {}[{}]".format(repr(self), string))
        yield ValuationQuery(query.current, query.ticker, query.expire, securities, valuations)

    @staticmethod
    def filter(dataframe, *args, apy=None, **kwargs):
        dataframe = dataframe.where(dataframe["apy"] >= apy) if apy is not None else dataframe
        dataframe = dataframe.dropna(axis=0, how="all")
        return dataframe


class ValuationFile(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        datetypes = {str(Valuations.Arbitrage.Minimum): ["date", "expire"], str(Valuations.Arbitrage.Maximum): ["date", "expire"], str(Valuations.Arbitrage.Current): ["date", "expire"]}
        datetypes.update({str(Securities.Stock.Long): ["date"], str(Securities.Stock.Short): ["date"]})
        datetypes.update({str(Securities.Option.Put.Long): ["date", "expire"], str(Securities.Option.Put.Short): ["date", "expire"]})
        datetypes.update({str(Securities.Option.Call.Long): ["date", "expire"], str(Securities.Option.Call.Short): ["date", "expire"]})
        stocks = {"ticker": str, "price": np.float32, "size": np.int64}
        options = {"ticker": str, "strike": np.float32, "price": np.float32, "size": np.int64}
        valuation = {"ticker": str, "npv": np.float32, "apy": np.float32, "cost": np.float32, "size": np.int64, "tau": np.int16}
        datatypes = {str(Valuations.Arbitrage.Minimum): valuation, str(Valuations.Arbitrage.Maximum): valuation, str(Valuations.Arbitrage.Current): valuation}
        datatypes.update({str(Securities.Stock.Long): stocks, str(Securities.Stock.Short): stocks})
        datatypes.update({str(Securities.Option.Put.Long): options, str(Securities.Option.Put.Short): options})
        datatypes.update({str(Securities.Option.Call.Long): options, str(Securities.Option.Call.Short): options})
        self.__datetypes = datetypes
        self.__datatypes = datatypes

    @property
    def datetypes(self): return self.__datetypes
    @property
    def datatypes(self): return self.__datatypes


class ValuationSaver(ValuationFile, Saver):
    def execute(self, query, *args, **kwargs):
        valuations = query.valuations
        if not bool(valuations) or not bool([dataframe.empty for dataframe in valuations.values()]):
            return
        current_folder = os.path.join(self.repository, str(query.current.strftime("%Y%m%d_%H%M%S")))
        assert os.path.isdir(current_folder)
        ticker_expire_name = "_".join([str(query.ticker), str(query.expire.strftime("%Y%m%d"))])
        ticker_expire_folder = os.path.join(current_folder, ticker_expire_name)
        assert os.path.isdir(ticker_expire_folder)
        with self.locks[ticker_expire_folder]:
            for valuation, dataframe in valuations.items():
                valuation_filename = str(valuation).replace("|", "_") + ".csv"
                valuation_file = os.path.join(ticker_expire_folder, valuation_filename)
                self.write(dataframe, file=valuation_file, mode="w")


class ValuationLoader(ValuationFile, Loader):
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
                    filenames = {security: str(security).replace("|", "_") + ".csv" for security in list(Securities)}
                    files = {security: os.path.join(ticker_expire_folder, filename) for security, filename in filenames.items()}
                    securities = {security: reader(security, file) for security, file in files.items()}
                    yield ValuationQuery(current, ticker, expire, securities, valuations)




