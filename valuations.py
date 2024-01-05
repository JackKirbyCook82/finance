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
from collections import OrderedDict as ODict
from collections import namedtuple as ntuple

from support.calculations import Calculation, equation, source, constant
from support.dispatchers import typedispatcher
from support.pipelines import Processor, Reader, Writer
from support.files import DataframeFile

from finance.securities import Securities

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Valuation", "Valuations", "Calculations", "ValuationFile", "ValuationReader", "ValuationWriter", "ValuationFilter", "ValuationCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)
Basis = IntEnum("Basis", ["ARBITRAGE"], start=1)
Scenarios = IntEnum("Scenarios", ["CURRENT", "MINIMUM", "MAXIMUM"], start=1)


class Valuation(ntuple("Valuation", "basis scenario")):
    def __str__(self): return "|".join([str(value.name).lower() for value in self if bool(value)])
    def __int__(self): return int(self.basis) * 10 + int(self.scenario) * 1

    @property
    def title(self): return "|".join([str(string).title() for string in str(self).split("|")])

MinimumArbitrage = Valuation(Basis.ARBITRAGE, Scenarios.MINIMUM)
MaximumArbitrage = Valuation(Basis.ARBITRAGE, Scenarios.MAXIMUM)
CurrentArbitrage = Valuation(Basis.ARBITRAGE, Scenarios.CURRENT)


class ValuationsMeta(type):
    def __iter__(cls): return iter([MinimumArbitrage])
    def __getitem__(cls, indexkey): return cls.retrieve(indexkey)

    @typedispatcher
    def retrieve(cls, indexkey): raise TypeError(type(indexkey).__name__)
    @retrieve.register(int)
    def integer(cls, index): return {int(valuation): valuation for valuation in iter(cls)}[index]
    @retrieve.register(str)
    def string(cls, string): return {int(valuation): valuation for valuation in iter(cls)}[str(string).lower()]

    @property
    def Arbitrages(cls): return iter([MinimumArbitrage, MaximumArbitrage, CurrentArbitrage])

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
        contents = {Valuations.Arbitrage.Minimum: MinimumCalculation, Valuations.Arbitrage.Maximum: MaximumCalculation}
        contents.update({Valuations.Arbitrage.Current: CurrentCalculation})
        return ((key, value) for key, value in contents.items())

    class Arbitrage:
        Minimum = MinimumCalculation
        Maximum = MaximumCalculation
        Current = CurrentCalculation

class Calculations(object, metaclass=CalculationsMeta):
    pass


class ValuationQuery(ntuple("Query", "current ticker expire valuations")):
    def __str__(self):
        strings = {str(valuation.title): str(len(dataframe.index)) for valuation, dataframe in self.valuations.items()}
        arguments = "{}|{}".format(self.ticker, self.expire.strftime("%Y-%m-%d"))
        parameters = ", ".join(["=".join([key, value]) for key, value in strings.items()])
        return ", ".join([arguments, parameters]) if bool(parameters) else str(arguments)


class ValuationCalculator(Processor):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        valuations = kwargs.get("calculations", ODict(list(Calculations)).keys())
        calculations = ODict([(valuation, calculation(*args, **kwargs)) for valuation, calculation in iter(Calculations) if valuation in valuations])
        self.__calculations = calculations

    def execute(self, query, *args, **kwargs):
        strategies = {strategy: dataset for strategy, dataset in query.strategies.items()}
        if not bool(strategies):
            return
        calculations = {valuation: calculation for valuation, calculation in self.calculations.items()}
        parser = lambda strategy, dataset: self.parser(dataset, *args, strategy=strategy, **kwargs)
        valuations = {valuation: {strategy: calculation(dataset, *args, **kwargs) for strategy, dataset in strategies.items()} for valuation, calculation in calculations.items()}
        valuations = {valuation: [parser(strategy, dataset) for strategy, dataset in strategies.items()] for valuation, strategies in valuations.items()}
        valuations = {valuation: pd.concat(dataframes, axis=0) for valuation, dataframes in valuations.items()}
        if not bool(valuations):
            return
        yield ValuationQuery(query.current, query.ticker, query.expire, valuations)

    @staticmethod
    def parser(dataset, *args, strategy, **kwargs):
        dataframe = dataset.to_dataframe()
        dataframe["strategy"] = str(strategy)
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        return dataframe

    @property
    def calculations(self): return self.__calculations


class ValuationFilter(Processor):
    def execute(self, query, *args, **kwargs):
        valuations = {valuation: dataframe for valuation, dataframe in query.valuations.items()}
        valuations = {valuation: self.filter(dataframe, *args, **kwargs) for valuation, dataframe in valuations.items()}
        query = ValuationQuery(query.current, query.ticker, query.expire, valuations)
        LOGGER.info("Filter: {}[{}]".format(repr(self), str(query)))
        yield query

    @staticmethod
    def filter(dataframe, *args, size=None, apy=None, **kwargs):
        dataframe = dataframe.where(dataframe["size"] >= size) if size is not None else dataframe
        dataframe = dataframe.where(dataframe["apy"] >= apy) if apy is not None else dataframe
        dataframe = dataframe.dropna(axis=0, how="all")
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe


class ValuationFile(DataframeFile):
    @staticmethod
    def dataheader(*args, **kwargs): return ["strategy", "ticker", "date", "expire"] + list(map(str, Securities.Options)) + ["apy", "npv", "cost", "tau", "size"]
    @staticmethod
    def datatypes(*args, **kwargs): return {"apy": np.float32, "npv": np.float32, "cost": np.float32, "size": np.int32, "tau": np.int32}
    @staticmethod
    def datetypes(*args, **kwargs): return ["date", "expire"]


class ValuationWriter(Writer):
    def execute(self, query, *args, **kwargs):
        valuations = query.valuations
        if not bool(valuations) or all([dataframe.empty for dataframe in valuations.values()]):
            return
        current_name = str(query.current.strftime("%Y%m%d_%H%M%S"))
        current_folder = self.destination.path(current_name)
        assert os.path.isdir(current_folder)
        ticker_expire_name = "_".join([str(query.ticker), str(query.expire.strftime("%Y%m%d"))])
        ticker_expire_folder = self.destination.path(current_name, ticker_expire_name)
        assert os.path.isdir(ticker_expire_folder)
        for valuation, dataframe in valuations.items():
            valuation_name = str(valuation).replace("|", "_") + ".csv"
            valuation_file = self.destination.path(current_name, ticker_expire_name, valuation_name)
            self.destination.write(dataframe, file=valuation_file, data=valuation, mode="w")
            LOGGER.info("Saved: {}[{}]".format(repr(self), str(valuation_file)))


class ValuationReader(Reader):
    def execute(self, *args, tickers=None, expires=None, dates=None, **kwargs):
        TickerExpire = ntuple("TickerExpire", "ticker expire")
        function = lambda foldername: Datetime.strptime(foldername, "%Y%m%d_%H%M%S")
        current_folders = list(self.source.directory())
        for current_name in sorted(current_folders, key=function, reverse=False):
            current = function(current_name)
            if dates is not None and current.date() not in dates:
                continue
            ticker_expire_folders = list(self.source.directory(current_name))
            for ticker_expire_name in ticker_expire_folders:
                ticker_expire = TickerExpire(*str(ticker_expire_name).split("_"))
                ticker = str(ticker_expire.ticker).upper()
                if tickers is not None and ticker not in tickers:
                    continue
                expire = Datetime.strptime(os.path.splitext(ticker_expire.expire)[0], "%Y%m%d").date()
                if expires is not None and expire not in expires:
                    continue

                filenames = {valuation: str(valuation).replace("|", "_") + ".csv" for valuation in list(Valuations)}
                files = {valuation: self.source.path(current_name, ticker_expire_name, filename) for valuation, filename in filenames.items()}
                valuations = {valuation: self.source.read(file=file, data=valuation) for valuation, file in files.items() if os.path.isfile(file)}
                yield ValuationQuery(current, ticker, expire, valuations)




