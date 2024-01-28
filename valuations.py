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
from support.dispatchers import kwargsdispatcher, typedispatcher
from support.pipelines import Producer, Processor, Consumer
from support.files import DataframeFile

from finance.securities import Securities

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Valuation", "Valuations", "Calculations", "ValuationFile", "ValuationReader", "ValuationWriter", "ValuationFilter", "ValuationCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)
Basis = IntEnum("Basis", ["ARBITRAGE"], start=1)
Scenarios = IntEnum("Scenarios", ["MINIMUM", "MAXIMUM", "CURRENT"], start=1)


class Valuation(ntuple("Valuation", "basis scenario")):
    def __str__(self): return "|".join([str(value.name).lower() for value in self if bool(value)])
    def __int__(self): return int(self.basis) * 10 + int(self.scenario) * 1

    @property
    def title(self): return "|".join([str(string).title() for string in str(self).split("|")])

MinimumArbitrage = Valuation(Basis.ARBITRAGE, Scenarios.MINIMUM)
MaximumArbitrage = Valuation(Basis.ARBITRAGE, Scenarios.MAXIMUM)
CurrentArbitrage = Valuation(Basis.ARBITRAGE, Scenarios.CURRENT)


class ValuationsMeta(type):
    def __iter__(cls): return iter([MinimumArbitrage, MaximumArbitrage, CurrentArbitrage])
    def __getitem__(cls, indexkey): return cls.retrieve(indexkey)

    @typedispatcher
    def retrieve(cls, indexkey): raise TypeError(type(indexkey).__name__)
    @retrieve.register(int)
    def integer(cls, index): return {int(valuation): valuation for valuation in iter(cls)}[index]
    @retrieve.register(str)
    def string(cls, string): return {int(valuation): valuation for valuation in iter(cls)}[str(string).lower()]
    @retrieve.register(tuple)
    def value(cls, value): return {(valuation.basis, valuation.scenario): valuation for valuation in iter(cls)}[value]

    @property
    def Arbitrages(cls): return iter([MinimumArbitrage, MaximumArbitrage, CurrentArbitrage])

    class Arbitrage:
        Minimum = MinimumArbitrage
        Maximum = MaximumArbitrage
        Current = CurrentArbitrage

class Valuations(object, metaclass=ValuationsMeta):
    pass


class ValuationCalculation(Calculation):
    Λ = source("Λ", "valuation", position=0, variables={"τ": "tau", "wo": "spot", "sμ": "underlying", "x": "size"})
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

class MinimumArbitrageCalculation(ArbitrageCalculation):
    Λ = source("Λ", "minimum", position=0, variables={"vτ": "minimum"})

class MaximumArbitrageCalculation(ArbitrageCalculation):
    Λ = source("Λ", "maximum", position=0, variables={"vτ": "maximum"})

class CurrentArbitrageCalculation(ArbitrageCalculation):
    Λ = source("Λ", "maximum", position=0, variables={"vτ": "current"})


class CalculationsMeta(type):
    def __iter__(cls):
        contents = {MinimumArbitrage: MinimumArbitrageCalculation, MaximumArbitrage: MaximumArbitrageCalculation, CurrentArbitrage: CurrentArbitrageCalculation}
        return ((key, value) for key, value in contents.items())

    class Arbitrage:
        Minimum = MinimumArbitrageCalculation
        Maximum = MaximumArbitrageCalculation
        Current = CurrentArbitrageCalculation

    @property
    def Arbitrages(cls): return iter({MinimumArbitrage: MinimumArbitrageCalculation, MaximumArbitrage: MaximumArbitrageCalculation, CurrentArbitrage: MaximumArbitrageCalculation}.items())

class Calculations(object, metaclass=CalculationsMeta):
    pass


class ValuationQuery(ntuple("Query", "current ticker expire arbitrages")):
    def __str__(self):
        strings = {str(valuation.title): str(len(dataframe.index)) for valuation, dataframe in self.arbitrages.items()}
        arguments = f"{self.ticker}|{self.expire.strftime('%Y-%m-%d')}"
        parameters = ", ".join(["=".join([key, value]) for key, value in strings.items()])
        return ", ".join([arguments, parameters]) if bool(parameters) else str(arguments)


class ValuationCalculator(Processor):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.calculations = {valuation: calculation(*args, **kwargs) for (valuation, calculation) in iter(Calculations)}

    def execute(self, query, *args, **kwargs):
        strategies = {strategy: dataset for strategy, dataset in query.strategies.items()}
        if not bool(strategies):
            return
        valuations = {valuation: self.calculate(strategies, calculation, *args, **kwargs) for valuation, calculation in self.calculations.items()}
        arbitrages = self.arbitrage({valuation: dataframe for valuation, dataframe in valuations.items() if valuation in list(Valuations.Arbitrages)}, *args, **kwargs)
        if not bool(arbitrages):
            return
        yield ValuationQuery(query.current, query.ticker, query.expire, arbitrages)

    @staticmethod
    def calculate(strategies, calculation, *args, **kwargs):
        strategies = {strategy: calculation(dataset, *args, **kwargs) for strategy, dataset in strategies.items()}
        strategies = {strategy: dataset.to_dataframe() for strategy, dataset in strategies.items()}
        for strategy, dataframe in strategies.items():
            dataframe["strategy"] = str(strategy)
        strategies = [dataframe.reset_index(drop=False, inplace=False) for dataframe in strategies.values()]
        strategies = pd.concat(strategies, axis=0).reset_index(drop=True, inplace=False)
        return strategies

    @staticmethod
    def arbitrage(valuations, *args, **kwargs):
        mask = valuations[MinimumArbitrage]["apy"] > 0
        valuations = {valuation: dataframe[mask] for valuation, dataframe in valuations.items()}
        valuations = {valuation: dataframe.dropna(axis=0, how="all") for valuation, dataframe in valuations.items()}
        valuations = {valuation: dataframe.reset_index(drop=True, inplace=False) for valuation, dataframe in valuations.items()}
        return valuations


class ValuationFilter(Processor):
    def execute(self, query, *args, **kwargs):
        arbitrages = {valuation: dataframe for valuation, dataframe in query.arbitrages.items()}
        arbitrages = self.arbitrage(arbitrages, *args, **kwargs)
        query = ValuationQuery(query.current, query.ticker, query.expire, arbitrages)
        LOGGER.info(f"Filter: {repr(self)}[{str(query)}]")
        yield query

    @staticmethod
    def arbitrage(dataframes, *args, size=None, apy=None, **kwargs):
        criteria = dict(size=size, apy=apy)
        for key, value in criteria.items():
            if value is not None:
                mask = dataframes[MinimumArbitrage][key] > value
                dataframes = {valuation: dataframe[mask] for valuation, dataframe in dataframes.items()}
                dataframes = {valuation: dataframe.dropna(axis=0, how="all") for valuation, dataframe in dataframes.items()}
                dataframes = {valuation: dataframe.reset_index(drop=True, inplace=False) for valuation, dataframe in dataframes.items()}
        return dataframes

class ValuationFile(DataframeFile):
    @staticmethod
    def dataheader(*args, **kwargs): return ["strategy", "ticker", "date", "expire"] + list(map(str, Securities.Options)) + ["apy", "npv", "cost", "tau", "size"]
    @staticmethod
    def datatypes(*args, **kwargs): return {"apy": np.float32, "npv": np.float32, "cost": np.float32, "size": np.int32, "tau": np.int32}
    @staticmethod
    def datetypes(*args, **kwargs): return ["date", "expire"]


class ValuationWriter(Consumer):
    def __init__(self, *args, file, **kwargs):
        assert isinstance(file, ValuationFile)
        super().__init__(*args, **kwargs)
        self.file = file

    def execute(self, query, *args, **kwargs):
        valuations = query.arbitrages
        if not bool(valuations) or all([dataframe.empty for dataframe in valuations.values()]):
            return
        current_name = str(query.current.strftime("%Y%m%d_%H%M%S"))
        current_folder = self.file.path(current_name)
        if not os.path.isdir(current_folder):
            os.mkdir(current_folder)
        ticker_expire_name = "_".join([str(query.ticker), str(query.expire.strftime("%Y%m%d"))])
        ticker_expire_folder = self.file.path(current_name, ticker_expire_name)
        if not os.path.isdir(ticker_expire_folder):
            os.mkdir(ticker_expire_folder)
        for valuation, dataframe in valuations.items():
            valuation_name = str(valuation).replace("|", "_") + ".csv"
            valuation_file = self.file.path(current_name, ticker_expire_name, valuation_name)
            self.file.write(dataframe, file=valuation_file, data=valuation, mode="w")
            LOGGER.info("Saved: {}[{}]".format(repr(self), str(valuation_file)))


class ValuationReader(Producer):
    def __init__(self, *args, file, **kwargs):
        assert isinstance(file, ValuationFile)
        super().__init__(*args, **kwargs)
        self.file = file

    def execute(self, *args, tickers=None, expires=None, dates=None, **kwargs):
        TickerExpire = ntuple("TickerExpire", "ticker expire")
        function = lambda foldername: Datetime.strptime(foldername, "%Y%m%d_%H%M%S")
        current_folders = list(self.file.directory())
        for current_name in sorted(current_folders, key=function, reverse=False):
            current = function(current_name)
            if dates is not None and current.date() not in dates:
                continue
            ticker_expire_folders = list(self.file.directory(current_name))
            for ticker_expire_name in ticker_expire_folders:
                ticker_expire = TickerExpire(*str(ticker_expire_name).split("_"))
                ticker = str(ticker_expire.ticker).upper()
                if tickers is not None and ticker not in tickers:
                    continue
                expire = Datetime.strptime(os.path.splitext(ticker_expire.expire)[0], "%Y%m%d").date()
                if expires is not None and expire not in expires:
                    continue
                filenames = {valuation: str(valuation).replace("|", "_") + ".csv" for valuation in list(Valuations)}
                files = {valuation: self.file.path(current_name, ticker_expire_name, filename) for valuation, filename in filenames.items()}
                valuations = {valuation: self.file.read(file=file, data=valuation) for valuation, file in files.items() if os.path.isfile(file)}
                yield ValuationQuery(current, ticker, expire, valuations)





