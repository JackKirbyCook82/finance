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
from datetime import datetime as Datetime
from collections import namedtuple as ntuple

from support.calculations import Calculation, equation, source, constant
from support.pipelines import Producer, Processor, Consumer
from support.files import DataframeFile

from finance.variables import Contract, Securities, Valuations

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationLoader", "ValuationSaver", "ValuationFilter", "ValuationCalculator", "ValuationFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


class ValuationCalculation(Calculation):
    Λ = source("Λ", "valuation", position=0, variables={"to": "date", "tτ": "expire", "wo": "spot", "x": "size"})
    ρ = constant("ρ", "discount", position="discount")

    tau = equation("τau", "tau", np.int32, domain=("Λ.to", "Λ.tτ"), function=lambda to, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))
    inc = equation("inc", "income", np.float32, domain=("Λ.vo", "Λ.vτ"), function=lambda vo, vτ: + np.maximum(vo, 0) + np.maximum(vτ, 0))
    exp = equation("exp", "cost", np.float32, domain=("Λ.vo", "Λ.vτ"), function=lambda vo, vτ: - np.minimum(vo, 0) - np.minimum(vτ, 0))
    apy = equation("apy", "apy", np.float32, domain=("r", "τau"), function=lambda r, τ: np.power(r + 1, np.power(τ / 365, -1)) - 1)
    npv = equation("npv", "npv", np.float32, domain=("π", "τau", "ρ"), function=lambda π, τ, ρ: π * np.power(ρ / 365 + 1, τ))
    π = equation("π", "profit", np.float32, domain=("inc", "exp"), function=lambda inc, exp: inc - exp)
    r = equation("r", "return", np.float32, domain=("π", "exp"), function=lambda π, exp: π / exp)

    def execute(self, feed, *args, discount, **kwargs):
        yield self.tau(feed, discount=discount)
        yield self.exp(feed, discount=discount)
        yield self.npv(feed, discount=discount)
        yield self.apy(feed, discount=discount)
        yield self["Λ"].x(feed)

class ArbitrageCalculation(ValuationCalculation):
    Λ = source("Λ", "arbitrage", position=0, variables={"vo": "spot", "vτ": "future"})

class MinimumArbitrageCalculation(ArbitrageCalculation):
    Λ = source("Λ", "minimum", position=0, variables={"vτ": "minimum"})

class MaximumArbitrageCalculation(ArbitrageCalculation):
    Λ = source("Λ", "maximum", position=0, variables={"vτ": "maximum"})

class CurrentArbitrageCalculation(ArbitrageCalculation):
    Λ = source("Λ", "maximum", position=0, variables={"vτ": "current"})


class ValuationQuery(ntuple("Query", "inquiry contract arbitrages")):
    def __str__(self):
        strings = {str(valuation.title): str(len(dataframe.index)) for valuation, dataframe in self.arbitrages.items()}
        arguments = f"{self.contract.ticker}|{self.contract.expire.strftime('%Y-%m-%d')}"
        parameters = ", ".join(["=".join([key, value]) for key, value in strings.items()])
        return ", ".join([arguments, parameters]) if bool(parameters) else str(arguments)


class ValuationCalculator(Processor, title="Calculated"):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        calculations = {Valuations.Arbitrage.Minimum: MinimumArbitrageCalculation, Valuations.Arbitrage.Maximum: MaximumArbitrageCalculation, Valuations.Arbitrage.Current: CurrentArbitrageCalculation}
        calculations = {security: calculation(*args, **kwargs) for security, calculation in calculations.items()}
        self.calculations = calculations

    def execute(self, query, *args, **kwargs):
        strategies = {strategy: dataset for strategy, dataset in query.strategies.items()}
        if not bool(strategies):
            return
        valuations = {valuation: self.calculate(strategies, calculation, *args, **kwargs) for valuation, calculation in self.calculations.items()}
        arbitrages = self.arbitrage({valuation: dataframe for valuation, dataframe in valuations.items() if valuation in list(Valuations.Arbitrages)}, *args, **kwargs)
        if not bool(arbitrages):
            return
        yield ValuationQuery(query.inquiry, query.contract, arbitrages)

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
        mask = valuations[Valuations.Arbitrage.Minimum]["apy"] > 0
        valuations = {valuation: dataframe[mask] for valuation, dataframe in valuations.items()}
        valuations = {valuation: dataframe.dropna(axis=0, how="all") for valuation, dataframe in valuations.items()}
        valuations = {valuation: dataframe.reset_index(drop=True, inplace=False) for valuation, dataframe in valuations.items()}
        return valuations


class ValuationFilter(Processor, title="Filtered"):
    def execute(self, query, *args, **kwargs):
        arbitrages = {valuation: dataframe for valuation, dataframe in query.arbitrages.items()}
        arbitrages = self.arbitrage(arbitrages, *args, **kwargs)
        query = ValuationQuery(query.inquiry, query.contract, arbitrages)
        LOGGER.info(f"Filter: {repr(self)}[{str(query)}]")
        yield query

    @staticmethod
    def arbitrage(dataframes, *args, size=None, apy=None, **kwargs):
        criteria = dict(size=size, apy=apy)
        for key, value in criteria.items():
            if value is not None:
                mask = dataframes[Valuations.Arbitrage.Minimum][key] > value
                dataframes = {valuation: dataframe[mask] for valuation, dataframe in dataframes.items()}
                dataframes = {valuation: dataframe.dropna(axis=0, how="all") for valuation, dataframe in dataframes.items()}
                dataframes = {valuation: dataframe.reset_index(drop=True, inplace=False) for valuation, dataframe in dataframes.items()}
        return dataframes


class ValuationSaver(Consumer, title="Saved"):
    def __init__(self, *args, file, **kwargs):
        assert isinstance(file, ValuationFile)
        super().__init__(*args, **kwargs)
        self.file = file

    def execute(self, query, *args, **kwargs):
        valuations = query.arbitrages
        if not bool(valuations) or all([dataframe.empty for dataframe in valuations.values()]):
            return
        inquiry_name = str(query.inquiry.strftime("%Y%m%d_%H%M%S"))
        inquiry_folder = self.file.path(inquiry_name)
        if not os.path.isdir(inquiry_folder):
            os.mkdir(inquiry_folder)
        contract_name = "_".join([str(query.contract.ticker), str(query.contract.expire.strftime("%Y%m%d"))])
        contract_folder = self.file.path(inquiry_name, contract_name)
        if not os.path.isdir(contract_folder):
            os.mkdir(contract_folder)
        for valuation, dataframe in valuations.items():
            valuation_name = str(valuation).replace("|", "_") + ".csv"
            valuation_file = self.file.path(inquiry_name, contract_name, valuation_name)
            self.file.write(dataframe, file=valuation_file, data=valuation, mode="w")
            LOGGER.info("Saved: {}[{}]".format(repr(self), str(valuation_file)))


class ValuationLoader(Producer, title="Loaded"):
    def __init__(self, *args, file, **kwargs):
        assert isinstance(file, ValuationFile)
        super().__init__(*args, **kwargs)
        self.file = file

    def execute(self, *args, tickers=None, expires=None, dates=None, **kwargs):
        function = lambda foldername: Datetime.strptime(foldername, "%Y%m%d_%H%M%S")
        inquiry_folders = list(self.file.directory())
        for inquiry_name in sorted(inquiry_folders, key=function, reverse=False):
            inquiry = function(inquiry_name)
            if dates is not None and inquiry.date() not in dates:
                continue
            contract_folders = list(self.file.directory(inquiry_name))
            for contract_name in contract_folders:
                contract = Contract(*str(contract_name).split("_"))
                ticker = str(contract.ticker).upper()
                if tickers is not None and ticker not in tickers:
                    continue
                expire = Datetime.strptime(os.path.splitext(contract.expire)[0], "%Y%m%d").date()
                if expires is not None and expire not in expires:
                    continue
                filenames = {valuation: str(valuation).replace("|", "_") + ".csv" for valuation in list(Valuations)}
                files = {valuation: self.file.path(inquiry_name, contract_name, filename) for valuation, filename in filenames.items()}
                valuations = {valuation: self.file.read(file=file, data=valuation) for valuation, file in files.items() if os.path.isfile(file)}
                contract = Contract(ticker, expire)
                yield ValuationQuery(inquiry, contract, valuations)


class ValuationFile(DataframeFile):
    @staticmethod
    def dataheader(*args, **kwargs): return ["strategy", "date", "ticker", "expire"] + list(map(str, Securities.Options)) + ["apy", "npv", "cost", "tau", "size"]
    @staticmethod
    def datatypes(*args, **kwargs): return {"apy": np.float32, "npv": np.float32, "cost": np.float32, "size": np.int32, "tau": np.int32}
    @staticmethod
    def datetypes(*args, **kwargs): return ["date", "expire"]



