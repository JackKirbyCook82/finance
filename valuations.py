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

from support.files import DataframeFile
from support.dispatchers import kwargsdispatcher
from support.pipelines import Producer, Processor, Consumer
from support.calculations import Calculation, equation, source, constant

from finance.variables import Query, Contract, Securities, Valuations

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationCalculation", "ValuationFilter", "ValuationCalculator", "ValuationLoader", "ValuationSaver", "ValuationFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


valuations_variables = {"to": "date", "tτ": "expire", "qo": "size", "Δo": "quantity"}
class ValuationCalculation(Calculation):
    v = source("v", "valuation", position=0, variables=valuations_variables)
    ρ = constant("ρ", "discount", position="discount")

    τ = equation("τ", "tau", np.int32, domain=("v.to", "v.tτ"), function=lambda to, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))
    π = equation("π", "profit", np.float32, domain=("inc", "exp"), function=lambda inc, exp: inc - exp)
    r = equation("r", "return", np.float32, domain=("π", "exp"), function=lambda π, exp: π / exp)
    inc = equation("inc", "income", np.float32, domain=("v.vo", "v.vτ"), function=lambda vo, vτ: + np.maximum(vo, 0) + np.maximum(vτ, 0))
    exp = equation("exp", "cost", np.float32, domain=("v.vo", "v.vτ"), function=lambda vo, vτ: - np.minimum(vo, 0) - np.minimum(vτ, 0))
    apy = equation("apy", "apy", np.float32, domain=("τ", "r"), function=lambda τ, r: np.power(r + 1, np.power(τ / 365, -1)) - 1)
    npv = equation("npv", "npv", np.float32, domain=("τ", "π", "ρ"), function=lambda τ, π, ρ: π * np.power(ρ / 365 + 1, τ))

    def execute(self, feed, *args, discount, **kwargs):
        yield self.tau(feed, discount=discount)
        yield self.exp(feed, discount=discount)
        yield self.npv(feed, discount=discount)
        yield self.apy(feed, discount=discount)
        yield self["v"].qo(feed)
        yield self["v"].Δo(feed)


class ArbitrageCalculation(ValuationCalculation):
    v = source("v", "arbitrage", position=0, variables={"vo": "reference", "vτ": "future"})

class SpotCalculation(ValuationCalculation):
    v = source("v", "spot", position=0, variables={"vo": "spot"})

class EntryCalculation(ValuationCalculation):
    v = source("v", "entry", position=0, variables={"vo": "entry"})

class MinimumArbitrageCalculation(ArbitrageCalculation):
    v = source("v", "minimum", position=0, variables={"vτ": "minimum"})

class MaximumArbitrageCalculation(ArbitrageCalculation):
    v = source("v", "maximum", position=0, variables={"vτ": "maximum"})

class MartingaleArbitrageCalculation(ArbitrageCalculation):
    v = source("v", "martingale", position=0, variables={"vτ": "martingale"})


class ValuationQuery(Query, fields=["valuation", "valuations"]): pass
class ValuationCalculator(Processor, title="Calculated"):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        arbitrage = {Valuations.Arbitrage.Minimum: MinimumArbitrageCalculation, Valuations.Arbitrage.Maximum: MaximumArbitrageCalculation, Valuations.Arbitrage.Martingale: MartingaleArbitrageCalculation}
        arbitrage = {valuation: calculation(*args, **kwargs) for valuation, calculation in arbitrage.items()}
        self.calculations = {Valuations.ARBITRAGE: arbitrage}

    def execute(self, query, *args, **kwargs):
        strategies = query.strategies
        for valuation, calculations in self.calculations.items():
            valuations = {scenario: calculation(strategies, *args, **kwargs) for scenario, calculation in calculations.items()}
            for scenario, dataset in valuations.items():
                dataset.expand_dims({"scenario": str(scenario)})
            valuations = {scenario: dataset.to_dataframe() for scenario, dataset in valuations.items()}
            valuations = {scenario: dataset.reset_index(drop=False, inplace=False) for scenario, dataset in valuations.items()}
            valuations = self.calculate(valuations, *args, valuation=valuation, **kwargs)
            valuations["valuations"] = str(valuations)
            if bool(valuations.empty):
                continue
            yield ValuationQuery(query.inquiry, query.contract, valuation, valuations)

    @kwargsdispatcher("valuation")
    def calculate(self, *args, valuation, **kwargs): raise ValueError(str(valuation.name).lower())

    @calculate.register(Valuations.ARBITRAGE)
    def arbitrage(self, dataframes, *args, **kwargs):
        mask = dataframes[Valuations.Arbitrage.Minimum]["apy"] > 0
        dataframes = [dataframe[mask].dropna(axis=0, how="all") for valuation, dataframe in dataframes.values()]
        dataframe = pd.concat(dataframes, axis=0).reset_index(drop=True, inplace=False)
        return dataframe


class ValuationFilter(Processor, title="Filtered"):
    def execute(self, query, *args, **kwargs):
        valuations = query.valuations
        valuations = {valuation: self.filter(dataframe, *args, valuation=valuation, **kwargs) for valuation, dataframe in valuations.items()}
        query = query(valuations=valuations)
        __logger__.info(f"Filter: {repr(self)}[{str(query)}]")
        yield query

    @kwargsdispatcher("valuation")
    def filter(self, *args, valuation, **kwargs): raise ValueError(str(valuation.name).lower())

    @filter.register(Valuations.ARBITRAGE)
    def arbitrage(self, dataframes, *args, **kwargs):
        dataframes = {scenario: dataframe for scenario, dataframe in iter(dataframes.groupby("scenario"))}
        mask = self.mask(dataframes[str(Scenarios.MINIMUM)], *args, **kwargs)
        dataframes = [dataframe.where(mask).dropna(axis=0, how="all") for dataframe in dataframes.values()]
        dataframe = pd.concat(dataframes, axis=0).reset_index(drop=True, inplace=False)
        return dataframe

    @staticmethod
    def mask(dataframe, *args, size=None, apy=None, **kwargs):
        mask = (dataframe["size"].notna() & dataframe["apy"].notna())
        mask = (mask & dataframe["size"] >= size) if size is not None else mask
        mask = (mask & dataframe["apy"] >= apy) if apy is not None else mask
        return mask


class ValuationSaver(Consumer, title="Saved"):
    def __init__(self, *args, file, **kwargs):
        assert isinstance(file, ValuationFile)
        super().__init__(*args, **kwargs)
        self.file = file

    def execute(self, query, *args, **kwargs):
        valuations = query.valuations
        inquiry_name = str(query.inquiry.strftime("%Y%m%d_%H%M%S"))
        inquiry_folder = self.file.path(inquiry_name)
        if not os.path.isdir(inquiry_folder):
            os.mkdir(inquiry_folder)
        contract_name = "_".join([str(query.contract.ticker), str(query.contract.expire.strftime("%Y%m%d"))])
        contract_folder = self.file.path(contract_name)
        if not os.path.isdir(contract_folder):
            os.mkdir(contract_folder)
        for valuation in valuations:
            valuation_name = str(valuation.name).lower()
            valuation_file = self.file.path(inquiry_name, contract_name, valuation_name + ".csv")
            self.file.write(valuations, file=valuation_file, mode="w")
            __logger__.info("Saved: {}[{}]".format(repr(self), str(valuation_file)))


class ValuationLoader(Producer, title="Loaded"):
    def __init__(self, *args, file, **kwargs):
        assert isinstance(file, ValuationFile)
        super().__init__(*args, **kwargs)
        self.file = file

    def execute(self, *args, tickers=None, expires=None, **kwargs):
        function = lambda foldername: Datetime.strptime(foldername, "%Y%m%d_%H%M%S")
        inquiry_folders = list(self.file.directory())
        for inquiry_name in sorted(inquiry_folders, key=function, reverse=False):
            inquiry = function(inquiry_name)
            contract_filenames = list(self.file.directory(inquiry_name))
            for contract_filename in contract_filenames:
                contract_name = str(contract_filename).split(".")[0]
                contract = Contract(*str(contract_filename).split("_"))
                ticker = str(contract.ticker).upper()
                if tickers is not None and ticker not in tickers:
                    continue
                expire = Datetime.strptime(os.path.splitext(contract.expire)[0], "%Y%m%d").date()
                if expires is not None and expire not in expires:
                    continue
                for valuation in list(Valuations):
                    valuation_name = str(valuation.name).lower()
                    valuation_file = self.file.path(inquiry_name, contract_name, valuation_name + ".csv")
                    valuations = self.file.read(file=valuation_file)
                    yield ValuationQuery(inquiry, contract, valuation, valuations)


class ValuationFile(DataframeFile):
    @staticmethod
    def dataheader(*args, **kwargs): return ["strategy", "valuation", "scenario", "ticker", "expire", "date"] + list(map(str, Securities.Options)) + ["apy", "npv", "cost", "tau", "size"]
    @staticmethod
    def datatypes(*args, **kwargs): return {"apy": np.float32, "npv": np.float32, "cost": np.float32, "size": np.int32, "tau": np.int32}
    @staticmethod
    def datetypes(*args, **kwargs): return ["expire", "date"]



