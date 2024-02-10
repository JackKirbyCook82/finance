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

from finance.variables import Query, Contract, Securities, Valuations, Basis

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationQuery", "ValuationCalculation", "ValuationLoader", "ValuationSaver", "ValuationFilter", "ValuationCalculator", "ValuationFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)


class ValuationCalculation(Calculation):
    Λ = source("Λ", "valuation", position=0, variables={"to": "date", "tτ": "expire", "qo": "size", "wo": "spot"})
    ρ = constant("ρ", "discount", position="discount")

    tau = equation("τau", "tau", np.int32, domain=("Λ.to", "Λ.tτ"), function=lambda to, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))
    inc = equation("inc", "income", np.float32, domain=("Λ.vo", "Λ.vτ"), function=lambda vo, vτ: + np.maximum(vo, 0) + np.maximum(vτ, 0))
    exp = equation("exp", "cost", np.float32, domain=("Λ.vo", "Λ.vτ"), function=lambda vo, vτ: - np.minimum(vo, 0) - np.minimum(vτ, 0))
    apy = equation("apy", "apy", np.float32, domain=("τau", "r"), function=lambda τ, r: np.power(r + 1, np.power(τ / 365, -1)) - 1)
    npv = equation("npv", "npv", np.float32, domain=("τau", "π", "ρ"), function=lambda τ, π, ρ: π * np.power(ρ / 365 + 1, τ))
    π = equation("π", "profit", np.float32, domain=("inc", "exp"), function=lambda inc, exp: inc - exp)
    r = equation("r", "return", np.float32, domain=("π", "exp"), function=lambda π, exp: π / exp)

    def execute(self, feed, *args, discount, **kwargs):
        yield self.tau(feed, discount=discount)
        yield self.exp(feed, discount=discount)
        yield self.npv(feed, discount=discount)
        yield self.apy(feed, discount=discount)
        yield self["Λ"].qo(feed)

class ArbitrageCalculation(ValuationCalculation):
    Λ = source("Λ", "arbitrage", position=0, variables={"vs": "entry", "vo": "spot", "vτ": "future"})

class MinimumArbitrageCalculation(ArbitrageCalculation):
    Λ = source("Λ", "minimum", position=0, variables={"vτ": "minimum"})

class MaximumArbitrageCalculation(ArbitrageCalculation):
    Λ = source("Λ", "maximum", position=0, variables={"vτ": "maximum"})

class MartingaleArbitrageCalculation(ArbitrageCalculation):
    Λ = source("Λ", "martingale", position=0, variables={"vτ": "martingale"})


class ValuationQuery(Query): pass
class ValuationCalculator(Processor, title="Calculated"):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        arbitrage = {Valuations.Arbitrage.Minimum: MinimumArbitrageCalculation, Valuations.Arbitrage.Maximum: MaximumArbitrageCalculation, Valuations.Arbitrage.Martingale: MartingaleArbitrageCalculation}
        arbitrage = {valuation: calculation(*args, **kwargs) for valuation, calculation in arbitrage.items()}
        self.calculations = {Basis.ARBITRAGE: arbitrage}

    def execute(self, query, *args, **kwargs):
        strategies = query.strategies
        for basis, calculations in self.calculations.items():
            valuations = {valuation: calculation(strategies, *args, **kwargs) for valuation, calculation in calculations.items()}
            for valuation, dataset in valuations.items():
                dataset.expand_dims({"valuation": str(valuation)})
            valuations = {valuation: dataset.to_dataframe() for valuation, dataset in valuations.items()}
            valuations = {valuation: dataset.reset_index(drop=False, inplace=False) for valuation, dataset in valuations.items()}
            valuations = self.calculation(valuations, *args, basis=basis, **kwargs)
            if bool(valuations.empty):
                continue
            yield ValuationQuery(query.inquiry, query.contract, basis, valuations)

    @kwargsdispatcher("basis")
    def calculation(self, *args, basis, **kwargs): raise ValueError(str(basis.name).lower())

    @calculation.register(Basis.ARBITRAGE)
    def arbitrage(self, valuations, *args, **kwargs):
        mask = valuations[Valuations.Arbitrage.Minimum]["apy"] > 0
        valuations = {valuation: dataframe[mask] for valuation, dataframe in valuations.items()}
        valuations = {valuation: dataframe.dropna(axis=0, how="all") for valuation, dataframe in valuations.items()}
        valuations = {valuation: dataframe.reset_index(drop=True, inplace=False) for valuation, dataframe in valuations.items()}
        valuations = pd.concat(list(valuations.values()), axis=0).reset_index(drop=True, inplace=False)
        return valuations


class ValuationFilter(Processor, title="Filtered"):
    def execute(self, query, *args, size=None, apy=None, **kwargs):
        criteria = {key: value for key, value in dict(size=size, apy=apy) if value is not None}
        valuations = self.filter(query.valuations, *args, basis=query.basis, criteria=criteria, **kwargs)
        query = ValuationQuery(query.inquiry, query.contract, query.basis, valuations)
        LOGGER.info(f"Filter: {repr(self)}[{str(query)}]")
        yield query

    @kwargsdispatcher("basis")
    def filter(self, *args, basis, **kwargs): raise ValueError(str(basis.name).lower())

    @filter.register(Basis.ARBITRAGE)
    def arbitrage(self, valuations, *args, criteria={}, **kwargs):
        valuations = {valuation: dataframe for valuation, dataframe in iter(valuations.groupby("valuation"))}
        for key, value in criteria.items():
            mask = valuations[Valuations.Arbitrage.Minimum][key] > value
            valuations = {valuation: dataframe[mask] for valuation, dataframe in valuations.items()}
            valuations = {valuation: dataframe.dropna(axis=0, how="all") for valuation, dataframe in valuations.items()}
            valuations = {valuation: dataframe.reset_index(drop=True, inplace=False) for valuation, dataframe in valuations.items()}
        valuations = pd.concat(list(valuations.values()), axis=0).reset_index(drop=True, inplace=False)
        return valuations


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
        basis_name = str(query.basis.name).lower()
        basis_file = self.file.path(inquiry_name, contract_name, basis_name + ".csv")
        self.file.write(valuations, file=basis_file, mode="w")
        LOGGER.info("Saved: {}[{}]".format(repr(self), str(basis_file)))


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
                basis_filenames = list(self.file.directory(contract_name))
                for basis_filename in basis_filenames:
                    basis_name = str(basis_filename).split(".")[0]
                    basis = Basis[str(basis_name)]
                    basis_file = self.file.path(inquiry_name, contract_name, basis_name + ".csv")
                    valuations = self.file.read(file=basis_file)
                    yield ValuationQuery(inquiry, contract, basis, valuations)


class ValuationFile(DataframeFile):
    @staticmethod
    def dataheader(*args, **kwargs): return ["strategy", "valuation", "ticker", "expire", "date"] + list(map(str, Securities.Options)) + ["apy", "npv", "cost", "tau", "size"]
    @staticmethod
    def datatypes(*args, **kwargs): return {"apy": np.float32, "npv": np.float32, "cost": np.float32, "size": np.int32, "tau": np.int32}
    @staticmethod
    def datetypes(*args, **kwargs): return ["expire", "date"]



