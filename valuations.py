# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Valuation Objects
@author: Jack Kirby Cook

"""

import os
import logging
import numpy as np
import xarray as xr
from datetime import datetime as Datetime

from support.files import DataframeFile
from support.pipelines import Producer, Processor, Consumer
from support.calculations import Calculation, equation, source, constant

from finance.variables import Query, Contract, Securities, Valuations, Scenarios

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationCalculation", "ValuationFilter", "ValuationParser", "ValuationCalculator", "ValuationLoader", "ValuationSaver", "ValuationFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


valuations_variables = {"to": "date", "tτ": "expire", "qo": "size"}
# {"ws": "cashflow", "Δo": "quantity"}

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
        yield self.npv(feed, discount=discount)
        yield self.apy(feed, discount=discount)
        yield self.exp(feed, discount=discount)
        yield self.τ(feed, discount=discount)
        yield self["v"].qo(feed)

#        yield self["v"].ws(feed)
#        yield self["v"].Δo(feed)


class ArbitrageCalculation(ValuationCalculation):
    v = source("v", "arbitrage", position=0, variables={"vo": "spot", "vτ": "scenario"})

class MinimumArbitrageCalculation(ArbitrageCalculation):
    v = source("v", "minimum", position=0, variables={"vτ": "minimum"})

class MaximumArbitrageCalculation(ArbitrageCalculation):
    v = source("v", "maximum", position=0, variables={"vτ": "maximum"})

# class MartingaleArbitrageCalculation(ArbitrageCalculation):
#     v = source("v", "martingale", position=0, variables={"vτ": "martingale"})


class ValuationQuery(Query, fields=["valuation", "valuations"]): pass
class ValuationCalculator(Processor, title="Calculated"):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        arbitrage = {Scenarios.MINIMUM: MinimumArbitrageCalculation, Scenarios.MAXIMUM: MaximumArbitrageCalculation}
        arbitrage = {valuation: calculation(*args, **kwargs) for valuation, calculation in arbitrage.items()}
        self.calculations = {Valuations.ARBITRAGE: arbitrage}

    def execute(self, query, *args, **kwargs):
        strategies = query.strategies
        assert isinstance(strategies, xr.Dataset)
        for valuation, calculations in self.calculations.items():
            valuations = {scenario: calculation(strategies, *args, **kwargs) for scenario, calculation in calculations.items()}
            valuations = [dataset.assign_coords({"scenario": str(scenario.name).lower()}).expand_dims("scenario") for scenario, dataset in valuations.items()]
            valuations = xr.concat(valuations, dim="scenario").assign_coords({"valuation": str(valuation.name).lower()}).expand_dims("valuation")
            yield ValuationQuery(query.inquiry, query.contract, valuation=valuation, valuations=valuations)


class ValuationFilter(Processor, title="Filtered"):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.filters = {Valuations.ARBITRAGE: Scenarios.MINIMUM}

    def execute(self, query, *args, **kwargs):
        valuation, valuations = query.valuation, query.valuations
        assert isinstance(valuations, xr.Dataset)
        mask = self.mask(valuations, *args, scenario=self.filters[valuation], **kwargs)
        valuations = valuations.where(mask, drop=False)
        query = query(valuation=valuation, valuations=valuations)
        __logger__.info(f"Filter: {repr(self)}[{str(query)}]")
        yield query

    @staticmethod
    def mask(dataset, *args, scenario, size=None, apy=None, **kwargs):
        scenario = str(scenario.name).lower()
        mask = dataset.sel({"scenario": scenario}).notnull() & dataset.sel({"scenario": scenario}).notnull()
        mask = (mask & (dataset.sel({"scenario": scenario})["size"] >= size)) if size is not None else mask
        mask = (mask & (dataset.sel({"scenario": scenario})["apy"] >= apy)) if apy is not None else mask
        return mask


class ValuationParser(Processor, title="Parsed"):
    def execute(self, query, *args, **kwargs):
        valuation, valuations = query.valuation, query.valuations
        assert isinstance(valuations, xr.Dataset)
        valuations = valuations.to_dataframe()
        valuations = valuations.dropna(axis=0, how="all")
        valuations = valuations.reset_index(drop=False, inplace=False)
        query = query(valuation=valuation, valuations=valuations)
        yield query


class ValuationSaver(Consumer, title="Saved"):
    def __init__(self, *args, file, **kwargs):
        assert isinstance(file, ValuationFile)
        super().__init__(*args, **kwargs)
        self.file = file

    def execute(self, query, *args, **kwargs):
        valuation, valuations = query.valuation, query.valuations
        inquiry_name = str(query.inquiry.strftime("%Y%m%d_%H%M%S"))
        contract_name = "_".join([str(query.contract.ticker), str(query.contract.expire.strftime("%Y%m%d"))])
        inquiry_folder = self.file.path(inquiry_name)
        contract_folder = self.file.path(inquiry_folder, contract_name)
        valuation_file = self.file.path(inquiry_name, contract_name, "valuations.csv")
        if valuations is not None and not valuations.empty:
            if not os.path.isdir(inquiry_folder):
                os.mkdir(inquiry_folder)
            if not os.path.isdir(contract_folder):
                os.mkdir(contract_folder)
            self.file.write(valuations, file=valuation_file, filemode="w")
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
                valuation_file = self.file.path(inquiry_name, contract_name, "valuations.csv")
                valuations = self.file.read(file=valuation_file)
                yield ValuationQuery(inquiry, contract, valuations)


class ValuationFile(DataframeFile):
    @staticmethod
    def dataheader(*args, **kwargs): return ["strategy", "valuation", "scenario", "ticker", "expire", "date"] + list(map(str, Securities.Options)) + ["apy", "npv", "cost", "tau", "size"]
    @staticmethod
    def datatypes(*args, **kwargs): return {"apy": np.float32, "npv": np.float32, "cost": np.float32, "size": np.int32, "tau": np.int32}
    @staticmethod
    def datetypes(*args, **kwargs): return ["expire", "date"]



