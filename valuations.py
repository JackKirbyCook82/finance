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
import xarray as xr
from datetime import datetime as Datetime
from collections import namedtuple as ntuple

from support.files import DataframeFile
from support.dispatchers import typedispatcher, kwargsdispatcher
from support.pipelines import Producer, Processor, Consumer
from support.calculations import Calculation, equation, source, constant

from finance.variables import Query, Contract, Securities, Valuations, Scenarios

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationCalculation", "ValuationFilter", "ValuationParsing", "ValuationParser", "ValuationCalculator", "ValuationLoader", "ValuationSaver", "ValuationFile"]
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


class ArbitrageCalculation(ValuationCalculation):
    v = source("v", "arbitrage", position=0, variables={"vo": "spot", "vτ": "scenario"})

class MinimumArbitrageCalculation(ArbitrageCalculation):
    v = source("v", "minimum", position=0, variables={"vτ": "minimum"})

class MaximumArbitrageCalculation(ArbitrageCalculation):
    v = source("v", "maximum", position=0, variables={"vτ": "maximum"})


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
            yield query(valuation=valuation, valuations=valuations)


class ValuationFilter(Processor, title="Filtered"):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.valuations = {Valuations.ARBITRAGE: Scenarios.MINIMUM}

    def execute(self, query, *args, **kwargs):
        length = lambda dataset: np.count_nonzero(~np.isnan(valuations["size"].values))
        valuation, valuations = query.valuation, query.valuations
        prior = length(valuations)
        valuations = self.filter(valuations, *args, scenario=self.valuations[valuation], **kwargs)
        post = length(valuations)
        __logger__.info(f"Filter: {repr(self)}|{str(query)}[{prior:.0f}|{post:.0f}]")
        yield query(valuations=valuations)

    @typedispatcher
    def filter(self, content, *args, **kwargs): raise TypeError(type(content).__name__)
    @typedispatcher
    def mask(self, content, *args, **kwargs): raise TypeError(type(content).__name__)

    @filter.register(xr.Dataset)
    def filter_dataset(self, dataset, *args, scenario, **kwargs):
        scenario = str(scenario.name).lower()
        mask = dataset.sel({"scenario": scenario})
        mask = self.mask(mask, *args, **kwargs)
        dataset = dataset.where(mask, drop=False)
        return dataset

    @filter.register(pd.DataFrame)
    def filter_dataframe(self, dataframe, *args, scenario, **kwargs):
        options = [option for option in list(map(str, Securities.Options)) if option in dataframe.columns]
        index = ["strategy", "valuation", "scenario", "ticker", "expire", "date"] + options
        dataframe = dataframe.drop_duplicates(subset=index, keep="last", inplace=False)
        dataframes = {key: value.set_index(index, inplace=False, drop=True) for key, value in iter(dataframe.groupby("scenario"))}
        scenario = str(scenario.name).lower()
        mask = dataframes[scenario]
        mask = self.mask(mask, *args, **kwargs)
        dataframes = [value.where(mask) for key, value in dataframes.values()]
        dataframes = [value.dropna(axis=0, how="all") for key, value in dataframes]
        dataframes = [value.reset_index(drop=False, inplace=False) for key, value in dataframes]
        dataframe = pd.concat(dataframes, axis=0)
        return dataframe

    @mask.register(xr.Dataset)
    def mask_dataset(self, dataset, *args, size=None, apy=None, **kwargs):
        mask = dataset["size"].notnull() & dataset["apy"].notnull()
        mask = (mask & (dataset["size"] >= size)) if size is not None else mask
        mask = (mask & (dataset["apy"] >= apy)) if apy is not None else mask
        return mask

    @mask.register(pd.DataFrame)
    def mask_dataframe(self, dataframe, *args, size=None, apy=None, **kwargs):
        mask = dataframe["size"].notna() & dataframe["apy"].notna()
        mask = (mask & (dataframe["size"] >= size)) if size is not None else mask
        mask = (mask & (dataframe["apy"] >= apy)) if apy is not None else mask
        return mask


class ValuationParsing(object):
    Parsing = ntuple("Parsing", "source destination")
    UNFLATTEN = Parsing(pd.DataFrame, xr.Dataset)
    FLATTEN = Parsing(xr.Dataset, pd.DataFrame)


class ValuationParser(Processor, title="Parsed"):
    def __init__(self, *args, name, parsing, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.parsing = parsing

    def execute(self, query, *args, **kwargs):
        valuations = query.valuations
        if not isinstance(valuations, self.parsing.source):
            raise TypeError(type(valuations).__name__)
        valuations = self.parse(valuations, *args, parsing=self.parsing.destination, **kwargs)
        yield query(valuations=valuations)

    @kwargsdispatcher("parsing")
    def parse(self, content, *args, parsing, **kwargs): raise TypeError(type(parsing).__name__)

    @parse.register.value(pd.DataFrame)
    def dataframe(self, dataset, *args, **kwarg):
        dataframe = dataset.to_dataframe()
        dataframe = dataframe.dropna(axis=0, how="all")
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        return dataframe

    @parse.register.value(xr.Dataset)
    def dataset(self, dataframe, *args, **kwargs):
        options = [option for option in list(map(str, Securities.Options)) if option in dataframe.columns]
        index = ["strategy", "valuation", "scenario", "ticker", "expire", "date"] + options
        dataframe = dataframe.set_index(index, inplace=False, drop=True)
        dataset = xr.Dataset.from_dataframe(dataframe)
        return dataset


class ValuationSaver(Consumer, title="Saved"):
    def __init__(self, *args, file, valuations=[Valuations.ARBITRAGE], **kwargs):
        assert isinstance(file, ValuationFile)
        super().__init__(*args, **kwargs)
        self.valuations = valuations
        self.file = file

    def execute(self, query, *args, **kwargs):
        valuation, valuations = query.valuation, query.valuations
        if valuation not in self.valuations:
            return
        inquiry_name = str(query.inquiry.strftime("%Y%m%d_%H%M%S"))
        contract_name = "_".join([str(query.contract.ticker), str(query.contract.expire.strftime("%Y%m%d"))])
        valuation_name = str(valuation.name).lower() + ".csv"
        inquiry_folder = self.file.path(inquiry_name)
        contract_folder = self.file.path(inquiry_folder, contract_name)
        valuation_file = self.file.path(inquiry_name, contract_name, valuation_name)
        if valuations is not None and not valuations.empty:
            if not os.path.isdir(inquiry_folder):
                os.mkdir(inquiry_folder)
            if not os.path.isdir(contract_folder):
                os.mkdir(contract_folder)
            self.file.write(valuations, file=valuation_file, filemode="w")
            __logger__.info("Saved: {}[{}]".format(repr(self), str(valuation_file)))


class ValuationLoader(Producer, title="Loaded"):
    def __init__(self, *args, file, valuations=[Valuations.ARBITRAGE], **kwargs):
        assert isinstance(file, ValuationFile)
        super().__init__(*args, **kwargs)
        self.valuations = valuations
        self.file = file

    def execute(self, *args, tickers=None, expires=None, **kwargs):
        function = lambda foldername: Datetime.strptime(foldername, "%Y%m%d_%H%M%S")
        inquiry_folders = list(self.file.directory())
        for inquiry_name in sorted(inquiry_folders, key=function, reverse=False):
            inquiry = function(inquiry_name)
            contract_names = list(self.file.directory(inquiry_name))
            for contract_name in contract_names:
                ticker, expire = str(os.path.splitext(contract_name)[0]).split("_")
                ticker = str(ticker).upper()
                if tickers is not None and ticker not in tickers:
                    continue
                expire = Datetime.strptime(expire, "%Y%m%d").date()
                if expires is not None and expire not in expires:
                    continue
                contract = Contract(ticker, expire)
                valuation_names = {valuation: str(valuation.name).lower() + ".csv" for valuation in self.valuations}
                valuation_files = {valuation: self.file.path(inquiry_name, contract_name, valuation_name) for valuation, valuation_name in valuation_names.items()}
                valuations = {valuation: self.file.read(file=valuation_file) for valuation, valuation_file in valuation_files.items() if os.path.isfile(valuation_file)}
                for valuation, dataframe in valuations.items():
                    yield Query(inquiry, contract, valuation=valuation, valuations=dataframe)


class ValuationFile(DataframeFile):
    @staticmethod
    def dataheader(*args, **kwargs): return ["strategy", "valuation", "scenario", "ticker", "expire", "date"] + list(map(str, Securities.Options)) + ["apy", "npv", "cost", "tau", "size"]
    @staticmethod
    def datatypes(*args, **kwargs): return {"apy": np.float32, "npv": np.float32, "cost": np.float32, "size": np.int32, "tau": np.int32}
    @staticmethod
    def datetypes(*args, **kwargs): return ["expire", "date"]



