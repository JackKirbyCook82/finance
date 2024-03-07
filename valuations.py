# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Valuation Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime as Datetime
from collections import OrderedDict as ODict

from support.files import DataframeFile, Header
from support.processes import Calculator, Saver, Loader, Filter, Parser, Axes
from support.calculations import Calculation, equation, source, constant

from finance.variables import Query, Contract, Securities, Valuations, Scenarios

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationCalculation", "ValuationCalculator", "ValuationFilter", "ValuationLoader", "ValuationSaver", "ValuationFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


INDEX = {option: str for option in list(map(str, Securities.Options))}
COLUMNS = {"strategy": str, "valuation": str, "scenario": str}
SCOPE = {"ticker": str, "expire": np.datetime64, "date": np.datetime64}
VALUES = {"cost": np.float32, "apy": np.float32, "npv": np.float32, "tau": np.int32, "size": np.int32}
HEADER = Header(INDEX | COLUMNS | SCOPE, VALUES)
AXES = Axes(INDEX, COLUMNS, VALUES, SCOPE)


class ValuationCalculation(Calculation, fields=["valuation", "scenario"]):
    inc = equation("inc", "income", np.float32, domain=("v.vo", "v.vτ"), function=lambda vo, vτ: + np.maximum(vo, 0) + np.maximum(vτ, 0))
    exp = equation("exp", "cost", np.float32, domain=("v.vo", "v.vτ"), function=lambda vo, vτ: - np.minimum(vo, 0) - np.minimum(vτ, 0))
    tau = equation("tau", "tau", np.int32, domain=("v.to", "v.tτ"), function=lambda to, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))
    npv = equation("npv", "npv", np.float32, domain=("inc", "exp", "tau", "ρ"), function=lambda inc, exp, tau, ρ: np.divide(inc, np.power(1 + ρ, tau / 365)) - exp)
    irr = equation("irr", "irr", np.float32, domain=("inc", "exp", "tau"), function=lambda inc, exp, tau: np.power(np.divide(inc, exp), np.power(tau, -1)) - 1)
    apy = equation("apy", "apy", np.float32, domain=("irr", "tau"), function=lambda irr, tau: np.power(irr + 1, np.power(tau / 365, -1)) - 1)
    v = source("v", "valuation", position=0, variables={"to": "date", "tτ": "expire", "qo": "size"})
    ρ = constant("ρ", "discount", position="discount")

    def execute(self, feed, *args, discount, **kwargs):
        yield self["v"].qo(feed)
        yield self.npv(feed, discount=discount)
        yield self.exp(feed)
        yield self.apy(feed)
        yield self.tau(feed)


class ArbitrageCalculation(ValuationCalculation, valuation=Valuations.ARBITRAGE):
    v = source("v", "arbitrage", position=0, variables={"vo": "spot"})

class MinimumArbitrageCalculation(ArbitrageCalculation, scenario=Scenarios.MINIMUM):
    v = source("v", "minimum", position=0, variables={"vτ": "minimum"})

class MaximumArbitrageCalculation(ArbitrageCalculation, scenario=Scenarios.MAXIMUM):
    v = source("v", "maximum", position=0, variables={"vτ": "maximum"})


class ValuationCalculator(Calculator, calculations=ODict(list(ValuationCalculation))):
    def __init__(self, *args, valuation, **kwargs):
        super().__init__(*args, **kwargs)
        self.valuation = valuation

    def execute(self, query, *args, **kwargs):
        strategies = query.strategies
        assert isinstance(strategies, xr.Dataset)
        calculations = {variable.scenario: calculation for variable, calculation in self.calculations.items()}
        valuations = {scenario: calculation(strategies, *args, **kwargs) for scenario, calculation in calculations.items()}
        valuations = [dataset.assign_coords({"scenario": str(scenario.name).lower()}).expand_dims("scenario") for scenario, dataset in valuations.items()]
        valuations = xr.concat(valuations, dim="scenario").assign_coords({"valuation": str(self.valuation.name).lower()})
        yield query(valuations=valuations)


class ValuationFilter(Filter):
    def __init__(self, *args, scenario, **kwargs):
        super().__init__(*args, **kwargs)
        self.scenario = scenario

    def execute(self, query, *args, **kwargs):
        valuations = query.valuations
        assert isinstance(valuations, xr.Dataset)
        prior = self.size(valuations["size"])
        scenario = str(self.scenario.name).lower()
        mask = valuations.sel({"scenario": scenario})
        mask = self.mask(mask, *args, **kwargs)
        mask = xr.broadcast(mask, valuations)[0]
        valuations = self.filter(valuations, *args, mask=mask, **kwargs)
        post = self.size(valuations["size"])
        query = query(valuations=valuations)
        __logger__.info(f"Filter: {repr(self)}|{str(query)}[{prior:.0f}|{post:.0f}]")
        yield query


class ValuationParser(Parser, axes=AXES):
    def execute(self, query, *args, **kwargs):
        valuations = query.valuations
        assert isinstance(valuations, xr.Dataset)
        valuations = self.flatten(valuations, *args, **kwargs)
        valuations = self.clean(valuations, *args, **kwargs)
        query = query(valuations=valuations)
        yield query


class ValuationFile(DataframeFile, header=HEADER): pass
class ValuationSaver(Saver):
    def execute(self, query, *args, **kwargs):
        valuations = query.valuations
        assert isinstance(valuations, pd.DataFrame)
        if bool(valuations.empty):
            return
        ticker = str(query.contract.ticker)
        expire = str(query.contract.expire.strftime("%Y%m%d"))
        valuations = self.parse(valuations, *args, **kwargs)
        folder = "_".join([ticker, expire])
        files = {"valuations.csv": valuations}
        self.write(folder=folder, files=files, mode="w")


class ValuationLoader(Loader):
    def execute(self, *args, **kwargs):
        folders = self.contents(folder=None)
        files = ["valuations.csv"]
        for folder, contents in self.reader(folders=folders, files=files):
            ticker, expire = str(folder).split("_")
            ticker = str(ticker).upper()
            expire = Datetime.strptime(expire, "%Y%m%d")
            contract = Contract(ticker, expire)
            yield Query(contract, **contents)



