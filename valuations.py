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
from collections import OrderedDict as ODict

from support.calculations import Calculation, equation, source, constant
from support.processes import Calculator, Filter
from support.pipelines import Processor
from support.dispatchers import typedispatcher
from support.files import DataframeFile

from finance.variables import Securities, Valuations, Scenarios

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationCalculation", "ValuationCalculator", "ValuationFilter", "ValuationFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


INDEX = {option: str for option in list(map(str, Securities.Options))} | {"strategy": str, "valuation": str, "scenario": str, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
VALUATION = {"apy": np.float32, "npv": np.float32, "cost": np.float32, "size": np.float32}


class ValuationFile(DataframeFile, name="valuation", index=INDEX, columns=VALUATION): pass
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
        yield self.npv(feed, discount=discount)
        yield self.apy(feed)
        yield self.exp(feed)
        yield self["v"].qo(feed)


class ArbitrageCalculation(ValuationCalculation, valuation=Valuations.ARBITRAGE):
    v = source("v", "arbitrage", position=0, variables={"vo": "spot"})

class MinimumArbitrageCalculation(ArbitrageCalculation, scenario=Scenarios.MINIMUM):
    v = source("v", "minimum", position=0, variables={"vτ": "minimum"})

class MaximumArbitrageCalculation(ArbitrageCalculation, scenario=Scenarios.MAXIMUM):
    v = source("v", "maximum", position=0, variables={"vτ": "maximum"})


class ValuationCalculator(Calculator, Processor, calculations=ODict(list(ValuationCalculation)), title="Calculated"):
    def __init__(self, *args, valuation, **kwargs):
        super().__init__(*args, **kwargs)
        self.__valuation = valuation

    def execute(self, query, *args, **kwargs):
        strategies = query["strategy"]
        valuation = str(self.valuation.name).lower()
        assert isinstance(strategies, xr.Dataset)
        calculations = {variable.scenario: calculation for variable, calculation in self.calculations.items()}
        valuations = {scenario: calculation(strategies, *args, **kwargs) for scenario, calculation in calculations.items()}
        valuations = [dataset.assign_coords({"scenario": str(scenario.name).lower()}).expand_dims("scenario") for scenario, dataset in valuations.items()]
        valuations = xr.concat(valuations, dim="scenario").assign_coords({"valuation": valuation})
        valuations = self.parser(valuations, *args, **kwargs)
        yield query | dict(valuation=valuations)

    @staticmethod
    def parser(dataset, *args, **kwargs):
        dataset = dataset.expand_dims(["valuation", "ticker", "expire", "date"])
        dataset = dataset.drop_vars(["instrument", "position"], errors="ignore")
        dataframe = dataset.to_dataframe()
        dataframe = dataframe.dropna(how="all", inplace=False)
        return dataframe

    @property
    def valuation(self): return self.__valuation


class ValuationFilter(Filter, Processor, title="Filtered"):
    def __init__(self, *args, scenario, **kwargs):
        super().__init__(*args, **kwargs)
        self.__scenario = scenario

    def execute(self, query, *args, **kwargs):
        contract = query["contract"]
        valuations = query["valuation"]
        scenario = str(self.scenario.name).lower()
        selections = dict(scenario=scenario)
        prior = self.size(valuations["size"], *args, **kwargs)

        dataframe = valuations
        dataset = xr.Dataset.from_dataframe(valuations)

        dataframe = self.filter(dataframe, *args, selections=selections, **kwargs)
        dataset = self.filter(dataset, *args, selections=selections, **kwargs)

        if dataframe.empty:
            return

        raise Exception()

        post = self.size(valuations["size"], *args, **kwargs)
        __logger__.info(f"Filter: {repr(self)}|{str(contract)}[{prior:.0f}|{post:.0f}]")
        yield query | dict(valuation=valuations)

    @property
    def scenario(self): return self.__scenario



