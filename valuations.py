# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Valuation Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import xarray as xr
from abc import ABC

from support.files import DataframeFile
from support.pipelines import Processor
from support.processes import Saver, Loader, Filter, Parser
from support.calculations import Calculation, equation, source, constant

from finance.variables import Securities, Valuations, Actions, Scenarios

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationCalculation", "ValuationCalculator", "ValuationFilter", "ValuationParser", "ValuationLoader", "ValuationSaver", "ValuationFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


COLUMNS_VARS = {"apy": np.float32, "npv": np.float32, "tau": np.int32, "size": np.int32, "quantity": np.int32}
INDEX_VARS = {"strategy": str, "valuation": str, "scenario": str} | {option: str for option in list(map(str, Securities.Options))}
SCOPE_VARS = {"ticker": str, "expire": np.datetime64, "date": np.datetime64}
OPEN_VARS = {"to": "date", "tτ": "expire", "qo": "size"}
CLOSE_VARS = {"to": "date", "tτ": "expire", "qo": "size", "Δo": "quantity"}


class ValuationCalculation(Calculation, ABC):
    tau = equation("tau", "tau", np.int32, domain=("v.to", "v.tτ"), function=lambda to, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))
    npv = equation("npv", "npv", np.float32, domain=("v.vo", "v.vτ", "tau", "ρ"), function=lambda vo, vτ, tau, ρ: vo + np.divide(vτ, np.power(1 + ρ, tau / 365)))
    irr = equation("irr", "irr", np.float32, domain=("v.vo", "v.vτ", "tau"), function=lambda vo, vτ, tau: np.power(-np.divide(vτ, vo), np.power(tau, -1)))
    apy = equation("apy", "apy", np.float32, domain=("irr", "tau"), function=lambda irr, tau: np.power(irr + 1, np.power(tau / 365, -1)) - 1)
    ρ = constant("ρ", "discount", position="discount")

    def __init_subclass__(cls, *args, **kwargs):
        valuation = kwargs.get("valuation", getattr(cls, "valuation", None))
        scenario = kwargs.get("scenario", getattr(cls, "scenario", None))
        action = kwargs.get("action", getattr(cls, "action", None))
        if all([valuation is not None, scenario is not None, action is not None]):
            valuations = cls.registry.get(action, {})
            scenarios = valuations.get(valuation, {})
            scenarios.update({scenario: cls})
            valuations.update({valuation: scenarios})
            cls.registry.update({action: valuations})
        if valuation is not None:
            cls.valuation = valuation
        if scenario is not None:
            cls.scenario = scenario
        if action is not None:
            cls.action = action


class OpenValuationCalculation(ValuationCalculation, action=Actions.OPEN):
    v = source("v", "valuation", position=0, variables=OPEN_VARS)

    def execute(self, feed, *args, discount, **kwargs):
        yield self.npv(feed, discount=discount)
        yield self.apy(feed)
        yield self.tau(feed)
        yield self["v"].qo(feed)


class CloseValuationCalculation(ValuationCalculation, action=Actions.CLOSE):
    v = source("v", "valuation", position=0, variables=CLOSE_VARS)

    def execute(self, feed, *args, discount, **kwargs):
        yield self.npv(feed, discount=discount)
        yield self.apy(feed)
        yield self.tau(feed)
        yield self["v"].qo(feed)
        yield self["v"].Δo(feed)


class ArbitrageCalculation(ValuationCalculation, ABC, valuation=Valuations.ARBITRAGE):
    v = source("v", "arbitrage", position=0, variables={"vo": "spot"})

class MinimumArbitrageCalculation(ArbitrageCalculation, ABC, scenario=Scenarios.MINIMUM):
    v = source("v", "minimum", position=0, variables={"vτ": "minimum"})

class MaximumArbitrageCalculation(ArbitrageCalculation, ABC, scenario=Scenarios.MAXIMUM):
    v = source("v", "maximum", position=0, variables={"vτ": "maximum"})


class OpenMinimumCalculation(OpenValuationCalculation, MinimumArbitrageCalculation): pass
class CloseMinimumCalculation(CloseValuationCalculation, MinimumArbitrageCalculation): pass
class OpenMaximumCalculation(OpenValuationCalculation, MaximumArbitrageCalculation): pass
class CloseMaximumCalculation(CloseValuationCalculation, MaximumArbitrageCalculation): pass


class ValuationCalculator(Processor, title="Calculated"):
    def __init__(self, *args, name=None, action, valuation, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        calculations = {scenario: calculation for scenario, calculation in ValuationCalculation[action][valuation].items()}
        calculations = {scenario: calculation(*args, **kwargs) for scenario, calculation in calculations.items()}
        self.calculations = calculations
        self.valuation = valuation
        self.action = action

    def execute(self, query, *args, **kwargs):
        strategies = query.strategies
        assert isinstance(strategies, xr.Dataset)
        valuations = {scenario: calculation(strategies, *args, **kwargs) for scenario, calculation in self.calculations.items()}
        valuations = [dataset.assign_coords({"scenario": str(scenario.name).lower()}).expand_dims("scenario") for scenario, dataset in valuations.items()]
        valuations = xr.concat(valuations, dim="scenario").assign_coords({"valuation": str(self.valuation.name).lower()})
        yield query(valuations=valuations)


class ValuationFilter(Filter):
    def __init__(self, *args, scenario, **kwargs):
        super().__init__(*args, **kwargs)
        self.scenario = scenario

    def execute(self, query, *args, **kwargs):
        valuations = query.valuations
        prior = np.count_nonzero(~np.isnan(valuations["size"].values))
        scenario = str(self.scenario.name).lower()
        mask = valuations.sel({"scenario": scenario})
        mask = self.mask(mask, *args, **kwargs)
        mask = xr.broadcast(mask, valuations)[0]

        print(valuations)

        valuations = self.filter(valuations, *args, mask=mask, **kwargs)

        print(valuations)
        raise Exception()

        post = np.count_nonzero(~np.isnan(valuations["size"].values))
        query = query(valuations=valuations)
        __logger__.info(f"Filter: {repr(self)}|{str(query)}[{prior:.0f}|{post:.0f}]")
        yield query


class ValuationParser(Parser, index=list(INDEX_VARS.keys()), scope=list(SCOPE_VARS.keys()), columns=list(COLUMNS_VARS.keys())):
    def execute(self, query, *args, **kwargs):
        valuations = query.valuations
        valuations = self.parse(valuations, *args, **kwargs)
        valuations = valuations.reset_index(drop=False, inplace=False)

        print(valuations.where(valuations["scenario"] == "minimum").dropna(how="all", inplace=False))
        print(valuations.where(valuations["scenario"] == "maximum").dropna(how="all", inplace=False))

        raise Exception()

        query = query(valuations=valuations)
        yield query


class ValuationFile(DataframeFile, header=INDEX_VARS | SCOPE_VARS | COLUMNS_VARS): pass
class ValuationSaver(Saver):
    def execute(self, query, *args, **kwargs):
        pass


class ValuationLoader(Loader):
    def execute(self, query, *args, **kwargs):
        yield



