# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Valuation Objects
@author: Jack Kirby Cook

"""

import logging
from abc import ABC
import numpy as np
import xarray as xr

from support.files import DataframeFile
from support.pipelines import Processor
from support.processes import Saver, Loader, Filter, Parser
from support.calculations import Calculation, equation, source, constant

from finance.variables import Securities, Actions

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ValuationCalculation", "ValuationCalculator", "ValuationFilter", "ValuationParser", "ValuationLoader", "ValuationSaver", "ValuationFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


COLUMNS_VARS = {"strategy": str, "apy": np.float32, "npv": np.float32, "tau": np.int32, "size": np.int32, "quantity": np.int32}
INDEX_VARS = {option: str for option in list(map(str, Securities.Options))}
SCOPE_VARS = {"ticker": str, "expire": np.datetime64, "date": np.datetime64}
OPEN_VARS = {"to": "date", "tτ": "expire", "qo": "size"}
CLOSE_VARS = {"to": "date", "tτ": "expire", "qo": "size", "Δo": "quantity"}


class ValuationCalculation(Calculation, ABC):
    tau = equation("tau", "tau", np.int32, domain=("v.to", "v.tτ"), function=lambda to, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))
    npv = equation("npv", "npv", np.float32, domain=("v.vo", "v.vτ", "tau", "ρ"), function=lambda vo, vτ, tau, ρ: vo + np.divide(vτ, np.power(1 + ρ, tau / 365)))
    irr = equation("irr", "irr", np.float32, domain=("v.vo", "v.vτ", "tau"), function=lambda vo, vτ, tau: np.power(-vo / vτ, np.power(tau, -1)))
    apy = equation("apy", "apy", np.float32, domain=("irr", "tau"), function=lambda irr, tau: np.power(irr + 1, np.power(tau / 365, -1)) - 1)
    ρ = constant("ρ", "discount", position="discount")

    def __init_subclass__(cls, *args, **kwargs):
        cls.action = kwargs.get("action", getattr(cls, "action", None))


class OpenValuationCalculation(ValuationCalculation, action=Actions.OPEN):
    v = source("v", "valuation", position=0, variables=OPEN_VARS)

    def execute(self, feed, *args, discount, **kwargs):
        yield self.npv(feed, discount=discount)
        yield self.apy(feed)
        yield self.exp(feed)
        yield self.tau(feed)
        yield self["v"].qo(feed)


class CloseValuationCalculation(ValuationCalculation, action=Actions.CLOSE):
    v = source("v", "valuation", position=0, variables=CLOSE_VARS)

    def execute(self, feed, *args, discount, **kwargs):
        yield self.npv(feed, discount=discount)
        yield self.apy(feed)
        yield self.exp(feed)
        yield self.tau(feed)
        yield self["v"].qo(feed)
        yield self["v"].Δo(feed)


class ArbitrageCalculation(ValuationCalculation, ABC):
    v = source("v", "arbitrage", position=0, variables={"vo": "spot"})

class MinimumArbitrageCalculation(ArbitrageCalculation, ABC):
    v = source("v", "minimum", position=0, variables={"vτ": "minimum"})

class MaximumArbitrageCalculation(ArbitrageCalculation, ABC):
    v = source("v", "maximum", position=0, variables={"vτ": "maximum"})


class OpenMinimumCalculation(OpenValuationCalculation, MinimumArbitrageCalculation): pass
class CloseMinimumCalculation(CloseValuationCalculation, MinimumArbitrageCalculation): pass
class OpenMaximumCalculation(OpenValuationCalculation, MaximumArbitrageCalculation): pass
class CloseMaximumCalculation(CloseValuationCalculation, MaximumArbitrageCalculation): pass


class ValuationCalculator(Processor, title="Calculated"):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)

    def execute(self, query, *args, **kwargs):
        strategies = query.strategies
        assert isinstance(strategies, xr.Dataset)

        print(strategies)
        raise Exception()


class ValuationFilter(Filter):
    def execute(self, query, *args, **kwargs):
        yield


class ValuationParser(Parser):
    def execute(self, query, *args, **kwargs):
        yield


class ValuationFile(DataframeFile, header=INDEX_VARS | SCOPE_VARS | COLUMNS_VARS): pass
class ValuationSaver(Saver):
    def execute(self, query, *args, **kwargs):
        pass


class ValuationLoader(Loader):
    def execute(self, query, *args, **kwargs):
        yield



