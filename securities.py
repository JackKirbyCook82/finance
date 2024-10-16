# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import types
import logging
import numpy as np
import pandas as pd
from abc import ABC
from scipy.stats import norm

from finance.variables import Variables, Querys
from support.mixins import Emptying, Sizing, Logging, Pipelining, Sourcing
from support.calculations import Variable, Equation, Calculation
from support.meta import RegistryMeta, ParametersMeta
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class SecurityParameters(metaclass=ParametersMeta):
    types = {"ticker": str, "strike": np.float32, "price": np.float32, "underlying": np.float32, "size": np.float32, "volume": np.float32, "interest": np.float32}
    parsers = {"instrument": Variables.Instruments, "option": Variables.Options, "position": Variables.Positions}
    formatters = {"instrument": int, "option": int, "position": int}
    dates = {"current": "%Y%m%d-%H%M", "expire": "%Y%m%d"}


class SecurityFile(File, datatype=pd.DataFrame, **dict(SecurityParameters)): pass
class StockFile(SecurityFile, variable=Variables.Instruments.STOCK): pass
class OptionFile(SecurityFile, variable=Variables.Instruments.OPTION): pass


class PricingEquation(Equation): pass
class BlackScholesEquation(PricingEquation):
    τ = Variable("τ", "tau", np.int32, function=lambda to, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))
    so = Variable("so", "ratio", np.float32, function=lambda xo, k: np.log(xo / k))
    yo = Variable("yo", "price", np.float32, function=lambda zx, zk, F: (zx - zk) * F)

    zx = Variable("zx", "zx", np.float32, function=lambda xo, Θ, N1: xo * Θ * N1)
    zk = Variable("zk", "zk", np.float32, function=lambda k, Θ, N2, D: k * Θ * D * N2)
    d1 = Variable("d1", "d1", np.float32, function=lambda so, α, β: (so + α) / β)
    d2 = Variable("d2", "d2", np.float32, function=lambda d1, β: d1 - β)
    N1 = Variable("N1", "N1", np.float32, function=lambda d1, Θ: norm.cdf(Θ * d1))
    N2 = Variable("N2", "N2", np.float32, function=lambda d2, Θ: norm.cdf(Θ * d2))

    α = Variable("α", "alpha", np.float32, function=lambda τ, δ, ρ: (ρ + np.divide(np.power(δ * np.sqrt(252), 2), 2)) * τ / 252)
    β = Variable("β", "beta", np.float32, function=lambda τ, δ: δ * np.sqrt(252) * np.sqrt(τ / 252))
    D = Variable("D", "discount", np.float32, function=lambda τ, ρ: np.exp(-ρ * τ / 252))
    F = Variable("F", "factor", np.float32, function=lambda f, Θ, Φ, ε, ω: 1 + f(Θ, Φ, ε, ω))
    Θ = Variable("Θ", "theta", np.int32, function=lambda i: + int(Variables.Theta(str(i))))
    Φ = Variable("Φ", "phi", np.int32, function=lambda j: + int(Variables.Phi(str(j))))

    tτ = Variable("tτ", "expire", np.datetime64, position=0, locator="expire")
    to = Variable("to", "current", np.datetime64, position=0, locator="date")
    xo = Variable("xo", "underlying", np.float32, position=0, locator="price")

    δ = Variable("δ", "volatility", np.float32, position=0, locator="volatility")
    i = Variable("i", "option", Variables.Options, position=0, locator="option")
    j = Variable("j", "position", Variables.Positions, position=0, locator="position")
    k = Variable("k", "strike", np.float32, position=0, locator="strike")
    ε = Variable("ε", "epsilon", np.float32, position="epsilon")
    ω = Variable("ω", "omega", np.float32, position="omega")
    ρ = Variable("ρ", "discount", np.float32, position="discount")
    f = Variable("f", "factor", types.FunctionType, position="factor")


class PricingCalculation(Calculation, ABC, metaclass=RegistryMeta): pass
class BlackScholesCalculation(PricingCalculation, equation=BlackScholesEquation, register=Variables.Pricing.BLACKSCHOLES):
    def execute(self, exposures, *args, discount, factor=lambda Θ, Φ, ε, ω: 0, epsilon=0, omega=0, **kwargs):
        invert = lambda position: Variables.Positions(int(Variables.Positions.LONG) + int(Variables.Positions.SHORT) - int(position))
        equation = self.equation(*args, **kwargs)
        yield from iter([exposures["ticker"], exposures["expire"]])
        yield from iter([exposures["instrument"], exposures["option"], exposures["position"].apply(invert), exposures["strike"]])
        yield equation.yo(exposures, factor=factor, discount=discount, epsilon=epsilon, omega=omega)
        yield equation.xo(exposures)
        yield equation.to(exposures)


class SecurityCalculator(Pipelining, Sourcing, Logging, Sizing, Emptying):
    def __init__(self, *args, instrument, pricing, sizings={}, timings={}, **kwargs):
        assert instrument in list(Variables.Instruments) and pricing in list(Variables.Pricing)
        Pipelining.__init__(self, *args, **kwargs)
        Logging.__init__(self, *args, **kwargs)
        index = {Variables.Instruments.STOCK: ["ticker"], Variables.Instruments.OPTION: ["ticker", "expire", "strike"]}
        columns = dict(pricing=["price", "underlying"], sizing=["volume", "size", "interest"], timing=["current"])
        self.__calculation = PricingCalculation[pricing](*args, **kwargs)
        self.__index = index[instrument]
        self.__columns = columns
        self.__sizings = sizings
        self.__timings = timings

    def execute(self, exposures, statistics, *args, **kwargs):
        assert isinstance(exposures, pd.DataFrame) and isinstance(statistics, pd.DataFrame)
        if self.empty(exposures): return
        exposures = self.exposures(exposures, statistics, *args, **kwargs)
        for contract, dataframe in self.source(exposures, keys=list(Querys.Contract)):
            contract = Querys.Contract(contract)
            if self.empty(dataframe): continue
            options = self.calculate(dataframe, *args, **kwargs)
            size = self.size(options)
            string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
            self.logger.info(string)
            if self.empty(options): continue
            yield options

    def calculate(self, exposures, *args, **kwargs):
        assert isinstance(exposures, pd.DataFrame)
        pricings = self.calculation(exposures, *args, **kwargs)
        pricings = pricings[self.index + self.columns["pricing"]]
        sizings = lambda cols: {sizing: self.sizings.get(sizing, np.NaN) for sizing in self.columns["sizing"]}
        sizings = pricings.apply(sizings, axis=1, result_type="expand")
        timings = lambda cols: {timing: self.timings.get(timing, np.NaN) for timing in self.columns["timing"]}
        timings = pricings.apply(timings, axis=1, result_type="expand")
        options = pd.concat([pricings, sizings, timings], axis=1)
        return options

    @staticmethod
    def exposures(exposures, statistics, *args, current, **kwargs):
        assert isinstance(exposures, pd.DataFrame) and isinstance(statistics, pd.DataFrame)
        statistics = statistics.where(statistics["date"] == pd.to_datetime(current))
        exposures = pd.merge(exposures, statistics, how="inner", on="ticker")
        return exposures

    @property
    def calculation(self): return self.__calculations
    @property
    def sizings(self): return self.__sizings
    @property
    def timings(self): return self.__timings
    @property
    def columns(self): return self.__columns
    @property
    def index(self): return self.__index





