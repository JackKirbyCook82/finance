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
import xarray as xr
from abc import ABC
from itertools import chain
from scipy.stats import norm

from finance.variables import Variables, Querys
from support.files import File
from support.meta import ParametersMeta, RegistryMeta
from support.calculations import Calculation, Variable
from support.mixins import Function, Emptying, Sizing, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["OptionCalculator", "StockFile", "OptionFile"]
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


class PricingCalculation(Calculation, ABC, metaclass=RegistryMeta): pass
class BlackScholesEquation(PricingCalculation, register=Variables.Pricing.BLACKSCHOLES):
    τ = Variable("tau", np.int32, function=lambda to, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))
    yo = Variable("price", np.float32, function=lambda zx, zk, F: (zx - zk) * F)
    so = Variable("ratio", np.float32, function=lambda xo, k: np.log(xo / k))

    zx = Variable("zx", np.float32, function=lambda xo, Θ, N1: xo * Θ * N1)
    zk = Variable("zk", np.float32, function=lambda k, Θ, N2, D: k * Θ * D * N2)
    d1 = Variable("d1", np.float32, function=lambda so, α, β: (so + α) / β)
    d2 = Variable("d2", np.float32, function=lambda d1, β: d1 - β)
    N1 = Variable("N1", np.float32, function=lambda d1, Θ: norm.cdf(Θ * d1))
    N2 = Variable("N2", np.float32, function=lambda d2, Θ: norm.cdf(Θ * d2))

    α = Variable("alpha", np.float32, function=lambda τ, δ, ρ: (ρ + np.divide(np.power(δ * np.sqrt(252), 2), 2)) * τ / 252)
    β = Variable("beta", np.float32, function=lambda τ, δ: δ * np.sqrt(252) * np.sqrt(τ / 252))
    D = Variable("discount", np.float32, function=lambda τ, ρ: np.exp(-ρ * τ / 252))
    F = Variable("factor", np.float32, function=lambda f, Θ, Φ, ε, ω: 1 + f(Θ, Φ, ε, ω))
    Θ = Variable("theta", np.int32, function=lambda i: + int(Variables.Theta(str(i))))
    Φ = Variable("phi", np.int32, function=lambda j: + int(Variables.Phi(str(j))))

    tτ = Variable("expire", np.datetime64, locator=)
    to = Variable("current", np.datetime64, locator=)
    xo = Variable("underlying", np.float32, locator=)

    δ = Variable("volatility", np.float32, locator=)
    i = Variable("option", Variables.Options, locator=)
    j = Variable("position", Variables.Positions, locator=)
    k = Variable("strike", np.float32, locator=)
    ε = Variable("epsilon", np.float32, locator=)
    ω = Variable("omega", np.float32, locator=)
    ρ = Variable("discount", np.float32, locator=)
    f = Variable("factor", types.FunctionType, locator=)

    def calculate(self, exposures, *args, discount, factor=lambda Θ, Φ, ε, ω: 0, epsilon=0, omega=0, **kwargs):
        constants = dict(factor=factor, discount=discount, epsilon=epsilon, omega=omega)
        sources = {source: exposures[source] for source in self.sources}
        for axis, content in chain(sources.items(), constants.items()): self[axis] = content

    def execute(self, exposures, *args, **kwargs):
        invert = lambda position: Variables.Positions(int(Variables.Positions.LONG) + int(Variables.Positions.SHORT) - int(position))
        dataarrays = {axis: exposures[axis] for axis in ("ticker", "expire", "instrument", "option", "position", "strike")}
        dataarrays["position"] = dataarrays["position"].apply(invert)
        dataarrays.update({axis: self[axis](*args, **kwargs) for axis in ("price", "underlying", "current")})
        return xr.merge(dataarrays)


class OptionCalculator(Function, Logging, Sizing, Emptying):
    def __init__(self, *args, pricing, sizings={}, timings={}, **kwargs):
        assert pricing in list(Variables.Pricing)
        Function.__init__(self, *args, **kwargs)
        Logging.__init__(self, *args, **kwargs)
        columns = dict(pricing=["price", "underlying"], sizing=["volume", "size", "interest"], timing=["current"])
        self.__calculation = PricingCalculation[pricing](*args, **kwargs)
        self.__index = list(Variables.Querys.Product)
        self.__columns = columns
        self.__sizings = sizings
        self.__timings = timings

    def execute(self, source, *args, **kwargs):
        assert isinstance(source, tuple)
        contract, exposures, statistics = source
        assert isinstance(contract, Querys.Contract) and isinstance(exposures, pd.DataFrame) and isinstance(statistics, pd.DataFrame)
        if self.empty(exposures): return
        exposures = self.exposures(exposures, statistics, *args, **kwargs)
        options = self.calculate(exposures, *args, **kwargs)
        size = self.size(options)
        string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
        self.logger.info(string)
        if self.empty(options): return
        yield options

    def calculate(self, exposures, *args, **kwargs):
        assert isinstance(exposures, pd.DataFrame)
        pricings = self.calculation(exposures, *args, **kwargs)
        pricings = pricings[self.index + self.columns["pricing"]]
        sizings = lambda columns: {sizing: self.sizings.get(sizing, lambda cols: np.NaN)(columns) for sizing in self.columns["sizing"]}
        sizings = pricings.apply(sizings, axis=1, result_type="expand")
        timings = lambda columns: {timing: self.timings.get(timing, lambda cols: np.NaN)(columns) for timing in self.columns["timing"]}
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





