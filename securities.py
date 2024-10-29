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
from numbers import Number
from scipy.stats import norm

from finance.variables import Variables, Querys
from support.files import File
from support.meta import ParametersMeta, RegistryMeta
from support.calculations import Calculation, Equation, Variable
from support.mixins import Function, Emptying, Sizing, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["OptionCalculator", "OptionFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class SecurityParameters(metaclass=ParametersMeta):
    filename = lambda query: "_".join([str(query.ticker), str(query.expire.strftime("%Y%m%d"))])
    types = {"ticker": str, "strike": np.float32, "price": np.float32, "underlying": np.float32, "size": np.float32, "volume": np.float32, "interest": np.float32}
    parsers = {"instrument": Variables.Instruments, "option": Variables.Options, "position": Variables.Positions}
    formatters = {"instrument": int, "option": int, "position": int, "strike": lambda strike: round(strike, 2), "underlying": lambda underlying: round(underlying, 2)}
    dates = {"current": "%Y%m%d-%H%M", "expire": "%Y%m%d"}


class SecurityFile(File, datatype=pd.DataFrame, **dict(SecurityParameters)): pass
class OptionFile(SecurityFile, variable=Variables.Instruments.OPTION): pass


class PricingEquation(Equation, ABC):
    τ = Variable("τ", "tau", np.int32, pd.Series, vectorize=True, function=lambda to, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))
    Θ = Variable("Θ", "theta", np.int32, pd.Series, vectorize=True, function=lambda i: + int(Variables.Theta(str(i))))
    Φ = Variable("Φ", "phi", np.int32, pd.Series, vectorize=True, function=lambda j: + int(Variables.Phi(str(j))))

    i = Variable("i", "option", Variables.Options, pd.Series, locator="option")
    j = Variable("j", "position", Variables.Positions, pd.Series, locator="position")
    k = Variable("k", "strike", np.float32, pd.Series, locator="strike")

    xo = Variable("xo", "underlying", np.float32, pd.Series, locator="underlying")
    to = Variable("to", "current", np.datetime64, pd.Series, locator="current")
    tτ = Variable("tτ", "expire", np.datetime64, pd.Series, locator="expire")

    δ = Variable("δ", "volatility", np.float32, pd.Series, locator="volatility")
    ρ = Variable("ρ", "discount", np.float32, types.NoneType, locator="discount")

class BlackScholesEquation(PricingEquation):
    yo = Variable("yo", "price", np.float32, pd.Series, vectorize=True, function=lambda zx, zk: (zx - zk))
    so = Variable("so", "ratio", np.float32, pd.Series, vectorize=True, function=lambda xo, k: np.log(xo / k))

    zx = Variable("zx", "zx", np.float32, pd.Series, vectorize=True, function=lambda xo, Θ, N1: xo * Θ * N1)
    zk = Variable("zk", "zk", np.float32, pd.Series, vectorize=True, function=lambda k, Θ, N2, D: k * Θ * D * N2)
    d1 = Variable("d1", "d1", np.float32, pd.Series, vectorize=True, function=lambda so, A, B: (so + A) / B)
    d2 = Variable("d2", "d2", np.float32, pd.Series, vectorize=True, function=lambda d1, B: d1 - B)
    N1 = Variable("N1", "N1", np.float32, pd.Series, vectorize=True, function=lambda d1, Θ: norm.cdf(Θ * d1))
    N2 = Variable("N2", "N2", np.float32, pd.Series, vectorize=True, function=lambda d2, Θ: norm.cdf(Θ * d2))

    A = Variable("A", "alpha", np.float32, pd.Series, vectorize=True, function=lambda τ, δ, ρ: (ρ + np.divide(np.power(δ * np.sqrt(252), 2), 2)) * τ / 252)
    B = Variable("B", "beta", np.float32, pd.Series, vectorize=True, function=lambda τ, δ: δ * np.sqrt(252) * np.sqrt(τ / 252))
    D = Variable("D", "discount", np.float32, pd.Series, vectorize=True, function=lambda τ, ρ: np.exp(-ρ * τ / 252))


class PricingCalculation(Calculation, ABC, metaclass=RegistryMeta): pass
class BlackScholesCalculation(PricingCalculation, equation=BlackScholesEquation, register=Variables.Pricing.BLACKSCHOLES):
    def execute(self, exposures, *args, discount, **kwargs):
        assert isinstance(exposures, pd.DataFrame) and isinstance(discount, Number)
        invert = lambda position: Variables.Positions(int(Variables.Positions.LONG) + int(Variables.Positions.SHORT) - int(position))
        with self.equation(exposures, discount=discount) as equation:
            yield equation["ticker"]
            yield equation["expire"]
            yield equation["instrument"]
            yield equation["option"]
            yield equation["position"].apply(invert)
            yield equation["strike"]
            yield equation["underlying"]
            yield equation.to()
            yield equation.yo()


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
        return options

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





