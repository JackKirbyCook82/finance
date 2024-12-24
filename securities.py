# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import types
import numpy as np
import pandas as pd
from abc import ABC
from numbers import Number
from scipy.stats import norm

from finance.variables import Variables, Querys
from support.mixins import Emptying, Sizing, Logging, Separating
from support.calculations import Calculation, Equation, Variable
from support.meta import RegistryMeta, DictionaryMeta
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["OptionCalculator", "OptionFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class SecurityParameters(object, metaclass=DictionaryMeta):
    formatters = {"instrument": int, "option": int, "position": int, "strike": lambda strike: round(strike, 2), "underlying": lambda underlying: round(underlying, 2)}
    types = {"ticker": str, "strike": np.float32, "price": np.float32, "underlying": np.float32, "size": np.float32, "volume": np.float32, "interest": np.float32}
    parsers = {"instrument": Variables.Instruments, "option": Variables.Options, "position": Variables.Positions}
    dates = {"current": "%Y%m%d-%H%M", "expire": "%Y%m%d"}

class SecurityFile(File, ABC, **dict(SecurityParameters)): pass
class StockFile(SecurityFile, variable=Variables.Instruments.STOCK): pass
class OptionFile(SecurityFile, variable=Variables.Instruments.OPTION): pass


class PricingEquation(Equation, ABC):
    τ = Variable("τ", "tau", np.int32, pd.Series, vectorize=True, function=lambda to, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))
    Θ = Variable("Θ", "theta", np.int32, pd.Series, vectorize=True, function=lambda i: + int(Variables.Theta[str(i)]))
    Φ = Variable("Φ", "phi", np.int32, pd.Series, vectorize=True, function=lambda j: + int(Variables.Phi[str(j)]))

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
            yield exposures["ticker"]
            yield exposures["expire"]
            yield exposures["instrument"]
            yield exposures["option"]
            yield exposures["position"].apply(invert)
            yield exposures["strike"]
            yield exposures["underlying"]
            yield exposures["current"]
            yield equation.yo()


class OptionCalculator(Logging, Sizing, Emptying, Separating):
    def __init__(self, *args, assumptions, **kwargs):
        super().__init__(*args, **kwargs)
        self.__calculation = PricingCalculation[assumptions.pricing](*args, **kwargs)
        self.__sizing = lambda cols: dict(assumptions.sizing)
        self.__current = assumptions.timing.current
        self.__pricing = assumptions.pricing
        self.__query = Querys.Contract

    def execute(self, exposures, statistics, *args, **kwargs):
        assert isinstance(exposures, pd.DataFrame) and isinstance(statistics, pd.DataFrame)
        if self.empty(exposures): return
        exposures = self.exposures(exposures, statistics, *args, **kwargs)
        for parameters, dataframe in self.separate(exposures, *args, fields=self.fields, **kwargs):
            contract = self.query(parameters)
            options = self.calculate(dataframe, *args, **kwargs)
            size = self.size(options)
            string = f"Calculated: {repr(self)}|{str(contract)}[{int(size):.0f}]"
            self.logger.info(string)
            if self.empty(options): continue
            yield options

    def calculate(self, exposures, *args, **kwargs):
        assert isinstance(exposures, pd.DataFrame)
        pricings = self.calculation(exposures, *args, **kwargs)
        sizings = pricings.apply(self.sizing, axis=1, result_type="expand")
        options = pd.concat([pricings, sizings], axis=1)
        return options

    def exposures(self, exposures, statistics, *args, **kwargs):
        assert isinstance(exposures, pd.DataFrame) and isinstance(statistics, pd.DataFrame)
        statistics = statistics.where(statistics["date"] == pd.to_datetime(self.current.date()))
        exposures = pd.merge(exposures, statistics, how="inner", on="ticker")
        exposures = exposures.rename(columns={"price": "underlying"}, inplace=False)
        exposures["current"] = pd.to_datetime(self.current)
        return exposures

    @property
    def fields(self): return list(self.__query)
    @property
    def calculation(self): return self.__calculation
    @property
    def current(self): return self.__current
    @property
    def pricing(self): return self.__pricing
    @property
    def sizing(self): return self.__sizing
    @property
    def query(self): return self.__query




