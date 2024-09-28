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

from finance.variables import Variables, Contract
from support.calculations import Variable, Equation, Calculation
from support.meta import RegistryMeta, ParametersMeta
from support.filtering import Filter
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityFilter", "SecurityCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class SecurityParameters(metaclass=ParametersMeta):
    filename = lambda contract: "_".join([str(contract.ticker).upper(), str(contract.expire.strftime("%Y%m%d"))])
    option = {"ticker": str, "strike": np.float32, "price": np.float32, "underlying": np.float32, "size": np.float32, "volume": np.float32, "interest": np.float32}
    stock = {"ticker": str, "price": np.float32, "volume": np.float32}
    parsers = {"instrument": Variables.Instruments, "option": Variables.Options, "position": Variables.Positions}
    formatters = {"instrument": int, "option": int, "position": int}
    dates = {"current": "%Y%m%d-%H%M", "expire": "%Y%m%d"}


class SecurityVariables(object):
    data = {Variables.Datasets.PRICING: ["price", "underlying", "current"], Variables.Datasets.SIZING: ["volume", "size", "interest"]}
    axes = {Variables.Querys.CONTRACT: ["ticker", "expire"], Variables.Datasets.SECURITY: ["instrument", "option", "position"]}

    def __init__(self, *args, **kwargs):
        self.contract = self.axes[Variables.Querys.CONTRACT]
        self.security = self.axes[Variables.Datasets.SECURITY]
        self.pricing = self.data[Variables.Datasets.PRICING]
        self.sizing = self.data[Variables.Datasets.SIZING]


class SecurityFile(File, datatype=pd.DataFrame, dates=SecurityParameters.dates, filename=SecurityParameters.filename, parsers=SecurityParameters.parsers, formatters=SecurityParameters.formatters): pass
class StockFile(File, variable=Variables.Instruments.STOCK, types=SecurityParameters.stock): pass
class OptionFile(File, variable=Variables.Instruments.OPTION, types=SecurityParameters.option): pass


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
    ε = Variable("ε", "divergence", np.float32, position="divergence")
    ω = Variable("ω", "lifespan", np.float32, position="lifespan")
    ρ = Variable("ρ", "discount", np.float32, position="discount")
    f = Variable("f", "factor", types.FunctionType, position="factor")


class PricingCalculation(Calculation, ABC, metaclass=RegistryMeta): pass
class BlackScholesCalculation(PricingCalculation, equation=BlackScholesEquation, register=Variables.Pricing.BLACKSCHOLES):
    def execute(self, exposures, *args, factor, discount, divergence, lifespan, **kwargs):
        invert = lambda position: Variables.Positions(int(Variables.Positions.LONG) + int(Variables.Positions.SHORT) - int(position))
        equation = self.equation(*args, **kwargs)
        yield from iter([exposures["ticker"], exposures["expire"]])
        yield from iter([exposures["instrument"], exposures["option"], exposures["position"].apply(invert), exposures["strike"]])
        yield equation.yo(exposures, factor=factor, discount=discount, divergence=divergence, lifespan=lifespan)
        yield equation.xo(exposures)
        yield equation.to(exposures)


class SecurityFilter(Filter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__variables = SecurityVariables(*args, **kwargs)

    def execute(self, securities, *args, **kwargs):
        assert isinstance(securities, pd.DataFrame)
        for contract, dataframe in self.contracts(securities, *args, **kwargs):
            if bool(dataframe.empty): continue
            prior = len(dataframe.dropna(how="all", inplace=False).index)
            dataframe = self.filter(dataframe, *args, **kwargs)
            assert isinstance(dataframe, pd.DataFrame)
            dataframe = dataframe.reset_index(drop=True, inplace=False)
            post = len(dataframe.dropna(how="all", inplace=False).index)
            string = f"Filtered: {repr(self)}|{str(contract)}[{prior:.0f}|{post:.0f}]"
            self.logger.info(string)
            if bool(dataframe.empty): continue
            yield dataframe

    def contracts(self, securities, *args, **kwargs):
        assert isinstance(securities, pd.DataFrame)
        for (ticker, expire), dataframe in securities.groupby(self.variables.contract):
            contract = Contract(ticker, expire)
            yield contract, dataframe

    @property
    def variables(self): return self.__variables


class SecurityCalculator(object):
    def __init__(self, *args, pricing, sizings, **kwargs):
        self.__calculation = PricingCalculation[pricing](*args, **kwargs)
        self.__variables = SecurityVariables(*args, **kwargs)
        self.__sizings = sizings
        self.__pricing = pricing

    def __call__(self, exposures, statistics, *args, **kwargs):
        assert isinstance(exposures, pd.DataFrame) and isinstance(statistics, pd.DataFrame)
        exposures = self.exposures(exposures, statistics, *args, **kwargs)
        for contract, dataframe in self.contracts(exposures):
            options = self.execute(dataframe, *args, **kwargs)
            size = len(options.dropna(how="all", inplace=False).index)
            string = f"Calculated: {repr(self)}|{str(contract)}[{size:.0f}]"
            self.logger.info(string)
            if bool(options.empty): continue
            yield options

    def contracts(self, exposures):
        assert isinstance(exposures, pd.DataFrame)
        for (ticker, expire), dataframe in exposures.groupby(self.variables.contract):
            if bool(dataframe.empty): continue
            contract = Contract(ticker, expire)
            yield contract, dataframe

    def execute(self, exposures, *args, **kwargs):
        assert isinstance(exposures, pd.DataFrame)
        options = self.calculate(exposures, *args, **kwargs)
        return options

    def calculate(self, exposures, *args, **kwargs):
        assert isinstance(exposures, pd.DataFrame)
        pricings = self.calculation(exposures, *args, lifespan=0, **kwargs)
        functions = {sizing: self.sizings.get(sizing, lambda cols: np.NaN) for sizing in self.variables.sizing}
        sizings = pricings.apply(functions, axis=1, result_type="expand")
        options = pd.concat([pricings, sizings], axis=1)
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
    def variables(self): return self.__variables
    @property
    def sizings(self): return self.__sizings
    @property
    def pricing(self): return self.__pricing




