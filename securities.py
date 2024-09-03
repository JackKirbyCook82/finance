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
from itertools import count
from scipy.stats import norm
from collections import OrderedDict as ODict

from finance.variables import Variables, Contract
from support.calculations import Variable, Equation, Calculation
from support.meta import ParametersMeta
from support.pipelines import Processor
from support.filtering import Filter
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityFiles", "SecurityFilter", "SecurityCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Parameters(metaclass=ParametersMeta):
    types = {"ticker": str, "strike": np.float32, "price": np.float32, "underlying": np.float32, "size": np.float32, "volume": np.float32, "interest": np.float32}
    parsers = {"instrument": Variables.Instruments, "option": Variables.Options, "position": Variables.Positions}
    formatters = {"instrument": int, "option": int, "position": int}
    dates = {"current": "%Y%m%d-%H%M", "expire": "%Y%m%d"}
    filename = lambda contract: "_".join([str(contract.ticker).upper(), str(contract.expire.strftime("%Y%m%d"))])
    datatype = pd.DataFrame

class Headers:
    options = ["ticker", "expire", "instrument", "option", "position", "strike", "volume", "size", "interest", "current"]
    stocks = ["ticker", "instrument", "position", "price", "volume", "size", "current"]


class StockFile(File, variable=Variables.Instruments.STOCK, header=Headers.stocks, **dict(Parameters)): pass
class OptionFile(File, variable=Variables.Instruments.OPTION, header=Headers.options, **dict(Parameters)): pass
class SecurityFiles(object): Stock = StockFile; Option = OptionFile


class SecurityFilter(Filter, reporting=True, variable=Variables.Querys.CONTRACT):
    def processor(self, contents, *args, **kwargs):
        parameters = dict(variable=contents[self.variable])
        securities = list(self.calculate(contents, *args, **parameters, **kwargs))
        if not bool(securities): return
        yield contents | ODict(securities)

    def calculate(self, contents, *args, **kwargs):
        for variable in list(Variables.Instruments):
            if not bool(variable): continue
            securities = contents.get(variable, None)
            if securities is None: continue
            securities = self.filter(securities, *args, **kwargs)
            securities = securities.reset_index(drop=True, inplace=False)
            if bool(securities.empty): continue
            yield variable, securities


class BlackScholesEquation(Equation):
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


class BlackScholesCalculation(Calculation, equation=BlackScholesEquation):
    def execute(self, exposures, *args, factor, discount, divergence, lifespan, **kwargs):
        invert = lambda position: Variables.Positions(int(Variables.Positions.LONG) + int(Variables.Positions.SHORT) - int(position))
        equation = self.equation(*args, **kwargs)
        yield from iter([exposures["ticker"], exposures["expire"]])
        yield from iter([exposures["instrument"], exposures["option"], exposures["position"].apply(invert), exposures["strike"]])
        yield equation.yo(exposures, factor=factor, discount=discount, divergence=divergence, lifespan=lifespan)
        yield equation.xo(exposures)
        yield equation.to(exposures)


class SecurityCalculator(Processor, title="Calculated", reporting=True, variable=Variables.Querys.CONTRACT):
    def __init__(self, *args, size, volume=lambda cols: np.NaN, interest=lambda cols: np.NaN, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        calculations = {Variables.Instruments.OPTION: BlackScholesCalculation}
        self.__calculations = {variable: calculation(*args, **kwargs) for variable, calculation in calculations.items()}
        self.__functions = dict(size=size, volume=volume, interest=interest)
        self.__lifespans = ODict()

    def processor(self, contents, *args, **kwargs):
        contract, exposures, statistics = contents[Variables.Querys.CONTRACT], contents[Variables.Datasets.EXPOSURE], contents[Variables.Technicals.STATISTIC]
        assert isinstance(contract, Contract) and isinstance(exposures, pd.DataFrame) and isinstance(statistics, pd.DataFrame)
        if contract not in self.lifespans: self.lifespans[contract] = count(start=0, step=1)
        exposures = self.exposures(exposures, statistics, *args, **kwargs)
        lifespan = next(self.lifespans[contract])
        securities = list(self.calculate(exposures, *args, lifespan=lifespan, **kwargs))
        if not bool(securities): return
        yield contents | ODict(securities)

    def calculate(self, exposures, *args, **kwargs):
        for variable, calculation in self.calculations.items():
            pricing = calculation(exposures, *args, **kwargs)
            if bool(pricing.empty): continue
            function = lambda cols: {key: value(cols) for key, value in self.functions.items()}
            sizing = pricing.apply(function, axis=1, result_type="expand")
            securities = pd.concat([pricing, sizing], axis=1)
            yield variable, securities

    @staticmethod
    def exposures(exposures, statistics, *args, current, **kwargs):
        statistics = statistics.where(statistics["date"] == pd.to_datetime(current))
        exposures = pd.merge(exposures, statistics, how="inner", on="ticker")
        return exposures

    @property
    def calculations(self): return self.__calculations
    @property
    def functions(self): return self.__functions
    @property
    def lifespans(self): return self.__lifespans



