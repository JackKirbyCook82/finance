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
from itertools import count
from scipy.stats import norm
from collections import OrderedDict as ODict

from finance.variables import Variables, Contract
from support.calculations import Variable, Equation, Calculation
from support.processes import Calculator, Filter
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityFilter", "SecurityCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


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
    def execute(self, contract, securities, *args, **kwargs):
        assert isinstance(contract, Contract) and isinstance(securities, pd.DataFrame)
        options = self.filter(securities, *args, variable=contract, **kwargs)
        assert isinstance(options, pd.DataFrame)
        options = options.reset_index(drop=True, inplace=False)
        return options


class SecurityVariables(object):
    data = {Variables.Datasets.PRICING: ["price", "underlying", "current"], Variables.Datasets.SIZING: ["volume", "size", "interest"]}
    axes = {Variables.Querys.CONTRACT: ["ticker", "expire"], Variables.Datasets.SECURITY: ["instrument", "option", "position"]}

    def __init__(self, *args, pricing, sizings={}, **kwargs):
        assert pricing in list(Variables.Pricing)
        self.sizings = {sizing: sizings.get(sizing, lambda cols: np.NaN) for sizing in self.data[Variables.Datasets.SIZING]}
        self.pricing = pricing


class SecurityCalculator(Calculator, calculations=dict(PricingCalculation), variables=SecurityVariables):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lifespans = ODict()

    def execute(self, contract, exposures, statistics, *args, **kwargs):
        assert isinstance(contract, Contract) and isinstance(exposures, pd.DataFrame) and isinstance(statistics, pd.DataFrame)
        exposures = self.exposures(exposures, statistics, *args, **kwargs)
        pricings = self.pricings(contract, exposures, *args, **kwargs)
        sizings = self.sizings(pricings, *args, **kwargs)
        options = pd.concat([pricings, sizings], axis=1)
        size = self.size(options)
        string = f"{str(self.title)}: {repr(self)}|{str(contract)}[{size:.0f}]"
        self.logger.info(string)
        return options

    def pricings(self, contract, exposures, *args, **kwargs):
        if contract not in self.lifespans: self.lifespans[contract] = count(start=0, step=1)
        lifespan = next(self.lifespans[contract])
        pricings = self.calculations[self.variables.pricing](exposures, *args, lifespan=lifespan, **kwargs)
        assert isinstance(pricings, pd.DataFrame)
        return pricings

    def sizings(self, pricings, *args, **kwargs):
        sizings = pricings.apply(self.variables.sizings, axis=1, result_type="expand")
        return sizings

    @staticmethod
    def exposures(exposures, statistics, *args, current, **kwargs):
        statistics = statistics.where(statistics["date"] == pd.to_datetime(current))
        exposures = pd.merge(exposures, statistics, how="inner", on="ticker")
        return exposures





