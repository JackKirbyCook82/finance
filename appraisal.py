# -*- coding: utf-8 -*-
"""
Created on Fri May 23 2025
@name:   Appraisal Objects
@author: Jack Kirby Cook

"""

import types
import numpy as np
import pandas as pd
from abc import ABC, ABCMeta
from scipy.stats import norm

from finance.concepts import Concepts, Querys
from support.mixins import Emptying, Sizing, Partition, Logging
from support.meta import RegistryMeta
from calculations import Variables, Equations

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["AppraisalCalculator"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


class AppraisalEquationMeta(RegistryMeta, type(Equations.Vectorized.Table), ABCMeta): pass
class AppraisalEquation(Equations.Vectorized.Table, ABC, metaclass=AppraisalEquationMeta):
    τ = Variables.Dependent("τ", "tau", np.float32, function=lambda to, tτ: (np.datetime64(tτ, "ns") - np.datetime64(to, "ns")) / np.timedelta64(364, 'D'))

    zx = Variables.Dependent("zx", ("zscore", "itm"), np.float32, function=lambda zxk, zvt, zrt: zxk + zvt + zrt)
    zk = Variables.Dependent("zk", ("zscore", "otm"), np.float32, function=lambda zxk, zvt, zrt: zxk - zvt + zrt)

    zxk = Variables.Dependent("zxk", ("zscore", "strike"), np.float32, function=lambda x, k, σ, τ: np.log(x / k) / np.sqrt(τ) / σ)
    zvt = Variables.Dependent("zvt", ("zscore", "volatility"), np.float32, function=lambda σ, τ: np.sqrt(τ) * σ / 2)
    zrt = Variables.Dependent("zrt", ("zscore", "interest"), np.float32, function=lambda σ, r, τ: np.sqrt(τ) * r / σ)

    tτ = Variables.Independent("tτ", "expire", np.datetime64, locator="expire")
    to = Variables.Constant("to", "current", np.datetime64, locator="current")

    i = Variables.Independent("i", "option", Concepts.Securities.Option, locator="option")
    j = Variables.Independent("j", "position", Concepts.Securities.Position, locator="position")
    x = Variables.Independent("x", "adjusted", np.float32, locator="adjusted")
    σ = Variables.Independent("σ", "volatility", np.float32, locator="volatility")
    μ = Variables.Independent("μ", "trend", np.float32, locator="trend")
    k = Variables.Independent("k", "strike", np.float32, locator="strike")
    r = Variables.Constant("r", "interest", np.float32, locator="interest")


class BlackScholesEquation(AppraisalEquation, register=Concepts.Appraisal.BLACKSCHOLES):
    vo = Variables.Dependent("vo", "value", np.float32, function=lambda x, k, zx, zk, r, τ, i: x * norm.cdf(zx * int(i)) * int(i) - k * norm.cdf(zk * int(i)) * int(i) / np.exp(r * τ))

    def execute(self, securities, /, current, interest):
        yield from super().execute(securities, current=current, interest=interest)
        yield self.vo(securities, current=current, interest=interest)


class GreekEquation(AppraisalEquation, register=Concepts.Appraisal.GREEKS):
    Θo = Variables.Dependent("Θo", "theta", np.float32, function=lambda zx, zk, x, k, r, σ, τ, i: - norm.cdf(zk * int(i)) * int(i) * k * r / np.exp(r * τ) - norm.pdf(zx) * x * σ / np.sqrt(τ) / 2)
    Po = Variables.Dependent("Po", "rho", np.float32, function=lambda zk, k, r, τ, i: + norm.cdf(zk * int(i)) * int(i) * k * τ / np.exp(r * τ))
    Δo = Variables.Dependent("Δo", "delta", np.float32, function=lambda zx, i: + norm.cdf(zx * int(i)) * int(i))
    Γo = Variables.Dependent("Γo", "gamma", np.float32, function=lambda zx, x, σ, τ: + norm.pdf(zx) / np.sqrt(τ) / σ / x)
    Vo = Variables.Dependent("Vo", "vega", np.float32, function=lambda zx, x, τ: + norm.pdf(zx) * np.sqrt(τ) * x)

    def execute(self, securities, /, current, interest):
        yield from super().execute(securities, current=current, interest=interest)
        yield self.Θo(securities, current=current, interest=interest)
        yield self.Po(securities, current=current, interest=interest)
        yield self.Δo(securities, current=current, interest=interest)
        yield self.Γo(securities, current=current, interest=interest)
        yield self.Vo(securities, current=current, interest=interest)


class ImpliedEquation(object, register=Concepts.Appraisal.IMPLIED):
    def execute(self, securities, /, current, interest):
        pass


class AppraisalCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, appraisals, **kwargs):
        super().__init__(*args, **kwargs)
        equations = [equation for appraisal, equation in iter(AppraisalEquation) if appraisal in appraisals]
        self.__equation = (AppraisalEquation + equations)(*args, **kwargs)

    def execute(self, options, technicals=None, /, **kwargs):
        assert isinstance(options, pd.DataFrame)
        assert isinstance(technicals, (pd.DataFrame, types.NoneType))
        if self.empty(options): return
        querys = self.keys(options, by=Querys.Settlement)
        querys = ",".join(list(map(str, querys)))
        if technicals is not None: options = self.technicals(options, technicals, **kwargs)
        options = self.calculate(options, **kwargs)
        size = self.size(options)
        self.console(f"{str(querys)}[{int(size):.0f}]")
        if self.empty(options): return
        yield options

    def calculate(self, securities, *args, current, interest, **kwargs):
        assert isinstance(securities, pd.DataFrame)
        appraisals = self.equation(securities, current=current, interest=interest)
        assert isinstance(appraisals, pd.DataFrame)
        appraisals = pd.concat([securities, appraisals], axis=1)
        appraisals = appraisals.reset_index(drop=True, inplace=False)
        return appraisals

    @staticmethod
    def technicals(options, technicals, *args, **kwargs):
        assert isinstance(options, pd.DataFrame) and isinstance(technicals, pd.DataFrame)
        mask = technicals["date"] == technicals["date"].max()
        technicals = technicals.where(mask).dropna(how="all", inplace=False)
        technicals = technicals.drop(columns="date", inplace=False)
        options = options.merge(technicals, how="left", on=list(Querys.Symbol), sort=False, suffixes=("", ""))
        return options

    @property
    def equation(self): return self.__equation



