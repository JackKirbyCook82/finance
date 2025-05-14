# -*- coding: utf-8 -*-
"""
Created on Tues May 13 2025
@name:   Options Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC
from scipy.stats import norm

from finance.variables import Variables, Querys
from support.calculations import Calculation, Equation, Variable
from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["OptionCalculator"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


class OptionEquation(Equation, ABC, datatype=pd.Series, vectorize=True):
    Θ = Variable.Dependent("Θ", "theta", np.float32, function=lambda vx, vk, zx, σ, q, r, τ, i, j: (vx * int(i) * q - vk * int(i) * r - norm.pdf(zx) * σ / np.exp(q * τ) / np.sqrt(τ) / 2) * int(j) / 364)
    Δ = Variable.Dependent("Δ", "delta", np.float32, function=lambda zx, q, τ, i, j: + norm.cdf(zx * int(i)) * int(i) * int(j) / np.exp(q * τ))
    Γ = Variable.Dependent("Γ", "gamma", np.float32, function=lambda zx, x, σ, q, τ, j: norm.pdf(zx) * int(j) / np.exp(q * τ) / np.sqrt(τ) / x / σ)
    P = Variable.Dependent("P", "rho", np.float32, function=lambda zk, k, r, τ, i, j: - norm.cdf(zk * int(i)) * int(i) * k * τ * int(j) / np.exp(r * τ))
    V = Variable.Dependent("V", "vega", np.float32, function=lambda zx, x, σ, q, τ, j: norm.pdf(zx) * np.sqrt(τ) * x * int(j) / np.exp(q * τ))

    τ = Variable.Dependent("τ", "tau", np.float32, function=lambda to, tτ: (np.datetime64(tτ, "ns") - np.datetime64(to, "ns")) / np.timedelta64(364, 'D'))
    v = Variable.Dependent("y", "valuation", np.float32, function=lambda vx, vk, j: (vx - vk) * int(j))
    vx = Variable.Dependent("yx", "underlying", np.float32, function=lambda x, z, τ, q, i: x * int(i) * norm.cdf(z * int(i)) / np.exp(q * τ))
    vk = Variable.Dependent("yk", "strike", np.float32, function=lambda k, z, τ, r, i: k * int(i) * norm.cdf(z * int(i)) / np.exp(r * τ))

    zx = Variable.Dependent("zx", "underlying", np.float32, function=lambda zxk, zvt, zrt, zqt: zxk + zvt + zrt + zqt)
    zk = Variable.Dependent("zx", "strike", np.float32, function=lambda zxk, zvt, zrt, zqt: zxk - zvt + zrt + zqt)

    zxk = Variable.Dependent("zxk", "strike", np.float32, function=lambda x, k, σ, τ: np.log(x / k) / np.sqrt(τ) / σ)
    zvt = Variable.Dependent("zvt", "volatility", np.float32, function=lambda σ, τ: np.sqrt(τ) * σ / 2)
    zrt = Variable.Dependent("zrt", "interest", np.float32, function=lambda σ, r, τ: np.sqrt(τ) * r / σ)
    zqt = Variable.Dependent("zqt", "dividend", np.float32, function=lambda σ, q, τ: np.sqrt(τ) * q / σ)

    tτ = Variable.Independent("tτ", "expire", np.datetime64, locator="expire")
    to = Variable.Constant("to", "current", np.datetime64, locator="current")

    x = Variable.Independent("x", "underlying", np.float32, locator="underlying")
    σ = Variable.Independent("σ", "volatility", np.float32, locator="volatility")
    μ = Variable.Independent("μ", "trend", np.float32, locator="trend")
    i = Variable.Independent("i", "option", Variables.Securities.Option, locator="option")
    j = Variable.Independent("j", "position", Variables.Securities.Position, locator="position")
    k = Variable.Independent("k", "strike", np.float32, locator="strike")
    r = Variable.Constant("r", "interest", np.float32, locator="interest")
    q = Variable.Constant("q", "dividend", np.float32, locator="dividend")


class OptionCalculation(Calculation, ABC):
    def execute(self, options, *args, **kwargs):
        with self.equation(*args, **kwargs) as equation:
            yield equation.y()
            yield equation.Δ()
            yield equation.Γ()
            yield equation.Θ()
            yield equation.P()
            yield equation.V()


class OptionCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__calculation = OptionCalculation(*args, **kwargs)

    def execute(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if self.empty(options): return

        print(options)

        option = self.calculate(options, *args, **kwargs)
        settlements = self.groups(option, by=Querys.Settlement)
        settlements = ",".join(list(map(str, settlements)))
        size = self.size(option)
        self.console(f"{str(settlements)}[{int(size):.0f}]")

        print(options)
        raise Exception()

        if self.empty(option): return
        yield option

    def calculate(self, options, *args, **kwargs):
        options = self.calculation(options, *args, **kwargs)
        assert isinstance(options, pd.DataFrame)
        return options

    @property
    def calculation(self): return self.__calculation

