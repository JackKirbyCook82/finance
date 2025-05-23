# -*- coding: utf-8 -*-
"""
Created on Tues May 13 2025
@name:   Securities Objects
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
__all__ = ["SecurityCalculator"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


class SecurityEquation(Equation, ABC, datatype=pd.Series, vectorize=True):
    Θ = Variable.Dependent("Θ", "theta", np.float32, function=lambda vx, vk, zx, σ, q, r, τ, i: (vx * int(i) * q - vk * int(i) * r - norm.pdf(zx) * σ / np.exp(q * τ) / np.sqrt(τ) / 2) / 364)
    Δ = Variable.Dependent("Δ", "delta", np.float32, function=lambda zx, q, τ, i: + norm.cdf(zx * int(i)) * int(i) / np.exp(q * τ))
    Γ = Variable.Dependent("Γ", "gamma", np.float32, function=lambda zx, x, σ, q, τ: norm.pdf(zx) / np.exp(q * τ) / np.sqrt(τ) / x / σ)
    P = Variable.Dependent("P", "rho", np.float32, function=lambda zk, k, r, τ, i: - norm.cdf(zk * int(i)) * int(i) * k * τ / np.exp(r * τ))
    V = Variable.Dependent("V", "vega", np.float32, function=lambda zx, x, σ, q, τ: norm.pdf(zx) * np.sqrt(τ) * x / np.exp(q * τ))

    τ = Variable.Dependent("τ", "tau", np.float32, function=lambda to, tτ: (np.datetime64(tτ, "ns") - np.datetime64(to, "ns")) / np.timedelta64(364, 'D'))
    v = Variable.Dependent("v", "value", np.float32, function=lambda vx, vk: (vx - vk))
    vx = Variable.Dependent("yx", "underlying", np.float32, function=lambda zx, x, τ, q, i: x * int(i) * norm.cdf(zx * int(i)) / np.exp(q * τ))
    vk = Variable.Dependent("yk", "strike", np.float32, function=lambda zk, k, τ, r, i: k * int(i) * norm.cdf(zk * int(i)) / np.exp(r * τ))

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
    k = Variable.Independent("k", "strike", np.float32, locator="strike")
    r = Variable.Constant("r", "interest", np.float32, locator="interest")
    q = Variable.Constant("q", "dividend", np.float32, locator="dividend")

    def execute(self, *args, **kwargs):
        yield self.v()
        yield self.Δ()
        yield self.Γ()
        yield self.Θ()
        yield self.P()
        yield self.V()


class SecurityCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__calculation = Calculation[pd.Series](*args, equation=SecurityEquation, **kwargs)
        self.__sizing = {Variables.Securities.Position.LONG: "supply", Variables.Securities.Position.SHORT: "demand"}
        self.__pricing = {Variables.Securities.Position.LONG: "ask", Variables.Securities.Position.SHORT: "bid"}
        self.__factors = {Variables.Securities.Position.LONG: + 1, Variables.Securities.Position.SHORT: - 1}
        self.__greeks = ("value", "delta", "gamma", "theta", "rho", "vega")

    def execute(self, stocks, options, technicals, *args, **kwargs):
        assert isinstance(stocks, pd.DataFrame) and isinstance(options, pd.DataFrame) and isinstance(technicals, pd.DataFrame)
        if self.empty(options): return
        querys = self.keys(options, by=Querys.Settlement)
        querys = ",".join(list(map(str, querys)))
        options = self.calculate(stocks, options, technicals, *args, **kwargs)
        size = self.size(options)
        self.console(f"{str(querys)}[{int(size):.0f}]")
        if self.empty(options): return

        print(options)
        raise Exception()

        yield options

    def calculate(self, stocks, options, technicals, *args, **kwargs):
        assert isinstance(stocks, pd.DataFrame) and isinstance(options, pd.DataFrame) and isinstance(technicals, pd.DataFrame)
        options = self.technicals(options, technicals, *args, **kwargs)
        options = self.stocks(options, stocks, *args, **kwargs)
        greeks = self.calculation(options, *args, **kwargs)
        assert isinstance(greeks, pd.DataFrame)
        options = pd.concat([options, greeks], axis=1)
        long = self.position(options, *args, position=Variables.Securities.Position.LONG, **kwargs)
        short = self.position(options, *args, position=Variables.Securities.Position.SHORT, **kwargs)
        options = pd.concat([long, short], axis=0)
        options = options.reset_index(drop=True, inplace=False)
        return options

    def position(self, options, *args, position, **kwargs):
        function = lambda column, factor: lambda series: series[column] * factor
        size = lambda series: series[self.sizing[position]]
        price = lambda series: series[self.pricing[position]]
        greeks = {column: function(column, self.factors[position]) for column in self.greeks}
        contents = dict(instrument=Variables.Securities.Instrument.OPTION, position=position, size=size, price=price)
        options = options.assign(**contents, **greeks)
        return options.drop(["ask", "bid", "supply", "demand"], axis=1, inplace=False)

    @staticmethod
    def technicals(options, technicals, *args, **kwargs):
        assert isinstance(options, pd.DataFrame) and isinstance(technicals, pd.DataFrame)
        mask = technicals["date"] == technicals["date"].max()
        technicals = technicals.where(mask).dropna(how="all", inplace=False)[["ticker", "trend", "volatility"]]
        return options.merge(technicals, how="left", on=list(Querys.Symbol), sort=False, suffixes=("", "_"))

    @staticmethod
    def stocks(options, stocks, *args, **kwargs):
        assert isinstance(options, pd.DataFrame) and isinstance(stocks, pd.DataFrame)
        underlying = lambda series: (series["ask"] * series["supply"] + series["bid"] * series["demand"]) / (series["supply"] + series["demand"])
        underlying = pd.concat([stocks["ticker"], stocks.apply(underlying, axis=1).rename("underlying")], axis=1)
        return options.merge(underlying, how="left", on=list(Querys.Symbol), sort=False, suffixes=("", "_"))

    @property
    def calculation(self): return self.__calculation
    @property
    def factors(self): return self.__factors
    @property
    def pricing(self): return self.__pricing
    @property
    def sizing(self): return self.__sizing
    @property
    def greeks(self): return self.__greeks



