# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""


import logging
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime as Datetime

from finance.variables import Variables
from support.calculations import Variable, Equation, Calculation
from support.pipelines import Processor
from support.filtering import Filter
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityFiles", "SecurityFilter", "SecurityCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


security_dates = {"current": "%Y%m%d-%H%M", "expire": "%Y%m%d"}
security_parsers = {"instrument": Variables.Instruments, "option": Variables.Options, "position": Variables.Positions}
security_formatters = {"instrument": int, "option": int, "position": int}
security_types = {"ticker": str, "strike": np.float32, "price": np.float32, "underlying": np.float32, "size": np.float32, "volume": np.float32, "interest": np.float32}
security_filename = lambda query: "_".join([str(query.ticker).upper(), str(query.expire.strftime("%Y%m%d"))])
security_formatter = lambda self, *, results, elapsed, **kw: f"{str(self.title)}: {repr(self)}|{str(results[Variables.Querys.CONTRACT])}[{elapsed:.02f}s]"
security_parameters = dict(datatype=pd.DataFrame, filename=security_filename, dates=security_dates, parsers=security_parsers, formatters=security_formatters, types=security_types)
stock_header = ["current", "ticker", "instrument", "position", "price", "volume", "size"]
option_header = ["current", "ticker", "expire", "instrument", "option", "position", "strike", "volume", "size", "interest"]


class StockFile(File, variable=Variables.Instruments.STOCK, header=stock_header, **security_parameters): pass
class OptionFile(File, variable=Variables.Instruments.OPTION, header=option_header, **security_parameters): pass
class SecurityFiles(object): Stock = StockFile; Option = OptionFile


class SecurityFilter(Filter, variables=[Variables.Instruments.STOCK, Variables.Instruments.OPTION], formatter=security_formatter):
    pass


class BlackScholesEquation(Equation):
    τ = Variable("τ", "tau", np.int32, function=lambda to, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))
    so = Variable("so", "so", np.float32, function=lambda xo, k: np.log(xo / k))
    yo = Variable("yo", "price", np.float32, function=lambda zx, zk, F: (zx - zk) * F)

    zx = Variable("zx", "zx", np.float32, function=lambda xo, Θ, N1: xo * Θ * N1)
    zk = Variable("zk", "zk", np.float32, function=lambda k, Θ, N2, D: k * Θ * D * N2)
    d1 = Variable("d1", "d1", np.float32, function=lambda so, α, β: (so + α) / β)
    d2 = Variable("d2", "d2", np.float32, function=lambda d1, β: d1 - β)
    N1 = Variable("N1", "N1", np.float32, function=lambda d1, Θ: norm.cdf(Θ * d1))
    N2 = Variable("N2", "N2", np.float32, function=lambda d2, Θ: norm.cdf(Θ * d2))

    α = Variable("α", "α", np.float32, function=lambda τ, δ, ρ: (ρ + np.divide(np.power(δ * np.sqrt(252), 2), 2)) * τ / 252)
    β = Variable("β", "β", np.float32, function=lambda τ, δ: δ * np.sqrt(252) * np.sqrt(τ / 252))
    D = Variable("D", "discount", np.float32, function=lambda τ, ρ: np.exp(-ρ * τ / 252))
    F = Variable("F", "factor", np.float32, function=lambda Φ, ε: 1 + (Φ * ε))
    Θ = Variable("Θ", "theta", np.int32, function=lambda i: + int(Variables.Theta(str(i))))
    Φ = Variable("Φ", "phi", np.int32, function=lambda j: + int(Variables.Phi(str(j))))

    tτ = Variable("tτ", "expire", np.datetime64, position=0, locator="expire")
    to = Variable("to", "current", np.datetime64, position=0, locator="date")
    xo = Variable("xo", "underlying", np.float32, position=0, locator="price")

    δ = Variable("δ", "volatility", np.float32, position=0, locator="volatility")
    i = Variable("i", "option", Variables.Options, position=0, locator="option")
    j = Variable("j", "position", Variables.Positions, position=0, locator="position")
    k = Variable("k", "strike", np.float32, position=0, locator="strike")
    ρ = Variable("ρ", "discount", np.float32, position="discount")
    ε = Variable("ε", "factor", np.float32, position="factor")


class BlackScholesCalculation(Calculation, equation=BlackScholesEquation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        function = kwargs.get("factor", lambda count: 0 * count)
        callable(function)
        self.__function = function
        self.__count = 0

    def execute(self, exposures, *args, discount, **kwargs):
        invert = lambda position: Variables.Positions(int(Variables.Positions.LONG) + int(Variables.Positions.SHORT) - int(position))
        equation = self.equation(*args, **kwargs)
        yield from iter([exposures["ticker"], exposures["expire"]])
        yield from iter([exposures["instrument"], exposures["option"], exposures["position"].apply(invert), exposures["strike"]])
        yield equation.yo(exposures, discount=discount, factor=self.factor)
        yield equation.xo(exposures)
        yield equation.to(exposures)
        self.count += 1

    @property
    def factor(self): return self.function(self.count)
    @property
    def function(self): return self.__function
    @property
    def count(self): return self.__count
    @count.setter
    def count(self, count): self.__count = count


class SecurityCalculator(Processor, formatter=security_formatter):
    def __init__(self, *args, size, volume=lambda cols: np.NaN, interest=lambda cols: np.NaN, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__calculation = BlackScholesCalculation(*args, **kwargs)
        self.__functions = dict(size=size, volume=volume, interest=interest)

    def processor(self, contents, *args, current, **kwargs):
        exposures, statistics = contents[Variables.Datasets.EXPOSURE], contents[Variables.Technicals.STATISTIC]
        assert isinstance(exposures, pd.DataFrame) and isinstance(statistics, pd.DataFrame) and isinstance(current, Datetime)
        statistics = statistics.where(statistics["date"] == pd.to_datetime(current))
        exposures = pd.merge(exposures, statistics, how="inner", on="ticker")
        options = self.calculate(exposures, *args, **kwargs)
        if bool(options.empty):
            return
        options = {Variables.Instruments.OPTION: options}
        yield contents | options

    def calculate(self, exposures, *args, **kwargs):
        dataframe = self.calculation(exposures, *args, **kwargs)
        for column, function in self.functions.items():
            dataframe[column] = dataframe.apply(function, axis=1)
        return dataframe

    @property
    def calculation(self): return self.__calculation
    @property
    def functions(self): return self.__functions




