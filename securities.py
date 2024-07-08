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
security_parameters = dict(datatype=pd.DataFrame, filename=security_filename, dates=security_dates, parsers=security_parsers, formatters=security_formatters, types=security_types)
stock_header = ["current", "ticker", "instrument", "position", "price", "volume", "size"]
option_header = ["current", "ticker", "expire", "instrument", "option", "position", "strike", "volume", "size", "interest"]


class StockFile(File, variable=Variables.Instruments.STOCK, header=stock_header, **security_parameters): pass
class OptionFile(File, variable=Variables.Instruments.OPTION, header=option_header, **security_parameters): pass
class SecurityFiles(object): Stock = StockFile; Option = OptionFile


class SecurityFilter(Filter, variables=[Variables.Instruments.STOCK, Variables.Instruments.OPTION]):
    pass


class BlackScholesEquation(Equation):
    τi = Variable("τi", "tau", np.int32, function=lambda ti, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(ti, "ns"), "D") / np.timedelta64(1, "D"))
    si = Variable("si", "ratio", np.float32, function=lambda xi, ki: np.log(xi / ki))
    yi = Variable("yi", "price", np.float32, function=lambda nzr, nzl: nzr - nzl)
    Θi = Variable("Θ", "theta", np.int32, function=lambda i: int(Variables.Theta(str(i))))
    nzr = Variable("nzr", "right", np.float32, function=lambda xi, zr, Θ: xi * Θ * norm.cdf(Θ * zr))
    nzl = Variable("nzl", "left", np.float32, function=lambda ki, zl, Θ, D: ki * Θ * D * norm.cdf(Θ * zl))
    zr = Variable("zr", "right", np.float32, function=lambda α, β: α + β)
    zl = Variable("zl", "left", np.float32, function=lambda α, β: α - β)
    α = Variable("α", "alpha", np.float32, function=lambda si, A, B: (si / B) + (A / B))
    β = Variable("β", "beta", np.float32, function=lambda B: (B ** 2) / (B * 2))
    A = Variable("A", "alpha", np.float32, function=lambda τi, ρ: np.multiply(τi, ρ))
    B = Variable("B", "beta", np.float32, function=lambda τi, δi: np.multiply(np.sqrt(τi), δi))
    D = Variable("D", "discount", np.float32, function=lambda τi, ρ: np.power(np.exp(τi * ρ), -1))

    δi = Variable("δi", "volatility", np.float32, position=0, locator="volatility")
    xi = Variable("xi", "underlying", np.float32, position=0, locator="price")
    tτ = Variable("tτ", "expire", np.datetime64, position=0, locator="expire")
    ti = Variable("ti", "current", np.datetime64, position=0, locator="date")
    ki = Variable("ki", "strike", np.float32, position=0, locator="strike")
    i = Variable("i", "option", Variables.Options, position=0, locator="option")
    ρ = Variable("ρ", "discount", np.float32, position="discount")


class BlackScholesCalculation(Calculation, equation=BlackScholesEquation):
    def execute(self, exposures, *args, discount, **kwargs):
        invert = lambda position: Variables.Positions(int(Variables.Positions.LONG) + int(Variables.Positions.SHORT) - int(position))
        equation = self.equation(*args, **kwargs)
        yield from iter([exposures["ticker"], exposures["expire"]])
        yield from iter([exposures["instrument"], exposures["option"], exposures["position"].apply(invert), exposures["strike"]])
        yield equation.yi(exposures, discount=discount)
        yield equation.xi(exposures)
        yield equation.ti(exposures)


class SecurityCalculator(Processor):
    def __init__(self, *args, size, volume, interest, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__calculation = BlackScholesCalculation(*args, **kwargs)
        self.__functions = dict(size=size, volume=volume, interest=interest)

    def execute(self, contents, *args, current, **kwargs):
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




