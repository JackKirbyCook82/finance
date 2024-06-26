# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""


import logging
import numpy as np
import pandas as pd
from abc import ABC
from scipy.stats import norm
from datetime import datetime as Datetime
from collections import OrderedDict as ODict

from finance.variables import Variables
from support.calculations import Variable, Equation, Calculation
from support.filtering import Filter, Criterion
from support.pipelines import Processor
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityFiles", "SecurityFilter", "SecurityCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


stock_index = {"ticker": str, "instrument": int, "position": int}
stock_columns = {"current": np.datetime64, "price": np.float32, "size": np.float32, "volume": np.float32}
option_index = {"ticker": str, "expire": np.datetime64, "strike": np.float32, "instrument": int, "option": int, "position": int}
option_columns = {"current": np.datetime64, "price": np.float32, "underlying": np.float32, "size": np.float32, "volume": np.float32, "interest": np.float32}
security_parsers = {"current": pd.to_datetime, "expire": pd.to_datetime, "instrument": Variables.Instruments, "option": Variables.Options, "position": Variables.Positions}
security_criterion = {Criterion.FLOOR: {"volume": 25, "interest": 25, "size": 10}}
security_filename = lambda query: "_".join([str(query.ticker).upper(), str(query.expire.strftime("%Y%m%d"))])


class StockFile(File, variable=Variables.Instruments.STOCK, datatype=pd.DataFrame, filename=security_filename, header=option_index | option_columns, parsers=security_parsers): pass
class OptionFile(File, variable=Variables.Instruments.OPTION, datatype=pd.DataFrame, filename=security_filename, header=option_index | option_columns, parsers=security_parsers): pass
class SecurityFilter(Filter, variables=[Variables.Instruments.STOCK, Variables.Instruments.OPTION], criterion=security_criterion): pass


class SecurityEquation(Equation): pass
class OptionEquation(SecurityEquation):
    τi = Variable("τi", "tau", np.int32, function=lambda ti, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(ti, "ns"), "D") / np.timedelta64(1, "D"))
    si = Variable("si", "ratio", np.float32, function=lambda xi, k: np.log(xi / k))
    yi = Variable("yi", "price", np.float32, function=lambda nzr, nzl: nzr - nzl)
    nzr = Variable("nzr", "right", np.float32, function=lambda xi, zr, Θ: xi * Θ * norm.cdf(Θ * zr))
    nzl = Variable("nzl", "left", np.float32, function=lambda k, zl, Θ, D: k * Θ * D * norm.cdf(Θ * zl))
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
    k = Variable("k", "strike", np.float32, position=0, locator="strike")
    Θ = Variable("Θ", "option", np.int32, position=0, locator="option")
    ρ = Variable("ρ", "discount", np.float32, position="discount")


class SecurityCalculation(Calculation, ABC, fields=["instrument"]): pass
class OptionCalculation(SecurityCalculation, instrument=Variables.Instruments.OPTION, equation=OptionEquation):
    def execute(self, exposures, *args, discount, **kwargs):
        invert = lambda position: Variables.Positions(int(Variables.Positions.LONG) + int(Variables.Positions.SHORT) - int(position))
        equation = self.equation(*args, **kwargs)
        yield from iter([exposures["ticker"], exposures["expire"]])
        yield from iter([exposures["instrument"], exposures["option"], exposures["position"].apply(invert), exposures["strike"]])
        yield equation.yi(exposures, discount=discount)
        yield equation.xi(exposures)
        yield equation.ti(exposures)


class SecurityCalculator(Processor):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        calculations = {variables["instrument"]: calculation for variables, calculation in ODict(list(SecurityCalculation)).items()}
        functions = dict(size=kwargs.get("size", lambda cols: np.int32(10)), volume=kwargs.get("volume", lambda cols: np.NaN), interest=kwargs.get("interest", lambda cols: np.NaN))
        self.__calculations = {instrument: calculation(*args, **kwargs) for instrument, calculation in calculations.items()}
        self.__functions = functions

    def execute(self, contents, *args, current, **kwargs):
        exposures, statistics = contents[Variables.Datasets.EXPOSURE], contents[Variables.Technicals.STATISTIC]
        assert isinstance(exposures, pd.DataFrame) and isinstance(statistics, pd.DataFrame) and isinstance(current, Datetime)
        statistics = statistics.where(statistics["date"] == pd.to_datetime(current))
        exposures = pd.merge(exposures, statistics, how="inner", on="ticker")
        exposures = ODict(list(self.calculate(exposures, *args, **kwargs)))
        if not bool(exposures):
            return
        yield contents | exposures

    def calculate(self, exposures, *args, **kwargs):
        for instrument, calculation in self.calculations.items():
            dataframe = calculation(exposures, *args, **kwargs)
            for column, function in self.functions.items():
                dataframe[column] = dataframe.apply(function, axis=1)
            yield instrument, dataframe

    @property
    def calculations(self): return self.__calculations
    @property
    def functions(self): return self.__functions


class SecurityFiles(object):
    Stock = StockFile
    Option = OptionFile



