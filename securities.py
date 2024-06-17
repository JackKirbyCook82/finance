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

from finance.variables import Querys, Variables
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


stock_index = {"ticker": str, "instrument": int, "position": int}
stock_columns = {"current": np.datetime64, "price": np.float32, "size": np.float32, "volume": np.float32}
stock_parsers = {"instrument": lambda x: Variables.Instruments(int(x)), "position": lambda x: Variables.Positions(int(x))}
option_index = {"ticker": str, "expire": np.datetime64, "strike": np.float32, "instrument": int, "position": int}
option_columns = {"current": np.datetime64, "price": np.float32, "underlying": np.float32, "size": np.float32, "volume": np.float32, "interest": np.float32}
option_parsers = {"instrument": lambda x: Variables.Instruments(int(x)), "position": lambda x: Variables.Positions(int(x))}


class StockFile(File, variable=Variables.Instruments.STOCK, query=Querys.Contract, datatype=pd.DataFrame, header=stock_index | stock_columns, parsers=stock_parsers): pass
class OptionFile(File, variable=Variables.Instruments.OPTION, query=Querys.Contract, datatype=pd.DataFrame, header=option_index | option_columns, parsers=option_parsers): pass


class SecurityFilter(Filter, variables=[Variables.Instruments.STOCK, Variables.Instruments.OPTION], query=Querys.Contract):
    pass


class SecurityEquation(Equation): pass
class OptionEquation(SecurityEquation):
    τi = Variable("tau", "tau", np.int32, function=lambda ti, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(ti, "ns"), "D") / np.timedelta64(1, "D"))
    si = Variable("si", "ratio", np.float32, function=lambda xi, k: np.log(xi / k))
    yi = Variable("yi", "price", np.float32, function=lambda nzr, nzl: nzr - nzl)
    nzr = Variable("nr", "right", np.float32, function=lambda xi, zr, θ: xi * θ * norm(θ * zr))
    nzl = Variable("nl", "left", np.float32, function=lambda k, zl, θ, D: k * θ * D * norm(θ * zl))
    zr = Variable("zr", "right", np.float32, function=lambda α, β: α + β)
    zl = Variable("zl", "left", np.float32, function=lambda α, β: α - β)
    α = Variable("α", "alpha", np.float32, function=lambda si, A, B: (si / B) + (A, B))
    β = Variable("β", "beta", np.float32, function=lambda B: (B ** 2) / (B * 2))
    A = Variable("A", "alpha", np.float32, function=lambda τi, ρ: np.multiply(τi, ρ))
    B = Variable("B", "beta", np.float32, function=lambda τi, δi: np.multiply(np.sqrt(τi), δi))
    D = Variable("D", "discount", np.float32, function=lambda τi, ρ: np.power(np.exp(τi * ρ), -1))

    δi = Variable("δ", "volatility", np.float32, position=0, locator="volatility")
    xi = Variable("xi", "underlying", np.float32, position=0, locator="price")
    tτ = Variable("tτ", "expire", np.datetime64, position=0, locator="expire")
    ti = Variable("ti", "current", np.datetime64, position=0, locator="date")
    k = Variable("k", "strike", np.float32, position=0, locator="strike")
    ρ = Variable("ρ", "discount", np.float32, position="discount")
    Θ = Variable("Θ", "instrument", np.int32, position=0, locator="theta")
    Φ = Variable("Φ", "position", np.int32, position=0, locator="phi")


class SecurityCalculation(Calculation, ABC, fields=["instrument"]): pass
class OptionCalculation(Calculation, instrument=Variables.Instruments.OPTION, equation=OptionEquation):
    def execute(self, exposures, *args, discount, **kwargs):
        invert = lambda position: Variables.Positions(int(Variables.Positions.LONG) + int(Variables.Positions.SHORT) - int(position))
        equation = self.equation(*args, **kwargs)
        yield from iter([exposures["ticker"], exposures["expire"], exposures["strike"]])
        yield equation.Φ(exposures).apply(invert)
        yield equation.Θ(exposures)
        yield equation.ti(exposures)
        yield equation.yi(exposures, discount=discount)
        yield equation.xi(exposures)


class SecurityCalculator(Processor):
    def __init__(self, *args, calculations=[], name=None, size, volume, interest, **kwargs):
        assert isinstance(calculations, list) and all([instrument in list(Variables.Instruments) for instrument in calculations])
        assert all([callable(function) for function in (size, volume, interest)])
        super().__init__(*args, name=name, **kwargs)
        calculations = {variables["instrument"]: calculation for variables, calculation in ODict(list(SecurityCalculation)).items() if variables["instrument"] in calculations}
        self.__calculations = {str(instrument.name).lower(): calculation(*args, **kwargs) for instrument, calculation in calculations.items()}
        self.__functions = dict(size=size, volume=volume, interest=interest)

    def execute(self, contents, *args, **kwargs):
        statistics, exposures = contents["statistic"], contents["exposure"]
        assert isinstance(statistics, pd.DataFrame) and isinstance(exposures, pd.DataFrame)
        securities = ODict(list(self.calculate(exposures, statistics, *args, **kwargs)))
        yield contents | securities

    def calculate(self, exposures, statistics, *args, current, **kwargs):
        assert isinstance(current, Datetime)
        statistics = statistics.where(statistics["date"] == current.date())
        exposures = pd.merge(exposures, statistics, how="inner", on="ticker")
        for security, calculation in self.calculations.items():
            dataframe = calculation(exposures, *args, **kwargs)
            for column, function in self.functions.items():
                dataframe[column] = dataframe.apply(function)
            yield security, dataframe

    @property
    def calculations(self): return self.__calculations
    @property
    def functions(self): return self.__functins


class SecurityFiles(object):
    Stock = StockFile
    Options = OptionFile



