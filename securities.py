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
from support.pipelines import Processor
from support.filtering import Filter
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["OptionFiles", "OptionFilter", "OptionCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


option_index = {"ticker": str, "expire": np.datetime64, "strike": np.float32, "instrument": int, "option": int, "position": int}
option_columns = {"current": np.datetime64, "price": np.float32, "underlying": np.float32, "size": np.float32, "volume": np.float32, "interest": np.float32}
option_parsers = {"instrument": lambda x: Variables.Instruments(int(x)), "option": lambda x: Variables.Options(int(x)), "position": lambda x: Variables.Positions(int(x))}
option_filename = lambda query: "_".join([str(query.ticker).upper(), str(query.expire.strftime("%Y%m%d"))])


class OptionFile(File, variable=Variables.Assets.OPTION, datatype=pd.DataFrame, filename=option_filename, header=option_index | option_columns, parsers=option_parsers): pass
class OptionFilter(Filter, variables=[Variables.Assets.OPTION]): pass


class OptionEquation(Equation):
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


class OptionCalculation(Calculation, equation=OptionEquation):
    def execute(self, exposures, *args, discount, **kwargs):
        invert = lambda position: Variables.Positions(int(Variables.Positions.LONG) + int(Variables.Positions.SHORT) - int(position))
        equation = self.equation(*args, **kwargs)
        yield from iter([exposures["ticker"], exposures["expire"]])
        yield from iter([exposures["instrument"], exposures["option"], exposures["position"].apply(invert), exposures["strike"]])
        yield equation.yi(exposures, discount=discount)
        yield equation.xi(exposures)
        yield equation.ti(exposures)


class OptionCalculator(Processor):
    def __init__(self, *args, name=None, size, volume, interest, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__functions = dict(size=size, volume=volume, interest=interest)
        self.__calculation = OptionCalculation(*args, **kwargs)

    def execute(self, contents, *args, current, **kwargs):
        statistics, exposures = contents["statistic"], contents["exposure"]
        assert isinstance(statistics, pd.DataFrame) and isinstance(exposures, pd.DataFrame)
        assert isinstance(current, Datetime)
        statistics = statistics.where(statistics["date"] == pd.to_datetime(current))
        exposures = pd.merge(exposures, statistics, how="inner", on="ticker")
        options = self.calculation(exposures, *args, **kwargs)
        for column, function in self.functions.items():
            options[column] = options.apply(function, axis=1)
        yield contents | {"options": options}

    @property
    def calculation(self): return self.__calculation
    @property
    def functions(self): return self.__functions


class OptionFiles(object):
    Option = OptionFile



