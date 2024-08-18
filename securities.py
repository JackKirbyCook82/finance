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
from collections import OrderedDict as ODict

from finance.variables import Pipelines, Variables
from support.calculations import Variable, Equation, Calculation
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityFiles", "SecurityFilter", "SecurityCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Variables:
    contract = ["ticker", "expire"]
    dates  = {"current": "%Y%m%d-%H%M", "expire": "%Y%m%d"}
    parsers = {"instrument": Variables.Instruments, "option": Variables.Options, "position": Variables.Positions}
    formatters = {"instrument": int, "option": int, "position": int}
    types = {"ticker": str, "strike": np.float32, "price": np.float32, "underlying": np.float32, "size": np.float32, "volume": np.float32, "interest": np.float32}
    filename = lambda query: "_".join([str(query.ticker).upper(), str(query.expire.strftime("%Y%m%d"))])
    parameters = dict(datatype=pd.DataFrame, filename=filename, dates=dates, parsers=parsers, formatters=formatters, types=types)

class Headers:
    options = ["ticker", "expire", "instrument", "option", "position", "strike", "volume", "size", "interest", "current"]
    stocks = ["ticker", "instrument", "position", "price", "volume", "size", "current"]


contract_variables =
dimensions_variables = {"instrument": False, "option": False, "position": True, "strike": False}
date_variables =
security_parsers =
security_formatters =
security_types =
security_filename =
security_parameters =
option_header =
stock_header =


class StockFile(File, variable=Variables.Instruments.STOCK, header=stock_header, **security_parameters): pass
class OptionFile(File, variable=Variables.Instruments.OPTION, header=option_header, **security_parameters): pass
class SecurityFiles(object): Stock = StockFile; Option = OptionFile


class SecurityFilter(Operations.Filter):
    def processor(self, contents, *args, **kwargs):
        securities = ODict(list(self.securities(contents, *args, **kwargs)))
        if not bool(securities): return
        yield contents | securities

    def securities(self, contents, *args, **kwargs):
        contract = contents[Variables.Querys.CONTRACT]
        for security in list(Variables.Instruments):
            if not bool(security): continue
            securities = self.calculate(contents, *args, contract=contract, **kwargs)
            securities = securities.reset_index(drop=True, inplace=False)
            if bool(securities.empty): continue
            yield security, securities


class BlackScholesEquation(Equation):
    τ = Variable("τ", "tau", np.int32, function=lambda to, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))
    so = Variable("so", "so", np.float32, function=lambda xo, k: np.log(xo / k))
    yo = Variable("yo", "price", np.float32, function=lambda zx, zk, Φ, ε: (zx - zk) * (1 + (Φ * ε)))

    zx = Variable("zx", "zx", np.float32, function=lambda xo, Θ, N1: xo * Θ * N1)
    zk = Variable("zk", "zk", np.float32, function=lambda k, Θ, N2, D: k * Θ * D * N2)
    d1 = Variable("d1", "d1", np.float32, function=lambda so, α, β: (so + α) / β)
    d2 = Variable("d2", "d2", np.float32, function=lambda d1, β: d1 - β)
    N1 = Variable("N1", "N1", np.float32, function=lambda d1, Θ: norm.cdf(Θ * d1))
    N2 = Variable("N2", "N2", np.float32, function=lambda d2, Θ: norm.cdf(Θ * d2))

    α = Variable("α", "alpha", np.float32, function=lambda τ, δ, ρ: (ρ + np.divide(np.power(δ * np.sqrt(252), 2), 2)) * τ / 252)
    β = Variable("β", "beta", np.float32, function=lambda τ, δ: δ * np.sqrt(252) * np.sqrt(τ / 252))
    ε = Variable("ε", "epsilon", np.float32, function=lambda ω: 0.5 * np.sin(ω * 2 * np.pi / 5) + 0.05 * ω)
    D = Variable("D", "discount", np.float32, function=lambda τ, ρ: np.exp(-ρ * τ / 252))
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
    ω = Variable("ω", "offset", np.float32, position="offset")


class BlackScholesCalculation(Calculation, equation=BlackScholesEquation):
    def execute(self, exposures, *args, discount, offset=0, **kwargs):
        invert = lambda position: Variables.Positions(int(Variables.Positions.LONG) + int(Variables.Positions.SHORT) - int(position))
        equation = self.equation(*args, **kwargs)
        yield from iter([exposures[scope] for scope in security_scope])
        yield from iter([exposures[dimension].apply(invert) if inverted else exposures[dimension] for dimension, inverted in security_dimensions.items()])
        yield equation.yo(exposures, discount=discount, offset=offset)
        yield equation.xo(exposures)
        yield equation.to(exposures)


class SecurityCalculator(Pipelines.Processor, title="Calculated"):
    def __init__(self, *args, size, volume=lambda cols: np.NaN, interest=lambda cols: np.NaN, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        calculations = {Variables.Instruments.OPTION: BlackScholesCalculation}
        self.__calculation = {variable: calculations(*args, **kwargs) for variable, calculation in calculations.items()}
        self.__functions = dict(size=size, volume=volume, interest=interest)

#    def processor(self, contents, *args, **kwargs):
#        exposures, statistics = contents[Variables.Datasets.EXPOSURE], contents[Variables.Technicals.STATISTIC]
#        assert isinstance(exposures, pd.DataFrame) and isinstance(statistics, pd.DataFrame)
#        exposures = self.exposures(exposures, statistics, *args, **kwargs)
#        options = self.calculate(exposures, *args, **kwargs)
#        if bool(options.empty): return
#        yield contents | {Variables.Instruments.OPTION: options}

#    def calculate(self, exposures, *args, **kwargs):
#        function = lambda cols: {key: value(cols[key]) for key, value in self.functions.items()}
#        options = self.calculation(exposures, *args, **kwargs)
#        dataframe = options.apply(function, axis=1, result_type="expand")
#        options = pd.concat([options, dataframe], axis=1)
#        return options

#    @staticmethod
#    def exposures(exposures, statistics, *args, current, **kwargs):
#        statistics = statistics.where(statistics["date"] == pd.to_datetime(current))
#        exposures = pd.merge(exposures, statistics, how="inner", on="ticker")
#        return exposures

    @property
    def calculation(self): return self.__calculation
    @property
    def functions(self): return self.__functions



