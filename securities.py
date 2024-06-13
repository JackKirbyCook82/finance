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
from datetime import datetime as Datetime
from collections import OrderedDict as ODict

from finance.variables import Querys, Variables
from support.calculations import Variable, Equation, Calculation
from support.pipelines import Processor
from support.filtering import Filter
from support.parsers import Header
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityFiles", "SecurityHeaders", "SecurityFilter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


stocks_index = {"ticker": str, "instrument": str, "position": str}
stocks_columns = {"current": np.datetime64, "price": np.float32, "size": np.float32, "volume": np.float32}
options_index = {"ticker": str, "expire": np.datetime64, "strike": np.float32, "instrument": str, "position": str}
options_columns = {"current": np.datetime64, "price": np.float32, "underlying": np.float32, "size": np.float32, "volume": np.float32, "interest": np.float32}


class StockFile(File, variable=Variables.Instruments.STOCK, query=Querys.Contract, datatype=pd.DataFrame, header=stocks_index | stocks_columns): pass
class OptionFile(File, variable=Variables.Instruments.OPTION, query=Querys.Contract, datatype=pd.DataFrame, header=options_index | options_columns): pass
class StockHeader(Header, variable=Variables.Instruments.STOCK, datatype=pd.DataFrame, axes={"index": stocks_index, "columns": stocks_columns}): pass
class OptionHeader(Header, variable=Variables.Instruments.OPTION, datatype=pd.DataFrame, axes={"index": options_index, "columns": options_columns}): pass
class SecurityFilter(Filter, variables=[Variables.Instruments.STOCK, Variables.Instruments.OPTION], query=Querys.Contract): pass


class SecurityEquation(Equation): pass
class OptionEquation(SecurityEquation):
    τ = Variable("tau", "tau", np.int32, function=lambda ti, tτ: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(ti, "ns"), "D") / np.timedelta64(1, "D"))
    xδ = Variable("xδ", "volatility", np.float32, position=0, locator="volatility")
    xμ = Variable("xμ", "trend", np.float32, position=0, locator="trend")
    xi = Variable("xi", "underlying", np.float32, position=0, locator="underlying")
    tτ = Variable("tτ", "expire", np.datetime64, position=0, locator="expire")
    ti = Variable("ti", "date", np.datetime64, position=0, locator="date")
    k = Variable("k", "strike", np.float32, position=0, locator="strike")
    ρ = Variable("ρ", "discount", np.float32, position="discount")


class SecurityCalculation(Calculation, ABC, fields=["instrument"]): pass
class OptionCalculation(Calculation, instrument=Variables.Instruments.OPTION, equation=OptionEquation):
    def execute(self, exposures, *args, discount, **kwargs):
        equation = self.equation(*args, **kwargs)


class SecurityCalculator(Processor):
    def __init__(self, *args, calculations=[], name=None, **kwargs):
        assert isinstance(calculations, list) and all([instrument in list(Variables.Instruments) for instrument in calculations])
        super().__init__(*args, name=name, **kwargs)
        calculations = {variables["instrument"]: calculation for variables, calculation in ODict(list(SecurityCalculation)).items() if variables["instrument"] in calculations}
        self.__calculations = {str(instrument.name).lower(): calculation(*args, **kwargs) for instrument, calculation in calculations.items()}

    def execute(self, contents, *args, **kwargs):
        statistics, exposures = contents["statistic"], contents["exposure"]
        assert isinstance(statistics, pd.DataFrame) and isinstance(exposures, pd.DataFrame)
        securities = ODict(list(self.calculate(exposures, statistics, *args, **kwargs)))
        yield contents | securities

    def calculate(self, exposures, statistics, *args, current, **kwargs):
        assert isinstance(current, Datetime)
        statistics = statistics.where(statistics["date"] == current.date())
        exposures = pd.merge(exposures, statistics, how="inner", on=["ticker", "date"])
        exposures = exposures.rename({"price": "underlying"})
        for security, calculation in self.calculations.items():
            dataframe = calculation(exposures, *args, **kwargs)
            yield security, dataframe

    @property
    def calculations(self): return self.__calculations


class SecurityFiles(object):
    Stock = StockFile
    Options = OptionFile

class SecurityHeaders(object):
    Stock = StockHeader
    Options = OptionHeader



