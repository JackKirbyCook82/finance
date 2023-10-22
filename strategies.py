# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Strategy Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from enum import IntEnum
from collections import namedtuple as ntuple

from support.pipelines import Calculator
from support.calculations import Calculation, feed, equation, calculation
from support.dispatchers import kwargsdispatcher
from finance.securities import Securities, Instruments, Positions, Calculations

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Strategy", "Strategies", "StrategyCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


d = lambda τ, σ: ((0.5 * σ ** 2) * τ) / (σ * np.sqrt(τ))
c = lambda k, s, τ, σ: (s * stats.norm.cdf(np.log(s / k) + d(τ, σ), 0.0, 1.0) - k * stats.norm.cdf(np.log(s / k) - d(τ, σ), 0.0, 1.0))
p = lambda k, s, τ, σ: (k * stats.norm.cdf(-np.log(s / k) + d(τ, σ), 0.0, 1.0) - s * stats.norm.cdf(-np.log(s / k) - d(τ, σ), 0.0, 1.0))


Spreads = IntEnum("Strategy", ["STRANGLE", "COLLAR", "VERTICAL", "CONDOR"], start=1)
class Strategy(ntuple("Strategy", "spread instrument position")):
    def __new__(cls, spread, instrument, position, *args, **kwargs): return super().__new__(cls, spread, instrument, position)
    def __init__(self, *args, securities, **kwargs): self.__securities = securities
    def __str__(self): return "|".join([str(value.name).lower() for value in self if bool(value)])

    @property
    def securities(self): return self.__securities


class StrategiesMeta(type):
    def __getitem__(cls, string):
        strategy = "".join([str(value).title() for value in str(string).split("|")])
        return getattr(cls, strategy)


class Strategies(metaclass=StrategiesMeta):
    StrangleLong = Strategy(Spreads.STRANGLE, 0, Positions.LONG, securities=[Securities.Option.Put.Long, Securities.Option.Call.Long])
    CollarLong = Strategy(Spreads.COLLAR, 0, Positions.LONG, securities=[Securities.Option.Put.Long, Securities.Option.Call.Short, Securities.Stock.Long])
    CollarShort = Strategy(Spreads.COLLAR, 0, Positions.SHORT, securities=[Securities.Option.Call.Long, Securities.Option.Put.Short, Securities.Stock.Short])
    VerticalPut = Strategy(Spreads.VERTICAL, Instruments.PUT, 0, securities=[Securities.Option.Put.Long, Securities.Option.Put.Short])
    VerticalCall = Strategy(Spreads.VERTICAL, Instruments.CALL, 0, securities=[Securities.Option.Call.Long, Securities.Option.Call.Short])
    Condor = Strategy(Spreads.CONDOR, 0, 0, securities=[Securities.Option.Put.Long, Securities.Option.Call.Long, Securities.Option.Put.Short, Securities.Option.Call.Short])


class StrategyCalculation(Calculation):
    io = equation("time", np.datetime64, domain=(), function=lambda : )
    xo = equation("size", np.int32, domain=(), function=lambda : )
    yo = equation("volume", np.int64, domain=(), function=lambda : )
    zo = equation("interest", np.int32, domain=(), function=lambda : )


class StrangleCalculation(StrategyCalculation): pass
class VerticalCalculation(StrategyCalculation): pass
class CollarCalculation(StrategyCalculation): pass

class StrangleLongCalculation(StrangleCalculation):
    pα = calculation(str(Securities.Option.Put.Long), Calculations.Option.Put.Long)
    cα = calculation(str(Securities.Option.Call.Long), Calculations.Option.Call.Long)

    wo = equation("spot", np.float32, domain=(), function=lambda : )
    vo = equation("value", np.float32, domain=(), function=lambda : )
    vmx = equation("max", np.float32, domain=(), function=lambda : )
    vmn = equation("min", np.float32, domain=(), function=lambda : )

class VerticalPutCalculation(VerticalCalculation):
    pα = calculation(str(Securities.Option.Put.Long), Calculations.Option.Put.Long)
    pβ = calculation(str(Securities.Option.Put.Short), Calculations.Option.Put.Short)

    wo = equation("spot", np.float32, domain=(), function=lambda : )
    vo = equation("value", np.float32, domain=(), function=lambda : )
    vmx = equation("max", np.float32, domain=(), function=lambda : )
    vmn = equation("min", np.float32, domain=(), function=lambda : )

class VerticalCallCalculation(VerticalCalculation):
    cα = calculation(str(Securities.Option.Call.Long), Calculations.Option.Call.Long)
    cβ = calculation(str(Securities.Option.Call.Short), Calculations.Option.Call.Short)

    wo = equation("spot", np.float32, domain=(), function=lambda : )
    vo = equation("value", np.float32, domain=(), function=lambda : )
    vmx = equation("max", np.float32, domain=(), function=lambda : )
    vmn = equation("min", np.float32, domain=(), function=lambda : )

class CollarLongCalculation(CollarCalculation):
    pα = calculation(str(Securities.Option.Put.Long), Calculations.Option.Put.Long)
    cβ = calculation(str(Securities.Option.Call.Short), Calculations.Option.Call.Short)

    wo = equation("spot", np.float32, domain=(), function=lambda : )
    vo = equation("value", np.float32, domain=(), function=lambda : )
    vmx = equation("max", np.float32, domain=(), function=lambda : )
    vmn = equation("min", np.float32, domain=(), function=lambda : )

class CollarShortCalculation(CollarCalculation):
    cα = calculation(str(Securities.Option.Call.Long), Calculations.Option.Call.Long)
    pβ = calculation(str(Securities.Option.Put.Short), Calculations.Option.Put.Short)

    wo = equation("spot", np.float32, domain=(), function=lambda : )
    vo = equation("value", np.float32, domain=(), function=lambda : )
    vmx = equation("max", np.float32, domain=(), function=lambda : )
    vmn = equation("min", np.float32, domain=(), function=lambda : )


calculations = {}
class StrategyCalculator(Calculator, calculations=calculations):
    def execute(self, contents, *args, partition=None, **kwargs):
        ticker, expire, dataframes = contents
        assert isinstance(dataframes, dict)
        assert all([isinstance(security, pd.DataFrame) for security in dataframes.values()])
        function = lambda security, dataframe: self.parser(dataframe, *args, instrument=security.instrument, position=security.position, partition=partition, **kwargs)
        securities = {security: function(security, dataframe) for security, dataframe in dataframes.items()}
        for calculation in iter(self.calculations):
            strategy = calculation.strategy
            strategies = calculation(securities, *args, **kwargs)
            if strategies is None:
                continue
            yield ticker, expire, strategy, strategies

    @kwargsdispatcher("instrument")
    def parser(self, dataset, *args, security, position, **kwargs): pass

    @parser.register.value(Instruments.STOCK)
    def stocks(self, dataframe, *args, **kwargs):
        dataset = xr.Dataset.from_dataframe(dataframe)
        dataset = dataset.squeeze("ticker").squeeze("date")
        return dataset

    @parser.register.value(Instruments.PUT, Instruments.CALL)
    def options(self, dataframe, *args, security, partition, **kwargs):
        dataset = xr.Dataset.from_dataframe(dataframe)
        dataset = dataset.squeeze("ticker").squeeze("date")
        dataset = dataset.rename({"strike": str(security)})
        dataset["strike"] = dataset[str(security)]
        dataset = dataset.chunk({str(security): partition}) if bool(partition) else dataset
        return dataset



