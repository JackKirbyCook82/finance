# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Strategy Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
import xarray as xr
from enum import IntEnum
from collections import namedtuple as ntuple

from support.pipelines import Calculator
from support.calculations import Calculation, feed, equation
from support.dispatchers import kwargsdispatcher
from finance.securities import Securities, Instruments, Positions

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Strategies", "StrategyCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


Spreads = IntEnum("Strategy", ["STRANGLE", "COLLAR", "VERTICAL", "CONDOR"], start=1)
class Strategy(ntuple("Strategy", "spread instrument position")):
    def __new__(cls, spread, instrument, position, *args, **kwargs): return super().__new__(cls, spread, instrument, position)
    def __init__(self, *args, securities, **kwargs): self.__securities = securities
    def __str__(self): return "|".join([str(value.name).lower() for value in self if bool(value)])

    @property
    def securities(self): return self.__securities


class Strategies:
    StrangleLong = Strategy(Spreads.STRANGLE, 0, Positions.LONG, securities=[Securities.Option.Put.Long, Securities.Option.Call.Long])
    CollarLong = Strategy(Spreads.COLLAR, 0, Positions.LONG, securities=[Securities.Option.Put.Long, Securities.Option.Call.Short, Securities.Stock.Long])
    CollarShort = Strategy(Spreads.COLLAR, 0, Positions.SHOR, securities=[Securities.Option.Call.Long, Securities.Option.Put.Short, Securities.Stock.Short])
    VerticalPut = Strategy(Spreads.VERTICAL, Instruments.PUT, 0, securities=[Securities.Option.Put.Long, Securities.Option.Put.Short])
    VerticalCall = Strategy(Spreads.VERTICAL, Instruments.CALL, 0, securities=[Securities.Option.Call.Long, Securities.Option.Call.Short])
    Condor = Strategy(Spreads.CONDOR, 0, 0, securities=[Securities.Option.Put.Long, Securities.Option.Call.Long, Securities.Option.Put.Short, Securities.Option.Call.Short])


class StrategyCalculation(Calculation):
    wpα = feed("put|long", np.float32, axes="(i)", key=Securities.Option.Put.Long, variable="price")
    wpβ = feed("put|short", np.float32, axes="(j)", key=Securities.Option.Put.Short, variable="price")
    wcα = feed("call|long", np.float32, axes="(k)", key=Securities.Option.Call.Long, variable="price")
    wcβ = feed("call|short", np.float32, axes="(l)", key=Securities.Option.Call.Short, variable="price")
    wsα = feed("stock|long", np.float32, axes="()", key=Securities.Stock.Long, variable="price")
    wsβ = feed("stock|short", np.float32, axes="()", key=Securities.Stock.Short, variable="price")

    kpα = feed("put|long", np.float32, axes="(i)", key=Securities.Option.Put.Long, variable="strike")
    kpβ = feed("put|short", np.float32, axes="(j)", key=Securities.Option.Put.Short, variable="strike")
    kcα = feed("call|long", np.float32, axes="(k)", key=Securities.Option.Call.Long, variable="strike")
    kcβ = feed("call|short", np.float32, axes="(l)", key=Securities.Option.Call.Short, variable="strike")

    def __init_subclass__(cls, *args, strategy, **kwargs):
        cls.__strategy__ = strategy

    def __call__(self, securities, *args, **kwargs):
        assert isinstance(securities, dict)
        if not all([security in securities.keys() for security in self.securities]):
            return
        strategies = self.vo(securities).to_dataset(name="spot")
        strategies["value"] = self.vω(securities)
        return strategies

    @property
    def strategy(self): return self.__class__.__strategy__
    @property
    def securities(self): return self.__class__.__strategy__.securities


class StrangleLongCalculation(StrategyCalculation, strategy=Strategies.StrangleLong):
    vo = equation("spot", np.float32, axes="(i,k)", function=lambda wpα, wcα: - np.add.outer(wpα, wcα))
    vω = equation("val-", np.float32, axes="(i,k)", function=lambda kpα, kcα: + np.maximum(np.add.outer(kpα, -kcα), 0))
    vγ = equation("val+", np.float32, axes="(i,k)", function=lambda kpα, kcα: + np.ones((kpα.shape, kcα.shape)) * np.inf)

class CollarLongCalculation(StrategyCalculation, strategy=Strategies.CollarLong):
    vo = equation("spot", np.float32, axes="(i,l)", function=lambda wpα, wcβ, wsα: - np.add.outer(wpα, -wcβ) - wsα)
    vω = equation("val-", np.float32, axes="(i,l)", function=lambda kpα, kcβ: + np.minimum.outer(kpα, kcβ))
    vγ = equation("val+", np.float32, axes="(i,l)", function=lambda kpα, kcβ: + np.maximum.outer(kpα, kcβ))

class CollarShortCalculation(StrategyCalculation, strategy=Strategies.CollarShort):
    vo = equation("spot", np.float32, axes="(j,k)", function=lambda wpβ, wcα, wsβ: - np.add.outer(-wpβ, wcα) + wsβ)
    vω = equation("val-", np.float32, axes="(j,k)", function=lambda kpβ, kcα: + np.minimum.outer(-kpβ, -kcα))
    vγ = equation("val+", np.float32, axes="(j,k)", function=lambda kpβ, kcα: + np.maximum.outer(-kpβ, -kcα))

class VerticalPutCalculation(StrategyCalculation, strategy=Strategies.VerticalPut):
    vo = equation("spot", np.float32, axes="(i,j)", function=lambda wpα, wpβ: - np.add.outer(wpα, -wpβ))
    vω = equation("val-", np.float32, axes="(i,j)", function=lambda kpα, kpβ: + np.minimum(np.add.outer(kpα, -kpβ), 0))
    vγ = equation("val+", np.float32, axes="(i,j)", function=lambda kpα, kpβ: + np.maximum(np.add.outer(kpα, -kpβ), 0))

class VerticalCallCalculation(StrategyCalculation, strategy=Strategies.VerticalCall):
    vo = equation("spot", np.float32, axes="(k,l)", function=lambda wcα, wcβ: - np.add.outer(wcα, -wcβ))
    vω = equation("val-", np.float32, axes="(k,l)", function=lambda kcα, kcβ: + np.minimum(np.add.outer(-kcα, kcβ), 0))
    vγ = equation("val+", np.float32, axes="(k,l)", function=lambda kcα, kcβ: + np.maximum(np.add.outer(-kcα, kcβ), 0))

class CondorCalculation(StrategyCalculation, strategy=Strategies.Condor):
    vo = equation("spot", np.float32, axes="(i,j,k,l)", function=lambda vp, vc: + np.add(vp[:, :, np.newaxis, np.newaxis], vc[np.newaxis, np.newaxis, :, :]))
    vω = equation("val-", np.float32, axes="(i,j,k,l)", function=lambda oω, fo: + np.maximum(oω, fo))
    vγ = equation("val+", np.float32, axes="(i,j,k,l)", function=lambda oγ, fo: + np.minimum(oγ, fo))

    vp = equation("vp", np.float32, axes="(i,j)", function=lambda wpα, wpβ: - np.add.outer(wpα, -wpβ))
    vc = equation("vc", np.float32, axes="(k,l)", function=lambda wcα, wcβ: - np.add.outer(wcα, -wcβ))

    pω = equation("pω", np.float32, axes="(i,j)", function=lambda kpα, kpβ: + np.add.outer(kpα, -kpβ))
    cω = equation("cω", np.float32, axes="(k,l)", function=lambda kcα, kcβ: + np.add.outer(-kcα, kcβ))
    oω = equation("oω", np.float32, axes="(i,j,k,l)", function=lambda pω, cω: + np.minimum(np.minimum(pω[:, :, np.newaxis, np.newaxis], cω[np.newaxis, np.newaxis, :, :]), 0))

    pγ = equation("pγ", np.float32, axes="(i,j)", function=lambda kpα, kpβ: + np.add.outer(kpα, -kpβ))
    cγ = equation("cγ", np.float32, axes="(k,l)", function=lambda kcα, kcβ: + np.add.outer(-kcα, kcβ))
    oγ = equation("oγ", np.float32, axes="(i,j,k,l)", function=lambda pγ, cγ: + np.minimum(np.minimum(pγ[:, :, np.newaxis, np.newaxis], cγ[np.newaxis, np.newaxis, :, :]), 0))

    fα = equation("fα", np.float32, axes="(i,k)", function=lambda kpα, kcα: np.maximum(np.add.outer(kpα, -kcα), 0))
    fβ = equation("fβ", np.float32, axes="(j,l)", function=lambda kpβ, kcβ: np.maximum(np.add.outer(kpβ, -kcβ), 0))
    fo = equation("fo", np.float32, axes="(i,j,k,l)", function=lambda fα, fβ: np.add(fα[:, np.newaxis, :, np.newaxis], -fβ[np.newaxis, :, np.newaxis, :]))


class StrategyCalculator(Calculator, calculations=list(StrategyCalculation.__subclasses__())):
    def execute(self, contents, *args, partition, **kwargs):
        ticker, expire, securities = contents
        assert isinstance(securities, dict)
        assert all([isinstance(security, pd.DataFrame) for security in securities.values()])
        function = lambda security, dataframe: self.parser(dataframe, *args, instrument=security.instrument, position=security.position, partition=partition, **kwargs)
        securities = {security: function(security, dataframe) for security, dataframe in securities.items()}
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



