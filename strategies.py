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
from support.calculations import Calculation, feed, equation
from support.dispatchers import kwargsdispatcher
from finance.securities import Securities, Instruments, Positions

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
#    τau = equation("τau", np.int16, function=lambda tτ, to: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))
#    σ = feed("volatility", np.float16, variable="volatility")
#    tτ = feed("expire", np.datetime64, variable="expire")
#    to = feed("date", np.datetime64, variable="date")

    wpα = feed("put|long|price", np.float32, axes="(i)", key=Securities.Option.Put.Long, variable="price")
    wpβ = feed("put|short|price", np.float32, axes="(j)", key=Securities.Option.Put.Short, variable="price")
    wcα = feed("call|long|price", np.float32, axes="(k)", key=Securities.Option.Call.Long, variable="price")
    wcβ = feed("call|short|price", np.float32, axes="(l)", key=Securities.Option.Call.Short, variable="price")
    wsα = feed("stock|long|price", np.float32, axes="()", key=Securities.Stock.Long, variable="price")
    wsβ = feed("stock|short|price", np.float32, axes="()", key=Securities.Stock.Short, variable="price")

    kpα = feed("put|long|strike", np.float32, axes="(i)", key=Securities.Option.Put.Long, variable="strike")
    kpβ = feed("put|short|strike", np.float32, axes="(j)", key=Securities.Option.Put.Short, variable="strike")
    kcα = feed("call|long|strike", np.float32, axes="(k)", key=Securities.Option.Call.Long, variable="strike")
    kcβ = feed("call|short|strike", np.float32, axes="(l)", key=Securities.Option.Call.Short, variable="strike")

    tpα = feed("put|long|time", np.float32, axes="(i)", key=Securities.Option.Put.Long, variable="time")
    tpβ = feed("put|short|time", np.float32, axes="(j)", key=Securities.Option.Put.Short, variable="time")
    tcα = feed("call|long|time", np.float32, axes="(k)", key=Securities.Option.Call.Long, variable="time")
    tcβ = feed("call|short|time", np.float32, axes="(l)", key=Securities.Option.Call.Short, variable="time")
    tsα = feed("stock|long|time", np.float32, axes="()", key=Securities.Stock.Long, variable="time")
    tsβ = feed("stock|short|time", np.float32, axes="()", key=Securities.Stock.Short, variable="time")

    xpα = feed("put|long|size", np.float32, axes="(i)", key=Securities.Option.Put.Long, variable="size")
    xpβ = feed("put|short|size", np.float32, axes="(j)", key=Securities.Option.Put.Short, variable="size")
    xcα = feed("call|long|size", np.float32, axes="(k)", key=Securities.Option.Call.Long, variable="size")
    xcβ = feed("call|short|size", np.float32, axes="(l)", key=Securities.Option.Call.Short, variable="size")
    xsα = feed("stock|long|size", np.float32, axes="()", key=Securities.Stock.Long, variable="size")
    xsβ = feed("stock|short|size", np.float32, axes="()", key=Securities.Stock.Short, variable="size")

    ypα = feed("put|long|volume", np.int64, axes="(i)", key=Securities.Option.Put.Long, variable="volume")
    ypβ = feed("put|short|volume", np.int64, axes="(j)", key=Securities.Option.Put.Short, variable="volume")
    ycα = feed("call|long|volume", np.int64, axes="(k)", key=Securities.Option.Call.Long, variable="volume")
    ycβ = feed("call|short|volume", np.int64, axes="(l)", key=Securities.Option.Call.Short, variable="volume")
    ysα = feed("stock|long|volume", np.int64, axes="()", key=Securities.Stock.Long, variable="volume")
    ysβ = feed("stock|short|volume", np.int64, axes="()", key=Securities.Stock.Short, variable="volume")

    zpα = feed("put|long|interest", np.int32, axes="(i)", key=Securities.Option.Put.Long, variable="interest")
    zpβ = feed("put|short|interest", np.int32, axes="(j)", key=Securities.Option.Put.Short, variable="interest")
    zcα = feed("call|long|interest", np.int32, axes="(k)", key=Securities.Option.Call.Long, variable="interest")
    zcβ = feed("call|short|interest", np.int32, axes="(l)", key=Securities.Option.Call.Short, variable="interest")

    wpμ = equation("put|average|price", np.float32, axes="(i,j)", function=lambda wpα, wpβ: + np.average(wpα, wpβ))
    wcμ = equation("call|average|price", np.float32, axes="(k,l)", function=lambda wcα, wcβ: + np.average(wcα, wcβ))
    wsμ = equation("stock|average|price", np.float32, axes="()", function=lambda wsα, wsβ: + np.average(wsα, wsβ))

    vpα = equation("put|long|value", np.float32, axes="(i)", function=lambda kpα, wsμ, τau, σ: p(kpα, wsμ, τau, σ))
    vpβ = equation("put|short|value", np.float32, axes="(j)", function=lambda kpβ, wsμ, τau, σ: p(kpβ, wsμ, τau, σ))
    vcα = equation("call|long|value", np.float32, axes="(k)", function=lambda kcα, wsμ, τau, σ: c(kcα, wsμ, τau, σ))
    vcβ = equation("call|short|value", np.float32, axes="(l)", function=lambda kcβ, wsμ, τau, σ: c(kcβ, wsμ, τau, σ))

    def __init_subclass__(cls, *args, strategy, **kwargs):
        cls.__strategy__ = strategy

    def __call__(self, securities, *args, fees, **kwargs):
        assert isinstance(securities, dict)
        if not all([security in securities.keys() for security in self.securities]):
            return
        strategies = self.to(securities).to_dataset(name="time")
        strategies["spot"] = self.wo(securities) * 100 - fees
        strategies["future"] = self.vl(securities) * 100 - fees
        return strategies

    @property
    def strategy(self): return self.__class__.__strategy__
    @property
    def securities(self): return self.__class__.__strategy__.securities


class StrangleLongCalculation(StrategyCalculation, strategy=Strategies.StrangleLong):
    to = equation("time", np.datetime64, axes="(i,k)", function=lambda tpα, tcα: np.maximum.outer(tpα, tcα))
    xo = equation("size", np.float32, axes="(i,k)", function=lambda xpα, xcα: np.minimum.outer(xpα, xcα))
    yo = equation("volume", np.int64, axes="(i,k)", function=lambda ypα, ycα: np.minimum.outer(ypα, ycα))
    zo = equation("interest", np.int32, axes="(i,k)", function=lambda zpα, zcα: np.minimum.outer(zpα, zcα))

    wo = equation("spot", np.float32, axes="(i,k)", function=lambda wpα, wcα: - np.add.outer(wpα, wcα))
    vo = equation("value", np.float32, axes="(i,k)", function=lambda vpα, vpβ: np.add.outer(vpα, vpβ))
    vl = equation("lower", np.float32, axes="(i,k)", function=lambda kpα, kcα: + np.maximum(np.add.outer(kpα, -kcα), 0))
    vu = equation("upper", np.float32, axes="(i,k)", function=lambda kpα, kcα: + np.ones((kpα.shape, kcα.shape)) * np.inf)

class VerticalPutCalculation(StrategyCalculation, strategy=Strategies.VerticalPut):
    to = equation("time", np.datetime64, axes="(i,j)", function=lambda tpα, tpβ: np.maximum.outer(tpα, tpβ))
    xo = equation("size", np.float32, axes="(i,j)", function=lambda xpα, xpβ: np.minimum.outer(xpα, xpβ))
    yo = equation("volume", np.int64, axes="(i,j)", function=lambda ypα, ypβ: np.minimum.outer(ypα, ypβ))
    zo = equation("interest", np.int32, axes="(i,j)", function=lambda zpα, zpβ: np.minimum.outer(zpα, zpβ))

    wo = equation("spot", np.float32, axes="(i,j)", function=lambda wpα, wpβ: - np.add.outer(wpα, -wpβ))
    vo = equation("value", np.float32, axes="(j,k)", function=lambda vpα, vpβ: np.add.outer(vpα, -vpβ))
    vl = equation("lower", np.float32, axes="(i,j)", function=lambda kpα, kpβ: + np.minimum(np.add.outer(kpα, -kpβ), 0))
    vu = equation("upper", np.float32, axes="(i,j)", function=lambda kpα, kpβ: + np.maximum(np.add.outer(kpα, -kpβ), 0))

class VerticalCallCalculation(StrategyCalculation, strategy=Strategies.VerticalCall):
    to = equation("time", np.datetime64, axes="(k,l)", function=lambda tcα, tcβ: np.maximum.outer(tcα, tcβ))
    xo = equation("size", np.float32, axes="(k,l)", function=lambda xcα, xcβ: np.minimum.outer(xcα, xcβ))
    yo = equation("volume", np.int64, axes="(k,l)", function=lambda ycα, ycβ: np.minimum.outer(ycα, ycβ))
    zo = equation("interest", np.int32, axes="(k,l)", function=lambda zcα, zcβ: np.minimum.outer(zcα, zcβ))

    wo = equation("spot", np.float32, axes="(k,l)", function=lambda wcα, wcβ: - np.add.outer(wcα, -wcβ))
    vo = equation("value", np.float32, axes="(j,k)", function=lambda vcα, vcβ: np.add.outer(vcα, -vcβ))
    vl = equation("lower", np.float32, axes="(k,l)", function=lambda kcα, kcβ: + np.minimum(np.add.outer(-kcα, kcβ), 0))
    vu = equation("upper", np.float32, axes="(k,l)", function=lambda kcα, kcβ: + np.maximum(np.add.outer(-kcα, kcβ), 0))

class CollarLongCalculation(StrategyCalculation, strategy=Strategies.CollarLong):
    to = equation("time", np.datetime64, axes="(i,l)", function=lambda tpα, tcβ, tsα: np.maximum(np.maximum.outer(tpα, tcβ), tsα))
    xo = equation("size", np.float32, axes="(i,l)", function=lambda xpα, xcβ, xsα: np.minimum(np.minimum.outer(xpα, xcβ), xsα))
    yo = equation("volume", np.int64, axes="(i,l)", function=lambda ypα, ycβ, ysα: np.minimum(np.minimum.outer(ypα, ycβ), ysα))
    zo = equation("interest", np.int32, axes="(i,l)", function=lambda zpα, zcβ: np.minimum.outer(zpα, zcβ))

    wo = equation("spot", np.float32, axes="(i,l)", function=lambda wpα, wcβ, wsα: - np.add.outer(wpα, -wcβ) - wsα)
    vo = equation("value", np.float32, axes="(i,k)", function=lambda vpα, vcβ: np.add.outer(vpα, -vcβ))
    vl = equation("lower", np.float32, axes="(i,l)", function=lambda kpα, kcβ: + np.minimum.outer(kpα, kcβ))
    vu = equation("upper", np.float32, axes="(i,l)", function=lambda kpα, kcβ: + np.maximum.outer(kpα, kcβ))

class CollarShortCalculation(StrategyCalculation, strategy=Strategies.CollarShort):
    to = equation("time", np.datetime64, axes="(j,k)", function=lambda tpβ, tcα, tsβ: np.maximum(np.maximum.outer(tpβ, tcα), tsβ))
    xo = equation("size", np.float32, axes="(j,k)", function=lambda xpβ, xcα, xsβ: np.minimum(np.minimum.outer(xpβ, xcα), xsβ))
    yo = equation("volume", np.int64, axes="(j,k)", function=lambda ypβ, ycα, ysβ: np.minimum(np.minimum.outer(ypβ, ycα), ysβ))
    zo = equation("interest", np.int32, axes="(j,k)", function=lambda zpβ, zcα: np.minimum.outer(zpβ, zcα))

    wo = equation("spot", np.float32, axes="(j,k)", function=lambda wpβ, wcα, wsβ: - np.add.outer(-wpβ, wcα) + wsβ)
    vo = equation("value", np.float32, axes="(j,k)", function=lambda vpβ, vcα: np.add.outer(-vpβ, vcα))
    vl = equation("lower", np.float32, axes="(j,k)", function=lambda kpβ, kcα: + np.minimum.outer(-kpβ, -kcα))
    vu = equation("upper", np.float32, axes="(j,k)", function=lambda kpβ, kcα: + np.maximum.outer(-kpβ, -kcα))

class CondorCalculation(StrategyCalculation, strategy=Strategies.Condor):
    to = equation("time", np.datetime64, axes="(i,j,k,l)", function=lambda tpα, tpβ, tcα, tcβ: np.maximum.outer(np.maximum.outer(tpα, tpβ), np.maximum.outer(tcα, tcβ)))
    xo = equation("size", np.float32, axes="(i,j,k,l)", function=lambda xpα, xpβ, xcα, xcβ: np.minimum.outer(np.minimum.outer(xpα, xpβ), np.minimum.outer(xcα, xcβ)))
    yo = equation("volume", np.int64, axes="(i,j,k,l)", function=lambda ypα, ypβ, ycα, ycβ: np.minimum.outer(np.minimum.outer(ypα, ypβ), np.minimum.outer(ycα, ycβ)))
    zo = equation("interest", np.int32, axes="(i,j,k,l)", function=lambda zpα, zpβ, zcα, zcβ: np.minimum.outer(np.maxminimumimum.outer(zpα, zpβ), np.minimum.outer(zcα, zcβ)))

    wo = equation("spot", np.float32, axes="(i,j,k,l)", function=lambda vp, vc: + np.add(vp[:, :, np.newaxis, np.newaxis], vc[np.newaxis, np.newaxis, :, :]))
    vo = equation("value", np.float32, axes="(i,j,k,l)", function=lambda vpα, vpβ, vcα, vcβ: np.add.outer(np.add.outer(vpα, -vpβ), np.add.outer(vcα, -vcβ)))
    vl = equation("lower", np.float32, axes="(i,j,k,l)", function=lambda ol, fo: + np.maximum(ol, fo))
    vu = equation("upper", np.float32, axes="(i,j,k,l)", function=lambda ou, fo: + np.minimum(ou, fo))

    vp = equation("vp", np.float32, axes="(i,j)", function=lambda wpα, wpβ: - np.add.outer(wpα, -wpβ))
    vc = equation("vc", np.float32, axes="(k,l)", function=lambda wcα, wcβ: - np.add.outer(wcα, -wcβ))

    pl = equation("pl", np.float32, axes="(i,j)", function=lambda kpα, kpβ: + np.add.outer(kpα, -kpβ))
    cl = equation("cl", np.float32, axes="(k,l)", function=lambda kcα, kcβ: + np.add.outer(-kcα, kcβ))
    ol = equation("ol", np.float32, axes="(i,j,k,l)", function=lambda pl, cl: + np.minimum(np.minimum(pl[:, :, np.newaxis, np.newaxis], cl[np.newaxis, np.newaxis, :, :]), 0))

    pu = equation("pu", np.float32, axes="(i,j)", function=lambda kpα, kpβ: + np.add.outer(kpα, -kpβ))
    cu = equation("cu", np.float32, axes="(k,l)", function=lambda kcα, kcβ: + np.add.outer(-kcα, kcβ))
    ou = equation("ou", np.float32, axes="(i,j,k,l)", function=lambda pu, cu: + np.minimum(np.minimum(pu[:, :, np.newaxis, np.newaxis], cu[np.newaxis, np.newaxis, :, :]), 0))

    fα = equation("fα", np.float32, axes="(i,k)", function=lambda kpα, kcα: np.maximum(np.add.outer(kpα, -kcα), 0))
    fβ = equation("fβ", np.float32, axes="(j,l)", function=lambda kpβ, kcβ: np.maximum(np.add.outer(kpβ, -kcβ), 0))
    fo = equation("fo", np.float32, axes="(i,j,k,l)", function=lambda fα, fβ: np.add(fα[:, np.newaxis, :, np.newaxis], -fβ[np.newaxis, :, np.newaxis, :]))


calculations = {calculation.__strategy__: calculation for calculation in list(StrategyCalculation.__subclasses__())}
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



