# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 2024
@name:   Technical Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC
from datetime import date as Date

from finance.concepts import Concepts, Querys
from calculations import Equation, Variables, Algorithms, Computations
from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TechnicalCalculator"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"


class TechnicalEquation(Computations.Table, Algorithms.UnVectorized.Table, Equation, ABC, root=True):
    xr = Variables.Dependent("xr", "return", np.float32, function=lambda x: x.pct_change(1))
    dx = Variables.Dependent("dx", "change", np.float32, function=lambda x: x.diff())

    xo = Variables.Independent("xo", "open", np.float32, locator="open")
    xc = Variables.Independent("xc", "close", np.float32, locator="close")
    xl = Variables.Independent("xl", "low", np.float32, locator="low")
    xh = Variables.Independent("xh", "high", np.float32, locator="high")

    x = Variables.Independent("x", "adjusted", np.float32, locator="adjusted")
    v = Variables.Independent("v", "volume", np.int64, locator="volume")
    s = Variables.Independent("s", "ticker", Date, locator="ticker")
    t = Variables.Independent("t", "date", Date, locator="date")
    dt = Variables.Constant("dt", "period", np.int32, locator="period")

    def execute(self, bars, /, period, **kwargs):
        parameters = dict(period=period)
        yield from super().execute(bars, **parameters)
        yield self.s(bars, **parameters)
        yield self.x(bars, **parameters)
        yield self.t(bars, **parameters)
        yield self.v(bars, **parameters)

class BarsEquation(TechnicalEquation, ABC, register=Concepts.Technical.BARS):
    def execute(self, bars, /, **kwargs):
        yield from super().execute(bars, **kwargs)
        yield self.xo(bars)
        yield self.xc(bars)
        yield self.xl(bars)
        yield self.xh(bars)

class StatsEquation(TechnicalEquation, ABC, register=Concepts.Technical.STATS):
    δ = Variables.Dependent("δ", "volatility", np.float32, function=lambda xr, *, dt: xr.rolling(dt).std())
    μ = Variables.Dependent("μ", "trend", np.float32, function=lambda xr, *, dt: xr.rolling(dt).mean())

    def execute(self, bars, /, period, **kwargs):
        parameters = dict(period=period)
        yield from super().execute(bars, **parameters, **kwargs)
        yield self.δ(bars, **parameters)
        yield self.μ(bars, **parameters)

class SMAEquation(TechnicalEquation, ABC, register=Concepts.Technical.SMA):
    sma = Variables.Dependent("sma", "SMA", np.float32, function=lambda x, *, dt: x.rolling(window=dt).mean())

    def execute(self, bars, /, period, **kwargs):
        parameters = dict(period=period)
        yield from super().execute(bars, **parameters, **kwargs)
        yield self.sma(bars, **parameters)

class EMAEquation(TechnicalEquation, ABC, register=Concepts.Technical.EMA):
    ema = Variables.Dependent("ema", "EMA", np.float32, function=lambda x, *, dt: x.ewm(span=dt, adjust=False).mean())

    def execute(self, bars, /, period, **kwargs):
        parameters = dict(period=period)
        yield from super().execute(bars, **parameters, **kwargs)
        yield self.ema(bars, **parameters)

class MACDEquation(TechnicalEquation, ABC, register=Concepts.Technical.MACD):
    ema12 = Variables.Dependent("ema12", "EMA12", np.float32, function=lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = Variables.Dependent("ema26", "EMA26", np.float32, function=lambda x: x.ewm(span=26, adjust=False).mean())
    macd = Variables.Dependent("macd", "MACD", np.float32, function=lambda ema12, ema26: ema12 - ema26)
    sign = Variables.Dependent("sign", "SIGN", np.float32, function=lambda macd: macd.ewm(span=9, adjust=False).mean())
    hist = Variables.Dependent("hist", "HIST", np.float32, function=lambda macd, sign: macd - sign)

    def execute(self, bars, **kwargs):
        yield from super().execute(bars, **kwargs)
        yield self.macd(bars)
        yield self.sign(bars)
        yield self.hist(bars)

class RSIEquation(TechnicalEquation, ABC, register=Concepts.Technical.RSI):
    gain = Variables.Dependent("gain", "GAIN", np.float32, function=lambda dx: dx.where(dx > 0, 0))
    loss = Variables.Dependent("loss", "LOSS", np.float32, function=lambda dx: dx.where(dx < 0, 0))
    smg14 = Variables.Dependent("smg14", "SMG14", np.float32, function=lambda gain: gain.rolling(window=14).mean())
    sml14 = Variables.Dependent("sml14", "SML14", np.float32, function=lambda loss: loss.rolling(window=14).mean())
    rs = Variables.Dependent("rs", "RS", np.float32, function=lambda smg14, sml14: smg14 / sml14)
    rsi = Variables.Dependent("rsi", "RSI", np.float32, function=lambda rs: 100 - (100 / (1 + rs)))

    def execute(self, bars, **kwargs):
        yield from super().execute(bars, **kwargs)
        yield self.ris(bars)

class BBEquation(TechnicalEquation, ABC, register=Concepts.Technical.BB):
    sma20 = Variables.Dependent("sma20", "SMA20", np.float32, function=lambda x: x.rolling(window=20).mean())
    smd20 = Variables.Dependent("smd20", "SMD20", np.float32, function=lambda x: x.rolling(window=20).std())
    bbh = Variables.Dependent("bbh", "BBH", np.float32, function=lambda sma20, smd20: sma20 + 2 * smd20)
    bbl = Variables.Dependent("bbl", "BBL", np.float32, function=lambda sma20, smd20: sma20 - 2 * smd20)

    def execute(self, bars, **kwargs):
        yield from super().execute(bars, **kwargs)
        yield self.bbh(bars)
        yield self.bbl(bars)

class MFIEquation(TechnicalEquation, ABC, register=Concepts.Technical.MFI):
    typ = Variables.Dependent("typ", "TYP", np.float32, function=lambda xc, xl, xh: (xc + xl + xh) / 3)
    rmf = Variables.Dependent("rmf", "RMF", np.float32, function=lambda typ, v: typ * v)
    pmf = Variables.Dependent("pmf", "PMF", np.float32, function=lambda typ, rmf: np.where(typ > 0, rmf.diff(), 0))
    nmf = Variables.Dependent("nmf", "NMF", np.float32, function=lambda typ, rmf: np.where(typ < 0, rmf.diff(), 0))
    mfr = Variables.Dependent("mfi", "MFI", np.float32, function=lambda pmf, nmf: pmf.rolling(14).sum() / nmf.rolling(14).sum())
    mfi = Variables.Dependent("mfi", "MFI", np.float32, function=lambda mfr: 100 - (100 / (1 + mfr)))

    def execute(self, bars, **kwargs):
        yield from super().execute(bars, **kwargs)
        yield self.mfi(bars)


class TechnicalCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, technicals, **kwargs):
        assert all([technical in list(Concepts.Technical) for technical in technicals])
        super().__init__(*args, **kwargs)
        equations = [equation for technical, equation in iter(TechnicalEquation) if technical in technicals]
        self.__equation = (TechnicalEquation & equations)(*args, **kwargs)

    def execute(self, bars, /, **kwargs):
        assert isinstance(bars, pd.DataFrame)
        if self.empty(bars): return
        symbols = self.keys(bars, by=Querys.Symbol)
        symbols = ",".join(list(map(str, symbols)))
        technicals = self.calculate(bars, **kwargs)
        size = self.size(technicals)
        self.console(f"{str(symbols)}[{int(size):.0f}]")
        if self.empty(technicals): return
        yield technicals

    def calculate(self, bars, /, **kwargs):
        assert isinstance(bars, pd.DataFrame)
        bars = list(self.values(bars, by=Querys.Symbol))
        technicals = list(self.calculator(bars, **kwargs))
        technicals = pd.concat(technicals, axis=0)
        technicals = technicals.reset_index(drop=True, inplace=False)
        return technicals

    def calculator(self, bars, /, period, **kwargs):
        assert isinstance(bars, list) and all([isinstance(dataframe, pd.DataFrame) for dataframe in bars])
        for dataframe in bars:
            assert (dataframe["ticker"].to_numpy()[0] == dataframe["ticker"]).all()
            dataframe = dataframe.sort_values("date", ascending=True, inplace=False)
            technicals = self.equation(dataframe, period=period)
            assert isinstance(technicals, pd.DataFrame)
            yield technicals

    @property
    def equation(self): return self.__equation



