# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 2024
@name:   Technical Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC, ABCMeta
from datetime import date as Date

from finance.concepts import Querys
from calculations import Equation, Variables, Algorithms, Computations
from support.mixins import Emptying, Sizing, Partition, Logging
from support.meta import AttributeMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TechnicalCalculator", "TechnicalEquation"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"


class TechnicalEquationMeta(AttributeMeta, type(Equation), ABCMeta): pass
class TechnicalEquation(Computations.Table, Algorithms.UnVectorized.Table, Equation, ABC, metaclass=TechnicalEquationMeta):
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

    def __init__(self, *args, period=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.period = period

    def execute(self, bars, /, **kwargs):
        yield from super().execute(bars)
        yield self.s(bars)
        yield self.t(bars)
        yield self.x(bars)
        yield self.v(bars)


class StateEquations(TechnicalEquation, ABC): pass
class TrendEquations(TechnicalEquation, ABC): pass
class MomentumEquations(TechnicalEquation, ABC): pass
class VolatilityEquations(TechnicalEquation, ABC): pass
class VolumeEquations(TechnicalEquation, ABC): pass


class BarsEquation(StateEquations, ABC, attribute="BARS"):
    def execute(self, bars, /, **kwargs):
        yield from super().execute(bars, **kwargs)
        yield self.xo(bars)
        yield self.xc(bars)
        yield self.xl(bars)
        yield self.xh(bars)
        yield self.x(bars)
        yield self.v(bars)

class StatsEquation(StateEquations, ABC, attribute="STATS"):
    δ = Variables.Dependent("δ", "volatility", np.float32, function=lambda xr, *, dt: xr.rolling(dt).std())
    μ = Variables.Dependent("μ", "trend", np.float32, function=lambda xr, *, dt: xr.rolling(dt).mean())

    def execute(self, bars, /, **kwargs):
        yield from super().execute(bars, **kwargs)
        parameters = dict(period=self.period)
        yield self.δ(bars, **parameters)
        yield self.μ(bars, **parameters)
        yield self.x(bars)


class SMAEquation(TrendEquations, ABC, attribute="SMA"):
    sma = Variables.Dependent("sma", "SMA", np.float32, function=lambda x, *, dt: x.rolling(window=dt).mean())

    def execute(self, bars, /, **kwargs):
        yield from super().execute(bars, **kwargs)
        parameters = dict(period=self.period)
        yield self.sma(bars, **parameters)

class EMAEquation(TrendEquations, ABC, attribute="EMA"):
    ema = Variables.Dependent("ema", "EMA", np.float32, function=lambda x, *, dt: x.ewm(span=dt, min_periods=dt, adjust=False).mean())

    def execute(self, bars, /, **kwargs):
        yield from super().execute(bars, **kwargs)
        parameters = dict(period=self.period)
        yield self.ema(bars, **parameters)

class MACDEquation(TrendEquations, ABC, attribute="MACD"):
    ema12 = Variables.Dependent("ema12", "EMA12", np.float32, function=lambda x: x.ewm(span=12, min_periods=12, adjust=False).mean())
    ema26 = Variables.Dependent("ema26", "EMA26", np.float32, function=lambda x: x.ewm(span=26, min_periods=26, adjust=False).mean())
    macd = Variables.Dependent("macd", "MACD", np.float32, function=lambda ema12, ema26: ema12 - ema26)
    sign = Variables.Dependent("sign", "SIGN", np.float32, function=lambda macd: macd.ewm(span=9, min_periods=9, adjust=False).mean())
    hist = Variables.Dependent("hist", "HIST", np.float32, function=lambda macd, sign: macd - sign)

    def execute(self, bars, **kwargs):
        yield from super().execute(bars, **kwargs)
        yield self.macd(bars)
        yield self.sign(bars)
        yield self.hist(bars)


class RSIEquation(MomentumEquations, ABC, attribute="RSI"):
    gain = Variables.Dependent("gain", "GAIN", np.float32, function=lambda dx: dx.where(dx > 0, 0))
    loss = Variables.Dependent("loss", "LOSS", np.float32, function=lambda dx: dx.where(dx < 0, 0))
    smg14 = Variables.Dependent("smg14", "SMG14", np.float32, function=lambda gain, *, dt: gain.rolling(window=dt).mean())
    sml14 = Variables.Dependent("sml14", "SML14", np.float32, function=lambda loss, *, dt: loss.rolling(window=dt).mean())
    rs = Variables.Dependent("rs", "RS", np.float32, function=lambda smg14, sml14: smg14 / sml14)
    rsi = Variables.Dependent("rsi", "RSI", np.float32, function=lambda rs: 100 - (100 / (1 + rs)))

    def execute(self, bars, **kwargs):
        yield from super().execute(bars, **kwargs)
        parameters = dict(period=self.period)
        yield self.rsi(bars, **parameters)


class BBEquation(VolatilityEquations, ABC, attribute="BB"):
    sma20 = Variables.Dependent("sma20", "SMA20", np.float32, function=lambda x, *, dt: x.rolling(window=dt).mean())
    smd20 = Variables.Dependent("smd20", "SMD20", np.float32, function=lambda x, *, dt: x.rolling(window=dt).std())
    bbh = Variables.Dependent("bbh", "BBH", np.float32, function=lambda sma20, smd20: sma20 + 2 * smd20)
    bbl = Variables.Dependent("bbl", "BBL", np.float32, function=lambda sma20, smd20: sma20 - 2 * smd20)

    def execute(self, bars, **kwargs):
        yield from super().execute(bars, **kwargs)
        parameters = dict(period=self.period)
        yield self.bbh(bars, **parameters)
        yield self.bbl(bars, **parameters)

class ATREquation(VolatilityEquations, ABC, attribute="ATR"):
    xhl = Variables.Dependent("xhl", "XHL", np.float32, function=lambda xh, xl: xh - xl)
    xhc = Variables.Dependent("xhc", "XHC", np.float32, function=lambda xc, xh: (xh - xc.shift(1)).abs())
    xlc = Variables.Dependent("xlc", "XLC", np.float32, function=lambda xc, xl: (xl - xc.shift(1)).abs())
    xtr = Variables.Dependent("xtr", "XTR", np.float32, function=lambda xhl, xhc, xlc: pd.concat([xhl, xhc, xlc], axis=1).max(axis=1))
    atr = Variables.Dependent("atr", "ATR", np.float32, function=lambda xtr, *, dt: xtr.ewm(alpha=1/dt, adjust=False).mean())

    def execute(self, bars, **kwargs):
        yield from super().execute(bars, **kwargs)
        parameters = dict(period=self.period)
        yield self.atr(bars, **parameters)


class MFIEquation(VolumeEquations, ABC, attribute="MFI"):
    typ = Variables.Dependent("typ", "TYP", np.float32, function=lambda xc, xl, xh: (xc + xl + xh) / 3)
    rmf = Variables.Dependent("rmf", "RMF", np.float32, function=lambda typ, v: typ * v)
    pmf = Variables.Dependent("pmf", "PMF", np.float32, function=lambda typ, rmf: rmf.where(typ.diff() > 0, 0))
    nmf = Variables.Dependent("nmf", "NMF", np.float32, function=lambda typ, rmf: rmf.where(typ.diff() < 0, 0))
    mfr = Variables.Dependent("mfr", "MFI", np.float32, function=lambda pmf, nmf, *, dt: pmf.rolling(dt).sum() / nmf.rolling(dt).sum())
    mfi = Variables.Dependent("mfi", "MFI", np.float32, function=lambda mfr: 100 - (100 / (1 + mfr)))

    def execute(self, bars, **kwargs):
        yield from super().execute(bars, **kwargs)
        parameters = dict(period=self.period)
        yield self.mfi(bars, **parameters)

class CMFEquation(VolumeEquations, ABC, attribute="CMF"):
    mfm = Variables.Dependent("mfm", "MFM", np.float32, function=lambda xc, xl, xh: (((xc - xl) - (xh - xc)) / (xh - xl)).replace([np.inf, -np.inf], 0).fillna(0))
    mfv = Variables.Dependent("mfv", "MFV", np.float32, function=lambda mfm, v: mfm * v)
    cmf = Variables.Dependent("cmf", "CMF", np.float32, function=lambda mfv, v, *, dt: mfv.rolling(window=dt).sum() / v.rolling(window=dt).sum())

    def execute(self, bars, **kwargs):
        yield from super().execute(bars, **kwargs)
        parameters = dict(period=self.period)
        yield self.cmf(bars, **parameters)

class OBVEquation(VolumeEquations, ABC, attribute="OBV"):
    rvf = Variables.Dependent("rvf", "RVF", np.float32, function=lambda dx, v: (np.sign(dx) * v).fillna(0).cumsum())
    mvf = Variables.Dependent("mvf", "MVF", np.float32, function=lambda rvf: rvf.abs().max())
    obv = Variables.Dependent("obv", "OBV", np.float32, function=lambda rvf, mvf: 100 * rvf / mvf)

    def execute(self, bars, **kwargs):
        yield from super().execute(bars, **kwargs)
        yield self.obv(bars)


class TechnicalCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, equations, **kwargs):
        assert all([isinstance(equation, TechnicalEquation) for equation in equations])
        super().__init__(*args, **kwargs)
        self.__equations = equations

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

    def calculator(self, bars, /, **kwargs):
        assert isinstance(bars, list) and all([isinstance(dataframe, pd.DataFrame) for dataframe in bars])
        for dataframe in bars:
            assert (dataframe["ticker"].to_numpy()[0] == dataframe["ticker"]).all()
            dataframe = dataframe.sort_values("date", ascending=True, inplace=False)
            technicals = [equation(dataframe) for equation in self.equations]
            assert all([isinstance(technical, pd.DataFrame) for technical in technicals])
            technicals = pd.concat(technicals, axis=1)
            technicals = technicals.loc[:, ~technicals.columns.duplicated()]
            yield technicals

    @property
    def equations(self): return self.__equations

