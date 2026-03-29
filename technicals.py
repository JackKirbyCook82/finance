# -*- coding: utf-8 -*-
"""
Created on Thurs Mar 26 2026
@name:   Technical Objects
@author: Jack Kirby Cook

"""

import types
import numpy as np
import pandas as pd
from abc import ABC

from finance.concepts import Concepts
from support.calculations import Calculation
from support.meta import RegistryMeta
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TechnicalCalculator"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class TechnicalCalculatorMeta(type(Calculation), RegistryMeta):
    def __call__(cls, *args, technicals=None, **kwargs):
        assert isinstance(technicals, (list, types.NoneType))
        if technicals is None: return super(TechnicalCalculatorMeta, cls).__call__(*args, **kwargs)
        subclasses = [cls[technical] for technical in technicals]
        bases = tuple(subclasses + [cls])
        newcls = type(cls.__name__, bases, {})
        instance = super(TechnicalCalculatorMeta, newcls).__call__(*args, **kwargs)
        return instance


class TechnicalCalculator(Calculation, Logging, ABC, metaclass=TechnicalCalculatorMeta):
    pctgains = lambda adjusted: adjusted.pct_change(1)
    netgains = lambda adjusted: adjusted.diff()

    def __call__(self, bars, *args, **kwargs):
        assert isinstance(bars, pd.DataFrame)
        if bool(bars.empty): return bars
        technicals = self.calculator(bars, *args, **kwargs)
        technicals = pd.concat(list(technicals), axis=0)
        technicals = technicals.sort_values(by=["ticker", "date"], ascending=[True, False], inplace=False)
        return technicals

    def calculator(self, bars, *args, **kwargs):
        bars["date"] = pd.to_datetime(bars["date"])
        for ticker, bars in bars.groupby("ticker"):
            bars = bars.sort_values(by="date", ascending=True)
            technicals = self.calculate(bars, *args, **kwargs)
            self.alert(technicals)
            yield technicals

    def alert(self, dataframe):
        instrument = str(Concepts.Securities.Instrument.STOCK).title()
        tickers = "|".join(list(dataframe["ticker"].unique()))
        self.console("Calculated", f"{str(instrument)}[{str(tickers)}, {len(dataframe):.0f}]")


class StateCalculator(TechnicalCalculator): pass
class BarsCalculator(StateCalculator, variables=["ticker", "date", "adjusted"], register=Concepts.Technicals.State.BARS): pass
class StatsCalculator(StateCalculator, register=Concepts.Technicals.State.STATS):
    volatility = lambda pctgains, *, period: pctgains.rolling(period).std()
    trend = lambda pctgains, *, period: pctgains.rolling(period).mean()


class TrendCalculator(TechnicalCalculator): pass
class SMACalculator(TrendCalculator, register=Concepts.Technicals.Trend.SMA):
    sma = lambda adjusted, *, period: adjusted.rolling(window=period).mean()

class EMACalculator(TrendCalculator, register=Concepts.Technicals.Trend.EMA):
    ema = lambda adjusted, *, period: adjusted.ewm(span=period, min_periods=period, adjust=False).mean()

class MACDCalculator(TrendCalculator, variables=["macd", "sign", "hist"], register=Concepts.Technicals.Trend.MACD):
    ema12 = lambda adjusted: adjusted.ewm(span=12, min_periods=12, adjust=False).mean()
    ema26 = lambda adjusted: adjusted.ewm(span=26, min_periods=26, adjust=False).mean()
    macd = lambda ema12, ema26: ema12 - ema26
    sign = lambda macd: macd.ewm(span=9, min_periods=9, adjust=False).mean()
    hist = lambda macd, sign: macd - sign


class MomentumCalculator(TechnicalCalculator): pass
class RSICalculator(MomentumCalculator, variables=["rsi"], register=Concepts.Technicals.Momentum.RSI):
    gain = lambda netgains: netgains.where(netgains > 0, 0)
    loss = lambda netgains: netgains.where(netgains < 0, 0)
    smg14 = lambda gain: gain.rolling(window=14).mean()
    sml14 = lambda loss: loss.rolling(window=14).mean()
    rsr = lambda smg14, sml14: smg14 / sml14
    rsi = lambda rsr: 100 - (100 / (1 + rsr))


class VolatilityCalculator(TechnicalCalculator): pass
class BBCalculator(VolatilityCalculator, variables=["bbh", "bbl"], register=Concepts.Technicals.Volatility.BB):
    sma20 = lambda adjusted: adjusted.rolling(window=20).mean()
    smd20 = lambda adjusted: adjusted.rolling(window=20).std()
    bbh = lambda sma20, smd20: sma20 + 2 * smd20
    bbl = lambda sma20, smd20: sma20 - 2 * smd20

class ATRCalculator(VolatilityCalculator, variables=["atr"], register=Concepts.Technicals.Volatility.ATR):
    xhl = lambda high, low: high - low
    xhc = lambda close, high: (high - close.shift(1)).abs()
    xlc = lambda close, low: (low - close.shift(1)).abs()
    xtr = lambda xhl, xhc, xlc: pd.concat([xhl, xhc, xlc], axis=1).max(axis=1)
    atr = lambda xtr: xtr.ewm(alpha=1/14, adjust=False).mean()


class VolumeCalculator(TechnicalCalculator): pass
class MFICalculator(VolumeCalculator, variables=["mfi"], register=Concepts.Technicals.Volume.MFI):
    typ = lambda close, low, high: (close + low + high) / 3
    rmf = lambda typ, volume: typ * volume
    pmf = lambda typ, rmf: rmf.where(typ.diff() > 0, 0)
    nmf = lambda typ, rmf: rmf.where(typ.diff() < 0, 0)
    mfr = lambda pmf, nmf: pmf.rolling(14).sum() / nmf.rolling(14).sum()
    mfi = lambda mfr: 100 - (100 / (1 + mfr))

class CMFCalculator(VolumeCalculator, variables=["cmf"], register=Concepts.Technicals.Volume.CMF):
    mfm = lambda close, low, high: (((close - low) - (high - close)) / (high - low)).replace([np.inf, -np.inf], 0).fillna(0)
    mfv = lambda mfm, volume: mfm * volume
    cmf = lambda mfv, volume: mfv.rolling(window=21).sum() / volume.rolling(window=21).sum()

class OBVCalculator(VolumeCalculator, variables=["obv"], register=Concepts.Technicals.Volume.OBV):
    rvf = lambda netgains, volume: (np.sign(netgains) * volume).fillna(0).cumsum()
    mvf = lambda rvf: rvf.abs().max()
    obv = lambda rvf, mvf: 100 * rvf / mvf


