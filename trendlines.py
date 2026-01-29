# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 2024
@name:   Technical Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from enum import Enum
from types import SimpleNamespace

from finance.concepts import Querys
from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TrendlineCalculator"]
__copyright__ = "Copyright 2024, Jack Kirby Cook"
__license__ = "MIT License"


class TrendlineCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, indicator, window, threshold, period, **kwargs):
        assert isinstance(window, int) and window > 0 and window % 2 != 0
        super().__init__(*args, **kwargs)
        self.__indicator = str(indicator).upper()
        self.__threshold = threshold
        self.__window = window
        self.__period = period

    def execute(self, technicals, /, **kwargs):
        assert isinstance(technicals, pd.DataFrame) and self.indicator in technicals.columns
        if self.empty(technicals): return
        symbols = self.keys(technicals, by=Querys.Symbol)
        symbols = ",".join(list(map(str, symbols)))
        technicals = self.calculate(technicals, **kwargs)
        size = self.size(technicals)
        self.console(f"{str(symbols)}[{int(size):.0f}]")
        if self.empty(technicals): return
        yield technicals

    def calculate(self, technicals, /, **kwargs):
        assert isinstance(technicals, pd.DataFrame)
        pivots = self.pivots(technicals[self.indicator], indicator=self.indicator, window=self.window)
        pivots = self.filters(pivots, indicator=self.indicator, threshold=self.threshold)
        trendlines = self.trendlines(pivots, indicator=self.indicator, period=self.period)
        technicals = pd.concat([technicals, trendlines.support, trendlines.resistance], axis=1)
        return technicals

    @staticmethod
    def pivots(series, /, indicator, window):
        assert isinstance(series, pd.Series)
        high = series.where(series.eq(series.rolling(window, center=True).max())).shift(window // 2)
        low = series.where(series.eq(series.rolling(window, center=True).min())).shift(window // 2)
        high = high.rename(f"{indicator}H")
        low = low.rename(f"{indicator}L")
        pivoted = SimpleNamespace(high=high, low=low)
        return pivoted

    @staticmethod
    def filters(series, /, indicator, threshold):
        assert isinstance(series, SimpleNamespace) and all([isinstance(value, pd.Series) for value in vars(series).values()])
        assert len(series.high) == len(series.low)
        size = min(len(series.high), len(series.low))
        Pivots = Enum("Pivot", [("HIGH", 1), ("LOW", 2)])
        array = np.full(size, np.NaN, dtype=float)
        filtered = SimpleNamespace(high=array.copy(), low=array)
        unfiltered = SimpleNamespace(high=series.high, low=series.low)
        last = SimpleNamespace(high=np.NaN, low=np.NaN)
        pivot = np.NaN
        for index in range(size):
            criteria = threshold.iloc[index] if isinstance(threshold, pd.Series) else float(threshold)
            if not np.isnan(unfiltered.high.iloc[index]) and (np.isnan(last.low) or abs(unfiltered.high.iloc[index] - last.low) >= criteria):
                if pivot != Pivots.HIGH or np.isnan(last.high) or unfiltered.high.iloc[index] > last.high:
                    filtered.high[index] = last.high = unfiltered.high.iloc[index]
                    pivot = Pivots.HIGH
            if not np.isnan(unfiltered.low.iloc[index]) and (np.isnan(last.high) or abs(last.high - unfiltered.low.iloc[index]) >= criteria):
                if pivot != Pivots.LOW or np.isnan(last.low) or unfiltered.low.iloc[index] < last.low:
                    filtered.low[index] = last.low = unfiltered.low.iloc[index]
                    pivot = Pivots.LOW
        high = pd.Series(filtered.high, index=series.high.index, name=f"{indicator}H")
        low = pd.Series(filtered.low, index=series.low.index, name=f"{indicator}L")
        filtered = SimpleNamespace(high=high, low=low)
        return filtered

    @staticmethod
    def trendlines(series, /, indicator, period):
        assert isinstance(series, SimpleNamespace) and all([isinstance(value, pd.Series) for value in vars(series).values()])
        assert len(series.high) == len(series.low)
        size = min(len(series.high), len(series.low))
        indexes = np.arange(size)
        array = np.full(size, np.NaN, dtype=float)
        trendlines = SimpleNamespace(support=array.copy(), resistance=array)
        points = SimpleNamespace(support=[], resistance=[])
        for index in range(size):
            value = SimpleNamespace(support=series.low.iloc[index], resistance=series.high.iloc[index])
            if not np.isnan(value.support):
                point = SimpleNamespace(x=index, y=float(value.support))
                points.support.append(point)
                if len(points.support) > period: points.support.pop(0)
            if not np.isnan(value.resistance):
                point = SimpleNamespace(x=index, y=float(value.resistance))
                points.resistance.append(point)
                if len(points.resistance) > period: points.resistance.pop(0)
            if len(points.support) >= 2:
                x = np.array([point.x for point in points.support], dtype=float)
                y = np.array([point.y for point in points.support], dtype=float)
                a, b = np.polyfit(x, y, 1)
                trendlines.support[index] = a * indexes[index] + b
            if len(points.resistance) >= 2:
                x = np.array([point.x for point in points.resistance], dtype=float)
                y = np.array([point.y for point in points.resistance], dtype=float)
                a, b = np.polyfit(x, y, 1)
                trendlines.resistance[index] = a * indexes[index] + b
        support = pd.Series(trendlines.support).rename(f"{indicator}SPT")
        resistance = pd.Series(trendlines.resistance).rename(f"{indicator}RST")
        trendlines = SimpleNamespace(support=support, resistance=resistance)
        return trendlines

    @property
    def indicator(self): return self.__indicator
    @property
    def threshold(self): return self.__threshold
    @property
    def period(self): return self.__period
    @property
    def window(self): return self.__window


