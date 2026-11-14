# -*- coding: utf-8 -*-
"""
Created on Weds May 27 2026
@name:   Finance Variables Objects
@author: Jack Kirby Cook
@file:   finance/options.py

"""

import regex as re
import numpy as np
import pandas as pd
from abc import ABC
from decimal import Decimal
from numpy.typing import NDArray
from datetime import date as Date
from functools import cached_property
from datetime import datetime as Datetime
from dataclasses import dataclass, asdict, fields

from finance.enumerations import Enumerations
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Variables", "Collections", "Metrics", "OSI"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class VariableError(Exception, metaclass=RegistryMeta): pass
class TickerError(VariableError, register="ticker"): pass
class ExpireError(VariableError, register="expire"): pass
class StrikeError(VariableError, register="strike"): pass
class MoneynessError(VariableError, register="moneyness"): pass
class TightnessError(VariableError, register="tightness"): pass
class ActivityError(VariableError, register="activity"): pass
class ZScoreError(VariableError, register="zscore"): pass
class MultipleError(VariableError, register="multiple"): pass
class RatioError(VariableError, register="ratio"): pass


class Variable(ABC): pass
class StrVariable(str, Variable):
    def __str__(self): return self.upper()

class NumVariable(float, Variable):
    def __str__(self): return f"{self:.2f}"
    def __float__(self): return float(self)

class DateVariable(Date, Variable):
    def __str__(self): return f"{self.strftime('%Y/%m/%d')}"


class TickerVariable(StrVariable): pass
class ExpireVariable(DateVariable): pass
class StrikeVariable(NumVariable): pass
class MoneynessVariable(NumVariable): pass
class TightnessVariable(NumVariable): pass
class ActivityVariable(NumVariable): pass
class ZScoreVariable(NumVariable): pass
class MultipleVariable(NumVariable): pass
class RatioVariable(NumVariable): pass


class Variables:
    Ticker = TickerVariable
    Expire = ExpireVariable
    Strike = StrikeVariable
    Moneyness = MoneynessVariable
    Tightness = TightnessVariable
    Activity = ActivityVariable
    ZScore = ZScoreVariable
    Multiple = MultipleVariable
    Ratio = RatioVariable


class Collection(frozenset, ABC):
    def __init_subclass__(cls, /, **kwargs):
        cls.variable = kwargs.get("variable", getattr(cls, "variable", None))

    def __new__(cls, contents):
        name = str(cls.variable.__name__).lower()
        if isinstance(contents, pd.DataFrame): contents = set(contents[name].unique())
        elif isinstance(contents, pd.Series): contents = set(contents.unique())
        elif isinstance(contents, list): contents = set([getattr(content, name) for content in contents])
        elif isinstance(contents, set):
            if not all([isinstance(content, cls.variable) for content in contents]): raise VariableError[name]()
            else: contents = set(contents)
        else: raise VariableError[name]()
        contents = {cls.variable(content) for content in contents}
        return super().__new__(cls, contents)


class ListCollection(Collection):
    def __str__(self):
        strings = list(map(str, self))
        return f"{','.join(strings)}"

class RangeCollection(Collection):
    def __str__(self):
        if min(self) == max(self): return str(min(self))
        return f"{str(min(self))}->{str(max(self))}"


class TickerCollection(ListCollection, variable=TickerVariable): pass
class ExpireCollection(RangeCollection, variable=ExpireVariable): pass
class StrikeCollection(RangeCollection, variable=StrikeVariable): pass


class Collections:
    Ticker = TickerCollection
    Expire = ExpireCollection
    Strike = StrikeCollection


class Metric(float, ABC):
    def __init_subclass__(cls, /, **kwargs):
        cls.variable = kwargs.get("variable", getattr(cls, "variable", None))
        cls.absolute = kwargs.get("absolute", getattr(cls, "absolute", False))
        cls.upward = kwargs.get("upward", getattr(cls, "upward", None))
        cls.name = kwargs.get("name", getattr(cls, "name", None))

    def __str__(self): return f"{float(self)}:.2f"
    def __float__(self): return float(self)

    @property
    def title(self): return f"|{str(self.name)}|" if self.absolute else str(self.name)
    @property
    def direction(self): return ">=" if self.upward else "<="


class MoneynessMetric(Metric, name="Moneyness", variable=MoneynessVariable, upward=False, absolute=True): pass
class TightnessMetric(Metric, name="Tightness", variable=TightnessVariable, upward=False): pass
class ActivityMetric(Metric, name="Activity", variable=ActivityVariable, upward=True): pass
class ZScoreMetric(Metric, name="ZScore", variable=ZScoreVariable, upward=True): pass
class MultipleMetric(Metric, name="Multiple", variable=MultipleVariable, upward=True): pass
class RatioMetric(Metric, name="Ratio", variable=RatioVariable, upward=True): pass


class Metrics:
    Moneyness = MoneynessMetric
    Tightness = TightnessMetric
    Activity = ActivityMetric
    ZScore = ZScoreMetric
    Multiple = MultipleMetric
    Ratio = RatioMetric


@dataclass
class Measure:
    contents: NDArray[np.float32] | pd.DataFrame | pd.Series | list[object]
    metric: Metric

    def __post_init__(self):
        name = str(self.metric.variable.__name__).lower()
        if isinstance(self.contents, pd.DataFrame): contents = self.contents[name].to_numpy()
        elif isinstance(self.contents, pd.Series): contents = self.contents.to_numpy()
        elif isinstance(self.contents, list): contents = np.array([getattr(content, name) for content in self.contents])
        else: raise VariableError[name]()
        if not self.metric.absolute: pass
        else: contents = np.abs(contents)
        self.contents = contents

    def __str__(self):
        metric, title, direction = str(self.metric), str(self.metric.title), str(self.metric.direction)
        return f"{title} {direction} {metric} [{self.minimum:+.2f} -> {self.maximum:+.2f}, {self.percentage:.0f}%]"

    @cached_property
    def percentage(self):
        if bool(self.metric.upward): return (self.contents >= float(self.metric)).sum() / len(self.contents) * 100
        else: return (self.contents <= float(self.metric)).sum() / len(self.contents) * 100

    @property
    def maximum(self): return self.contents.max()
    @property
    def minimum(self): return self.contents.min()


# @dataclass
# class Metrics(ABC):
#     def __call__(self, contents):
#         metrics = [field for field in fields(self) if isinstance(field, Metric)]
#         measures = [Measure(contents, metric) for metric in metrics]
#         return Measures(measures)


class OSIError(Exception): pass
class OSIParseError(OSIError): pass
class OSICreateError(OSIError): pass
class OSIEmptyError(OSIError): pass


class OSIMeta(type):
    def __call__(cls, contents):
        if isinstance(contents, cls): return contents
        if isinstance(contents, pd.Series): contents = contents.to_dict()
        elif hasattr(contents, "items"): contents = dict(contents.items())
        if isinstance(contents, dict): instance = super().__call__(**{field.name: contents[field.name] for field in fields(cls)})
        elif isinstance(contents, str): instance = super().__call__(*cls.parse(contents))
        elif isinstance(contents, (list, tuple)): instance = super().__call__(*contents)
        else: raise OSICreateError()
        return instance

    @staticmethod
    def parse(contents):
        pattern = r"^(?P<ticker>[A-Z]+)(?P<expire>\d{6})(?P<right>[PC])(?P<strike>\d{8})$"
        match = re.search(pattern, contents)
        if not match: raise OSIParseError()
        values = match.groupdict()
        ticker = values["ticker"].upper()
        expire = Datetime.strptime(values["expire"], "%y%m%d").date()
        right = {str(right).upper()[0]: right for right in Enumerations.Right}[values["right"]]
        strike = float(Decimal(values["strike"]) / Decimal("1000"))
        return [ticker, expire, right, strike]


@dataclass(frozen=True)
class OSI(metaclass=OSIMeta):
    ticker: str; expire: Date; right: Enumerations.Right; strike: float

    def __str__(self):
        ticker = self.ticker.upper()
        expire = self.expire.strftime("%y%m%d")
        right = {right: str(right).upper()[0] for right in Enumerations.Right}[self.right]
        strike = int((Decimal(self.strike) * Decimal("1000")).to_integral_value())
        strike = f"{strike:08d}"
        return f"{ticker}{expire}{right}{strike}"

    def items(self): return asdict(self).items()
    def values(self): return asdict(self).values()
    def keys(self): return asdict(self).keys()




