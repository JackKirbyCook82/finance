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
from types import SimpleNamespace
from functools import cached_property
from datetime import datetime as Datetime
from dataclasses import dataclass, asdict, fields

from finance.enumerations import Enumerations
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Variables", "Metrics", "Collection", "OSI"]
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


class Variable(ABC):
    def __init_subclass__(cls, **kwargs):
        cls.name = kwargs.get("name", getattr(cls, "name", None))

class StrVariable(str, Variable):
    def __str__(self): return self.upper()

class NumVariable(float, Variable):
    def __str__(self): return f"{self:.2f}"

class DateVariable(Date, Variable):
    def __str__(self): return f"{self.strftime('%Y/%m/%d')}"

    def __new__(cls, *args, **kwargs):
        try: attributes = SimpleNamespace(year=args[0].year, month=args[0].month, day=args[0].day)
        except (IndexError, AttributeError):
            try: attributes = SimpleNamespace(year=kwargs["year"], month=kwargs["month"], day=kwargs["day"])
            except KeyError: raise ExpireError()
        return super().__new__(cls, year=attributes.year, month=attributes.month, day=attributes.day)


class TickerVariable(StrVariable, name="ticker"): pass
class ExpireVariable(DateVariable, name="expire"): pass
class StrikeVariable(NumVariable, name="strike"): pass
class MoneynessVariable(NumVariable, name="moneyness"): pass
class TightnessVariable(NumVariable, name="tightness"): pass
class ActivityVariable(NumVariable, name="activity"): pass
class ZScoreVariable(NumVariable, name="zscore"): pass
class MultipleVariable(NumVariable, name="multiple"): pass
class RatioVariable(NumVariable, name="ratio"): pass


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


class CollectionMeta(RegistryMeta):
    def __init__(cls, *args, **kwargs):
        variable = kwargs.get("variable", getattr(cls, "variable", None))
        super().__init__(*args, register=variable, **kwargs)
        cls.variable = variable

    def __call__(cls, contents):
        name = str(cls.variable.name).lower()
        if isinstance(contents, pd.DataFrame): contents = set(contents[name].unique())
        elif isinstance(contents, pd.Series): contents = set(contents.unique())
        elif isinstance(contents, list): contents = set([getattr(content, name) for content in contents])
        elif isinstance(contents, set):
            if not all([isinstance(content, cls.variable) for content in contents]): raise VariableError[name]()
            else: contents = set(contents)
        else: raise VariableError[name]()
        contents = {cls.variable(content) for content in contents}
        return super().__call__(contents)


class Collection(frozenset, metaclass=CollectionMeta): pass
class ListCollection(Collection):
    def __str__(self):
        if not bool(self): return ""
        strings = list(map(str, self))
        return f"{','.join(strings)}"

class RangeCollection(Collection):
    def __str__(self):
        if not bool(self): return ""
        if min(self) == max(self): return str(min(self))
        return f"{str(min(self))}->{str(max(self))}"


class TickerCollection(ListCollection, variable=TickerVariable): pass
class ExpireCollection(RangeCollection, variable=ExpireVariable): pass
class StrikeCollection(RangeCollection, variable=StrikeVariable): pass


class MetricMeta(RegistryMeta):
    def __init__(cls, *args, **kwargs):
        variable = kwargs.get("variable", getattr(cls, "variable", None))
        absolute = kwargs.get("absolute", getattr(cls, "absolute", False))
        upward = kwargs.get("upward", getattr(cls, "upward", None))
        title = kwargs.get("title", getattr(cls, "title", None))
        super().__init__(*args, register=variable, **kwargs)
        cls.variable = variable
        cls.absolute = absolute
        cls.upward = upward
        cls.title = title


class Metric(float, ABC, metaclass=MetricMeta):
    def __str__(self): return f"{float(self):.2f}"

    @property
    def title(self): return f"|{type(self).title}|" if type(self).absolute else type(self).title
    @property
    def direction(self): return ">=" if type(self).upward else "<="


class MoneynessMetric(Metric, title="Moneyness", variable=MoneynessVariable, upward=False, absolute=True): pass
class TightnessMetric(Metric, title="Tightness", variable=TightnessVariable, upward=False): pass
class ActivityMetric(Metric, title="Activity", variable=ActivityVariable, upward=True): pass
class ZScoreMetric(Metric, title="ZScore", variable=ZScoreVariable, upward=True): pass
class MultipleMetric(Metric, title="Multiple", variable=MultipleVariable, upward=True): pass
class RatioMetric(Metric, title="Ratio", variable=RatioVariable, upward=True): pass


@dataclass
class Measure:
    values: NDArray[np.float32]
    metric: Metric

    def __str__(self):
        metric, title, direction, percentage = str(self.metric), str(self.metric.title), str(self.metric.direction), self.percentage
        return f"{title} {direction} {metric} [{self.minimum:+.2f} -> {self.maximum:+.2f}, {percentage.total:.0f}%, {percentage.vaild:.0f.}%]"

    @cached_property
    def percentage(self):
        valid = self.population.passing / self.population.valid * 100 if self.population.valid else np.nan
        total = self.population.passing + self.population.total * 100 if self.population.total else np.nan
        return SimpleNamespace(valid=valid, total=total)

    @cached_property
    def population(self):
        if self.metric.upward: passing = self.values >= float(self.metric)
        else: passing = self.values <= float(self.metric)
        passing = np.count_nonzero(passing & np.isfinite(self.values))
        valid = np.isfinite(self.values).sum()
        total = len(self.values)
        return SimpleNamespace(passing=passing, valid=valid, total=total)

    @property
    def maximum(self): return self.values.max()
    @property
    def minimum(self): return self.values.min()


class MetricsError(Exception): pass
class Metrics(tuple):
    def __init_subclass__(cls, /, **kwargs):
        cls.variables = kwargs.get("variables", getattr(cls, "variables", None))

    def __new__(cls, /, **kwargs):
        mapping = dict.fromkeys([str(variable.name).lower() for variable in cls.variables])
        variables = {str(variable.name).lower(): variable for variable in cls.variables}
        for key in list(mapping.keys()):
            try: mapping[key] = kwargs[key]
            except KeyError: raise MetricsError()
        metrics = [Metric[variables[key]](value) for key, value in mapping.items()]
        instance = super().__new__(cls, metrics)
        for metric in metrics:
            attribute = str(metric.name).lower()
            setattr(instance, attribute, metric)
        return instance

    def __call__(self, contents):
        for metric in self:
            name = str(metric.variable.name).lower()
            if isinstance(contents, pd.DataFrame): values = contents[name].to_numpy()
            elif isinstance(contents, pd.Series): values = contents.to_numpy()
            elif isinstance(contents, list): values = np.array([getattr(content, name) for content in contents])
            else: raise VariableError[name]()
            if not metric.absolute: pass
            else: values = np.abs(values)
            yield Measure(values, metric)


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




