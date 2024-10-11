# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 2024
@name:   Finance Variable Objects
@author: Jack Kirby Cook

"""

import logging
import datetime
import pandas as pd
from abc import ABC, ABCMeta
from enum import Enum, EnumMeta
from collections import namedtuple as ntuple

from support.dispatchers import typedispatcher
from support.mixins import Field, Fields

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DateRange", "Variables", "Querys"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class Ticker(Field, type=str): pass
class Date(Field, type=datetime.date, format="%Y%m%d"): pass
class Expire(Field, type=datetime.date, format="%Y%m%d"): pass
class Strike(Field, type=float, decimals=2): pass
class Symbol(Fields, fields={"ticker": Ticker}): pass
class History(Fields, fields={"ticker": Ticker, "date": Date}): pass
class Contract(Fields, fields={"ticker": Ticker, "expire": Expire}): pass
class Product(Fields, fields={"ticker": Ticker, "expire": Expire, "strike": Strike}): pass


class DateRange(ntuple("DateRange", "minimum maximum")):
    def __contains__(self, date): return self.minimum <= date <= self.maximum
    def __new__(cls, dates):
        assert isinstance(dates, list)
        assert all([isinstance(date, Date) for date in dates])
        dates = [date if isinstance(date, Date) else date.date() for date in dates]
        return super().__new__(cls, min(dates), max(dates)) if dates else None

    def __iter__(self): return (date for date in pd.date_range(start=self.minimum, end=self.maximum))
    def __str__(self): return f"{str(self.minimum)}|{str(self.maximum)}"
    def __bool__(self): return self.minimum < self.maximum
    def __len__(self): return (self.maximum - self.minimum).days


class Variable(object):
    def __str__(self): return str(self.name).lower()
    def __bool__(self): return bool(self.index)
    def __int__(self): return int(self.index)
    def __init__(self, name, index):
        self.__name = str(name).upper()
        self.__index = int(index)

    @property
    def index(self): return self.__index
    @property
    def name(self): return self.__name


class VariableMeta(EnumMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        assert not any([issubclass(base, Enum) for base in bases])
        bases = (mcs.variable(name), Enum)
        for member, variable in mcs.variables(*args, **kwargs):
            attrs[member] = variable
        return super(VariableMeta, mcs).__new__(mcs, name, bases, attrs)

    def __call__(cls, content, *args, **kwargs):
        if isinstance(content, (int, str)):
            content = int(content) if str(content).isdigit() else str(content)
            mapping = {type(content)(variable): variable for variable in iter(cls)}
            variable = mapping[content]
            return variable
        if isinstance(content, Enum):
            return content
        return super(VariableMeta, cls).__call__(content, *args, **kwargs)

    @staticmethod
    def variable(name):
        def __str__(self): return str(self.value).lower()
        def __bool__(self): return bool(self.value)
        def __int__(self): return int(self.value)
        attrs = dict(__str__=__str__, __bool__=__bool__, __int__=__int__)
        return type(name, (object,), attrs)

    @staticmethod
    def variables(*args, members, start=1, **kwargs):
        assert isinstance(members, list) and bool(members)
        members = list(map(str.upper, list(members)))
        for index, member in enumerate(members, start=start):
            variable = Variable(member, index)
            yield member, variable


class MultiVariable(ABC):
    def __init_subclass__(cls, fields=[], attributes=[]):
        cls.fields, cls.attributes = list(fields), list(attributes)

    def __init__(self, *args, **kwargs):
        attributes, fields, contents = type(self).attributes, type(self).fields, list(args)
        assert isinstance(contents, list) and len(contents) == len(fields)
        self.attributes, self.fields, self.contents = attributes, fields, contents
        for field, content in zip(fields, contents):
            setattr(self, field, content)
        for attribute in attributes:
            setattr(self, attribute, kwargs[attribute])

    def __int__(self): return int(sum([pow(10, index) * int(content) for index, content in enumerate(reversed(self))]))
    def __str__(self): return str("|".join([str(content) for content in iter(self) if bool(content)]))
    def __hash__(self): return hash(tuple(zip(self.fields, self.contents)))
    def __reversed__(self): return reversed(self.contents)
    def __iter__(self): return iter(self.contents)

    def items(self): return tuple(zip(self.fields, self.contents))
    def values(self): return tuple(self.fields)
    def keys(self): return tuple(self.contents)


class MultiVariableMeta(ABCMeta):
    def __new__(mcs, name, base, attrs, *args, **kwargs):
        return super(MultiVariableMeta, mcs).__new__(mcs, name, base, attrs)

    def __init__(cls, *args, contents=[], **kwargs): cls.contents = list(contents)
    def __iter__(cls): return iter(cls.contents)

    def __call__(cls, content): return cls.create(int(content) if str(content).isdigit() else str(content))
    def __getitem__(cls, manifest): return {tuple(content): content for content in iter(cls)}[manifest]

    @typedispatcher
    def create(cls, content): raise TypeError(type(content).__name__)
    @create.register(MultiVariable)
    def value(cls, value): return {hash(content): content for content in iter(cls)}[value]
    @create.register(int)
    def integer(cls, number): return {int(content): content for content in iter(cls)}[number]
    @create.register(str)
    def string(cls, string): return {str(content): content for content in iter(cls)}[string]


class Theta(members=["PUT", "NEUTRAL", "CALL"], start=-1, metaclass=VariableMeta): pass
class Phi(members=["SHORT", "NEUTRAL", "LONG"], start=-1, metaclass=VariableMeta): pass
class Omega(members=["BEAR", "NEUTRAL", "BULL"], start=-1, metaclass=VariableMeta): pass
class Technicals(members=["BARS", "STATISTIC", "STOCHASTIC"], metaclass=VariableMeta): pass
class Scenarios(members=["MINIMUM", "MAXIMUM"], metaclass=VariableMeta): pass
class Valuations(members=["ARBITRAGE"], metaclass=VariableMeta): pass
class Pricing(members=["BLACKSCHOLES"], metaclass=VariableMeta): pass
class Status(members=["PROSPECT", "PENDING", "ABANDONED", "REJECTED", "ACCEPTED"], metaclass=VariableMeta): pass

# class Querys(members=["HISTORY", "SYMBOL", "CONTRACT", "PRODUCT", "SECURITY"], metaclass=VariableMeta): pass
# class Datasets(members=["PRICING", "SIZING", "TIMING", "EXPOSURE"], metaclass=VariableMeta): pass

class Markets(members=["EMPTY", "BEAR", "BULL"], start=0, metaclass=VariableMeta): pass
class Instruments(members=["EMPTY", "STOCK", "OPTION"], start=0, metaclass=VariableMeta): pass
class Options(members=["EMPTY", "PUT", "CALL"], start=0, metaclass=VariableMeta): pass
class Positions(members=["EMPTY", "LONG", "SHORT"], start=0, metaclass=VariableMeta): pass
class Spreads(members=["STRANGLE", "COLLAR", "VERTICAL"], start=1, metaclass=VariableMeta): pass

class Security(MultiVariable, fields=["instrument", "option", "position"]): pass
class Strategy(MultiVariable, fields=["spread", "option", "position"], attributes=["stocks", "options"]): pass


StockLong = Security(Instruments.STOCK, Options.EMPTY, Positions.LONG)
StockShort = Security(Instruments.STOCK, Options.EMPTY, Positions.SHORT)
OptionPutLong = Security(Instruments.OPTION, Options.PUT, Positions.LONG)
OptionPutShort = Security(Instruments.OPTION, Options.PUT, Positions.SHORT)
OptionCallLong = Security(Instruments.OPTION, Options.CALL, Positions.LONG)
OptionCallShort = Security(Instruments.OPTION, Options.CALL, Positions.SHORT)
VerticalPut = Strategy(Spreads.VERTICAL, Options.PUT, Positions.EMPTY, options=[OptionPutLong, OptionPutShort], stocks=[])
VerticalCall = Strategy(Spreads.VERTICAL, Options.CALL, Positions.EMPTY, options=[OptionCallLong, OptionCallShort], stocks=[])
CollarLong = Strategy(Spreads.COLLAR, Options.EMPTY, Positions.LONG, options=[OptionPutLong, OptionCallShort], stocks=[StockLong])
CollarShort = Strategy(Spreads.COLLAR, Options.EMPTY, Positions.SHORT, options=[OptionCallLong, OptionPutShort], stocks=[StockShort])


class Securities(contents=[StockLong, StockShort, OptionPutLong, OptionPutShort, OptionCallLong, OptionCallShort], metaclass=MultiVariableMeta):
    Options = [OptionPutLong, OptionCallLong, OptionPutShort, OptionCallShort]
    Puts = [OptionPutLong, OptionPutShort]
    Calls = [OptionCallLong, OptionCallShort]
    Stocks = [StockLong, StockShort]

    class Stock: Long = StockLong; Short = StockShort
    class Option:
        class Put: Long = OptionPutLong; Short = OptionPutShort
        class Call: Long = OptionCallLong; Short = OptionCallShort


class Strategies(contents=[VerticalPut, VerticalCall, CollarLong, CollarShort], metaclass=MultiVariableMeta):
    Verticals = [VerticalPut, VerticalCall]
    Collars = [CollarLong, CollarShort]

    class Vertical: Put = VerticalPut; Call = VerticalCall
    class Collar: Long = CollarLong; Short = CollarShort


#class Variables:
#    Securities = Securities
#    Strategies = Strategies
#    Markets = Markets
#    Pricing = Pricing
#    Instruments = Instruments
#    Options = Options
#    Positions = Positions
#    Spreads = Spreads
#    Valuations = Valuations
#    Scenarios = Scenarios
#    Technicals = Technicals
#    Status = Status
#    Querys = Querys
#    Datasets = Datasets
#    Omega = Omega
#    Theta = Theta
#    Phi = Phi

#class Querys:
#    Product = Product
#    Contract = Contract
#    Symbol = Symbol
#    History = History

