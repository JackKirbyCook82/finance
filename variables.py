# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 2024
@name:   Finance Variable Objects
@author: Jack Kirby Cook

"""

import logging
import pandas as pd
from abc import ABC, ABCMeta
from enum import Enum, EnumMeta
from functools import total_ordering
from datetime import date as Date
from datetime import datetime as Datetime
from collections import namedtuple as ntuple

from support.dispatchers import typedispatcher

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DateRange", "Variables", "Symbol", "Contract", "Product"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class DateRange(ntuple("DateRange", "minimum maximum")):
    def __contains__(self, date): return self.minimum <= date <= self.maximum
    def __new__(cls, dates):
        assert isinstance(dates, list)
        assert all([isinstance(date, Date) for date in dates])
        dates = [date if isinstance(date, Date) else date.date() for date in dates]
        return super().__new__(cls, min(dates), max(dates)) if dates else None

    def __iter__(self): return (date for date in pd.date_range(start=self.minimum, end=self.maximum))
    def __repr__(self): return f"{self.__class__.__name__}({repr(self.minimum)}, {repr(self.maximum)})"
    def __str__(self): return f"{str(self.minimum)}|{str(self.maximum)}"
    def __bool__(self): return self.minimum < self.maximum
    def __len__(self): return (self.maximum - self.minimum).days


@total_ordering
class Symbol(ntuple("Symbol", "ticker")):
    def __str__(self): return str(self.ticker).upper()
    def __hash__(self): return hash(self.ticker)
    def __eq__(self, other): return str(self.ticker) == str(self.ticker)
    def __lt__(self, other): return str(self.ticker) < str(self.ticker)

    def tostring(self, *args, **kwargs):
        return str(self.ticker).upper()

    @classmethod
    def fromstring(cls, string, *args, **kwargs):
        return cls(str(string).upper())


@total_ordering
class History(ntuple("History", "ticker date")):
    def __str__(self): return f"{str(self.ticker).upper()}|{str(self.date.strftime('%Y-%m-%d'))}"
    def __hash__(self): return hash((self.ticker, int(self.date.timestamp())))
    def __eq__(self, other): return str(self.ticker) == str(other.ticker) and self.date == other.date
    def __lt__(self, other): return str(self.ticker) < str(other.ticker) and self.date < other.date

    def tostring(self, *args, delimiter="_", dateformat="%Y%m%d", **kwargs):
        ticker = str(self.ticker).upper()
        date = str(self.date.strftime(str(dateformat)))
        return str(delimiter).join([ticker, date])

    @classmethod
    def fromstring(cls, string, *args, delimiter="_", **kwargs):
        ticker, date = str(string).split(delimiter)
        ticker = str(ticker).upper()
        date = Datetime.strptime(date, "%Y%m%d")
        return cls(ticker, date)


@total_ordering
class Contract(ntuple("Contract", "ticker expire")):
    def __str__(self): return f"{str(self.ticker).upper()}|{str(self.expire.strftime('%Y-%m-%d'))}"
    def __hash__(self): return hash((self.ticker, int(self.expire.timestamp())))
    def __eq__(self, other): return str(self.ticker) == str(other.ticker) and self.expire == other.expire
    def __lt__(self, other): return str(self.ticker) < str(other.ticker) and self.expire < other.expire

    def tostring(self, *args, delimiter="_", dateformat="%Y%m%d", **kwargs):
        ticker = str(self.ticker).upper()
        expire = str(self.expire.strftime(str(dateformat)))
        return str(delimiter).join([ticker, expire])

    @classmethod
    def fromstring(cls, string, *args, delimiter="_", **kwargs):
        ticker, expire = str(string).split(delimiter)
        ticker = str(ticker).upper()
        expire = Datetime.strptime(expire, "%Y%m%d")
        return cls(ticker, expire)


@total_ordering
class Product(ntuple("Product", "ticker expire strike")):
    def __str__(self): return f"{str(self.ticker).upper()}|{str(self.expire.strftime('%Y-%m-%d'))}|{float(self.strike):.02f}"
    def __hash__(self): return hash((self.ticker, int(self.expire.timestamp()), round(self.strike, 2)))
    def __eq__(self, other): return str(self.ticker) == str(other.ticker) and self.expire == other.expire and round(self.strike, 2) == round(other.strike, 2)
    def __lt__(self, other): return str(self.ticker) < str(other.ticker) and self.expire < other.expire and round(self.strike, 2) < round(other.strike, 2)

    def tostring(self, *args, delimiter="_", dateformat="%Y%m%d", decimals=2, **kwargs):
        ticker = str(self.ticker).upper()
        expire = str(self.date.strftime(str(dateformat)))
        strike = str(round(float(self.strike, int(decimals))))
        return str(delimiter).join([ticker, expire, strike])

    @classmethod
    def fromstring(cls, string, *args, delimiter="_", decimals=2, **kwargs):
        ticker, expire, strike = str(string).split(delimiter)
        ticker = str(ticker).upper()
        expire = Datetime.strptime(expire, "%Y%m%d")
        strike = round(float(strike, int(decimals)))
        return cls(ticker, expire, strike)


class Variable(object):
    def __init__(self, name, index):
        self.__name = str(name).upper()
        self.__index = int(index)

    def __repr__(self): return f"{self.__class__.__name__}({str(self.name).lower()}, {int(self.index):.0f})"
    def __str__(self): return str(self.name).lower()
    def __bool__(self): return bool(self.index)
    def __int__(self): return self.index

    @property
    def index(self): return self.__index
    @property
    def name(self): return self.__name


class VariableMeta(EnumMeta):
    def __new__(mcs, name, bases, attrs, *args, **kwargs):
        assert not any([issubclass(base, Enum) for base in bases])
        bases = (mcs.variable(name), Enum)
        for key, member in mcs.members(*args, **kwargs):
            attrs[key] = member
        return super(VariableMeta, mcs).__new__(mcs, name, bases, attrs)

    def __call__(cls, variable, *args, **kwargs):
        if isinstance(variable, (int, str)):
            variable = int(variable) if str(variable).isdigit() else str(variable)
            mapping = {type(variable)(member): member for member in iter(cls)}
            return mapping[variable]
        if isinstance(variable, Enum):
            return variable
        return super(VariableMeta, cls).__call__(variable, *args, **kwargs)

    @staticmethod
    def variable(name):
        def __str__(self): return str(self.value).lower()
        def __int__(self): return int(self.value)
        def __bool__(self): return bool(self.value)
        attrs = dict(__int__=__int__, __str__=__str__, __bool__=__bool__)
        return type(name, (object,), attrs)

    @staticmethod
    def members(*args, members, start=1, **kwargs):
        assert isinstance(members, list) and bool(members)
        members = list(map(str.upper, list(members)))
        for index, member in enumerate(members, start=start):
            variable = Variable(member, index)
            yield member, variable


class MultiVariable(ABC):
    def __init_subclass__(cls, fields=[], parameters=[]):
        cls.fields, cls.parameters = list(fields), list(parameters)

    def __init__(self, *args, **kwargs):
        parameters, fields, variables = type(self).parameters, type(self).fields, list(args)
        assert isinstance(variables, list) and len(variables) == len(fields)
        self.parameters, self.fields, self.variables = parameters, fields, variables
        for field, variable in zip(fields, variables):
            setattr(self, field, variable)
        for parameter in parameters:
            setattr(self, parameter, kwargs[parameter])

    def __int__(self): return int(sum([pow(10, index) * int(value) for index, value in enumerate(reversed(self))]))
    def __str__(self): return str("|".join([str(value) for value in iter(self) if bool(value)]))
    def __hash__(self): return hash(tuple(zip(self.variables, self.fields)))
    def __reversed__(self): return reversed(self.variables)
    def __iter__(self): return iter(self.variables)

    def items(self): return tuple(zip(self.variables, self.fields))
    def values(self): return tuple(self.variables)
    def keys(self): return tuple(self.fields)


class MultiVariableMeta(ABCMeta):
    def __new__(mcs, name, base, attrs, *args, **kwargs):
        return super(MultiVariableMeta, mcs).__new__(mcs, name, base, attrs)

    def __init__(cls, *args, variables=[], **kwargs): cls.variables = list(variables)
    def __iter__(cls): return iter(cls.variables)

    def __getitem__(cls, variable): return cls.retrieve(variable)
    def __call__(cls, variable):
        variable = int(variable) if str(variable).isdigit() else str(variable)
        return cls.create(variable)

    @typedispatcher
    def create(cls, variable): raise TypeError(type(variable).__name__)
    def retrieve(cls, variable): return {tuple(variable): variable for variable in iter(cls)}[variable]
    @create.register(MultiVariable)
    def value(cls, value): return {hash(variable): variable for variable in iter(cls)}[value]
    @create.register(int)
    def integer(cls, number): return {int(variable): variable for variable in iter(cls)}[number]
    @create.register(str)
    def string(cls, string): return {str(variable): variable for variable in iter(cls)}[string]


class Theta(members=["PUT", "NEUTRAL", "CALL"], start=-1, metaclass=VariableMeta): pass
class Phi(members=["SHORT", "NEUTRAL", "LONG"], start=-1, metaclass=VariableMeta): pass
class Omega(members=["BEAR", "NEUTRAL", "BULL"], start=-1, metaclass=VariableMeta): pass
class Technicals(members=["BARS", "STATISTIC", "STOCHASTIC"], metaclass=VariableMeta): pass
class Scenarios(members=["MINIMUM", "MAXIMUM"], metaclass=VariableMeta): pass
class Valuations(members=["ARBITRAGE"], metaclass=VariableMeta): pass
class Pricing(members=["BLACKSCHOLES"], metaclass=VariableMeta): pass
class Querys(members=["HISTORY", "SYMBOL", "CONTRACT", "PRODUCT", "SECURITY"], metaclass=VariableMeta): pass
class Datasets(members=["PRICING", "SIZING", "TIMING", "EXPOSURE"], metaclass=VariableMeta): pass
class Status(members=["PROSPECT", "PENDING", "ABANDONED", "REJECTED", "ACCEPTED"], metaclass=VariableMeta): pass

class Markets(members=["EMPTY", "BEAR", "BULL"], start=0, metaclass=VariableMeta): pass
class Instruments(members=["EMPTY", "STOCK", "OPTION"], start=0, metaclass=VariableMeta): pass
class Options(members=["EMPTY", "PUT", "CALL"], start=0, metaclass=VariableMeta): pass
class Positions(members=["EMPTY", "LONG", "SHORT"], start=0, metaclass=VariableMeta): pass
class Spreads(members=["STRANGLE", "COLLAR", "VERTICAL"], start=1, metaclass=VariableMeta): pass

class Security(MultiVariable, fields=["instrument", "option", "position"]): pass
class Strategy(MultiVariable, fields=["spread", "option", "position"], parameters=["stocks", "options"]): pass


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


class Securities(variables=[StockLong, StockShort, OptionPutLong, OptionPutShort, OptionCallLong, OptionCallShort], metaclass=MultiVariableMeta):
    Options = [OptionPutLong, OptionCallLong, OptionPutShort, OptionCallShort]
    Puts = [OptionPutLong, OptionPutShort]
    Calls = [OptionCallLong, OptionCallShort]
    Stocks = [StockLong, StockShort]

    class Stock: Long = StockLong; Short = StockShort
    class Option:
        class Put: Long = OptionPutLong; Short = OptionPutShort
        class Call: Long = OptionCallLong; Short = OptionCallShort


class Strategies(variables=[VerticalPut, VerticalCall, CollarLong, CollarShort], metaclass=MultiVariableMeta):
    Verticals = [VerticalPut, VerticalCall]
    Collars = [CollarLong, CollarShort]

    class Vertical: Put = VerticalPut; Call = VerticalCall
    class Collar: Long = CollarLong; Short = CollarShort


class Variables:
    Securities = Securities
    Strategies = Strategies
    Markets = Markets
    Pricing = Pricing
    Instruments = Instruments
    Options = Options
    Positions = Positions
    Spreads = Spreads
    Valuations = Valuations
    Scenarios = Scenarios
    Technicals = Technicals
    Status = Status
    Querys = Querys
    Datasets = Datasets
    Omega = Omega
    Theta = Theta
    Phi = Phi



