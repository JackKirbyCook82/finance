# -*- coding: utf-8 -*-
"""
Created on Weds May 27 2026
@name:   Finance Variable Objects
@author: Jack Kirby Cook

"""

from enum import Enum
from decimal import Decimal
from abc import ABC, abstractmethod
from typing import ClassVar, Callable, Any
from types import SimpleNamespace
from datetime import date as Date
from datetime import datetime as Datetime
from dataclasses import dataclass

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Querys", "Enumerations", "Specifications"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"

from numba.core.typing.enumdecl import EnumAttribute

decimal_formatter = lambda value: f"{Decimal(str(value)):.2f}"
decimal_parser = lambda value: Decimal(str(value))
enum_parser = lambda concept: lambda value: concept(value)
date_formatter = lambda value: value.strftime("%Y%m%d")


def date_parser(value):
    assert isinstance(value, (str, Date, Datetime))
    if isinstance(value, Datetime): return value.date()
    elif isinstance(value, Date): return value
    else: return Datetime.strptime(str(value), "%Y%m%d").date()


class Enumeration(Enum):
    def __str__(self): return str(self.name).lower()
    def __int__(self): return int(self.value)

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            string = value.upper().replace(" ", "").replace("_", "")
            if str(string).lstrip("-").isdigit():
                return cls(int(string))
            for member in cls:
                if member.name.replace("_", "") == string:
                    return member
        return None


class Technical(Enumeration): BARS, STATS, SMA, EMA, MACD, RSI, BB, ATR, MFI, CMF, OBV = range(11)
class Spread(Enumeration): EMPTY, VERTICAL, COLLAR, FLY, CALENDAR, CONDOR = range(6)
class Instrument(Enumeration): EMPTY, STOCK, OPTION, SPREAD, CONTRACT = range(5)
class Website(Enumeration): ETRADE, ALPACA, INTERACTIVE = range(3)
class Option(Enumeration): PUT, EMPTY, CALL = range(-1, 2)
class Position(Enumeration): SHORT, EMPTY, LONG = range(-1, 2)
class Terms(Enumeration): MARKET, LIMIT, STOP = range(3)
class Tenure(Enumeration): DAY, GTC, FOK = range(3)
class Intent(Enumeration): OPEN, CLOSE = range(2)
class Action(Enumeration): BUY, SELL = range(2)


@dataclass(frozen=True, slots=True)
class Specification(ABC):
    delimiter: ClassVar[str] = "|"

    def __str__(self): return self.delimiter.join(str(value) for value in self.key)
    def __iter__(self): return iter(self.key)

    @property
    @abstractmethod
    def key(self): pass


@dataclass(frozen=True, slots=True)
class Security(Specification):
    instrument: Instrument; option: Option.EMPTY; position: Position.EMPTY

    @property
    def key(self): return self.instrument, self.option, self.position


@dataclass(frozen=True, slots=True)
class Strategy(Specification):
    spread: Spread; option: Option.EMPTY; position: Position.EMPTY

    @property
    def key(self): return self.spread, self.option, self.position


StockLongSecurity = Security(Instrument.STOCK, Option.EMPTY, Position.LONG)
StockShortSecurity = Security(Instrument.STOCK, Option.EMPTY, Position.SHORT)
OptionPutLongSecurity = Security(Instrument.OPTION, Option.PUT, Position.LONG)
OptionPutShortSecurity = Security(Instrument.OPTION, Option.PUT, Position.SHORT)
OptionCallLongSecurity = Security(Instrument.OPTION, Option.CALL, Position.LONG)
OptionCallShortSecurity = Security(Instrument.OPTION, Option.CALL, Position.SHORT)

VerticalPutStrategy = Strategy(Spread.VERTICAL, Option.PUT,Position.EMPTY)
VerticalCallStrategy = Strategy(Spread.VERTICAL, Option.CALL, Position.EMPTY)
CondorLongStrategy = Strategy(Spread.CONDOR, Option.EMPTY, Position.LONG)
CondorShortStrategy = Strategy(Spread.CONDOR, Option.EMPTY, Position.SHORT)


class Registry(set):
    def __call__(self, value):
        if isinstance(value, Specification): return self.bykey()[tuple(value)]
        elif isinstance(value, str): return self.bystr()[value]
        elif isinstance(value, tuple): return self.bykey()[value]
        else: raise TypeError(type(value))

    def bystr(self): return {str(specification): specification for specification in self}
    def bykey(self): return {tuple(specification.key): specification for specification in self}


Securities = Registry([StockLongSecurity, StockShortSecurity, OptionPutLongSecurity, OptionPutShortSecurity, OptionCallLongSecurity, OptionCallShortSecurity])
Strategies = Registry([VerticalPutStrategy, VerticalCallStrategy, CondorLongStrategy, CondorShortStrategy,])


class FieldsError(Exception): pass
class FieldsMissingError(FieldsError): pass
class FieldsExcessiveError(FieldsError): pass


@dataclass(frozen=True, slots=True)
class Field:
    name: str
    parser: Callable[[Any], Any] = lambda value: value
    formatter: Callable[[Any], str] = str
    optional: bool = False

    def format(self, value):
        missing = value is None or value == ""
        if missing and not self.optional: raise FieldsMissingError(self.name)
        if missing and self.optional: return ""
        return self.formatter(value)

    def parse(self, string):
        missing = string is None or string == ""
        if missing and not self.optional: raise FieldsMissingError(self.name)
        if missing and self.optional: return None
        return self.parser(string)


InstrumentField = Field("instrument", enum_parser(Instrument), str)
OptionField = Field("option", enum_parser(Option), str)
PositionField = Field("position", enum_parser(Position), str)

TickerField = Field("ticker", str, str)
DateField = Field("date", date_parser, date_formatter)
ExpireField = Field("expire", date_parser, date_formatter)
StrikeField = Field("strike", decimal_parser, decimal_formatter)
PriceField = Field("price", decimal_parser, decimal_formatter)
BidField = Field("bid", decimal_parser, decimal_formatter)
AskField = Field("ask", decimal_parser, decimal_formatter)


@dataclass(frozen=True, slots=True)
class Record:
    name: str
    fields: tuple[Field, ...]
    delimiter: str = "|"

    def __iter__(self): return iter([field.name for field in self.fields])
    def __str__(self): return str(self.name)

    def __call__(self, values):
        if isinstance(values, str):
            strings = str(values).split(self.delimiter)
            if len(strings) > len(self.fields): raise FieldsExcessiveError()
            contents = {field.name: field.parse(value)for field, value in zip(self.fields, strings)}
        elif isinstance(values, (list, tuple)):
            if len(values) > len(self.fields): raise FieldsExcessiveError()
            contents = {field.name: field.parse(value) for field, value in zip(self.fields, values)}
        elif isinstance(values, dict):
            contents = {field.name: field.parse(values.get(field.name)) for field in self.fields}
        else: raise TypeError(type(values))
        return Query(self, contents)


@dataclass(frozen=True, slots=True)
class Query:
    record: Record
    data: dict[str, Any]

    def __iter__(self): return iter([(field, self.data.get(field.name)) for field in self.query.fields])
    def __str__(self):
        strings = [field.format(content) for field, content in iter(self)]
        return self.query.delimiter.join(strings).rstrip(self.query.delimiter)

    def __getattr__(self, name):
        try: return self.data[name]
        except KeyError: raise AttributeError(name) from None

    def items(self): return self.data.items()
    def values(self): return self.data.values()
    def keys(self): return self.data.keys()


SymbolRecord = Record("Symbol", fields=(TickerField,))
TradeRecord = Record("Trade", fields=(TickerField, PriceField))
QuoteRecord = Record("Quote", fields=(TickerField, BidField, AskField))
HistoryRecord = Record("History", fields=(TickerField, DateField))
SettlementRecord = Record("Settlement", fields=(TickerField, ExpireField))
ContractRecord = Record("Contract", fields=(TickerField, ExpireField, OptionField, StrikeField))


Enumerations = SimpleNamespace(Technical=Technical, Spread=Spread, Instrument=Instrument, Option=Option, Position=Position, Terms=Terms, Tenure=Tenure, Action=Action, Intent=Intent, Website=Website)
Querys = SimpleNamespace(Symbol=SymbolRecord, Trade=TradeRecord, Quote=QuoteRecord, History=HistoryRecord, Settlement=SettlementRecord, Contract=ContractRecord)
Specifications = SimpleNamespace(Securities=Securities, Strategies=Strategies)



