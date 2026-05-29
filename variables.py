# -*- coding: utf-8 -*-
"""
Created on Weds May 27 2026
@name:   Finance Concept Objects
@author: Jack Kirby Cook

"""

import regex as re
import pandas as pd
from decimal import Decimal
from enum import Enum, IntEnum
from abc import ABC, abstractmethod
from typing import ClassVar, Callable, Any
from types import SimpleNamespace
from datetime import date as Date
from datetime import datetime as Datetime
from dataclasses import dataclass, asdict, fields

from support.decorators import Dispatchers
from support.custom import DateRange
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Alerting", "OSI", "Querys", "Concepts", "Variables"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


decimal_formatter = lambda value: f"{Decimal(str(value)):.2f}"
decimal_parser = lambda value: Decimal(str(value))
enum_parser = lambda concept: lambda value: concept.create(value)
date_formatter = lambda value: value.strftime("%Y%m%d")


def date_parser(value):
    assert isinstance(value, (str, Date, Datetime))
    if isinstance(value, Datetime): return value.date()
    elif isinstance(value, Date): return value
    else: return Datetime.strptime(str(value), "%Y%m%d").date()


class ConceptEnum(IntEnum):
    def __str__(self): return str(self.name.lower())

    @classmethod
    def create(cls, value):
        if isinstance(value, cls): return value
        if isinstance(value, int): return cls(value)
        if isinstance(value, str):
            string = value.upper().replace(" ", "").replace("_", "")
            if str(string).lstrip("-").isdigit(): return cls(int(string))
            for member in cls:
                if member.name.replace("_", "") == string: return member
        raise ValueError(value)


class Technical(ConceptEnum): BARS, STATS, SMA, EMA, MACD, RSI, BB, ATR, MFI, CMF, OBV = range(11)
class Spread(ConceptEnum): EMPTY, VERTICAL, COLLAR, FLY, CALENDAR, CONDOR = range(6)
class Instrument(ConceptEnum): EMPTY, STOCK, OPTION, SPREAD = range(4)
class Option(ConceptEnum): PUT, EMPTY, CALL = range(-1, 2)
class Position(ConceptEnum): SHORT, EMPTY, LONG = range(-1, 2)
class Terms(ConceptEnum): MARKET, LIMIT, STOP = range(3)
class Action(ConceptEnum): BUY, SELL = range(2)


@dataclass(frozen=True, slots=True)
class ConceptDataclass(ABC):
    delimiter: ClassVar[str] = "|"

    def __str__(self): return self.delimiter.join(str(value) for value in self.key)
    def __iter__(self): return iter(self.key)

    @property
    @abstractmethod
    def key(self): pass


@dataclass(frozen=True, slots=True)
class Security(ConceptDataclass):
    instrument: Instrument; option: Option.EMPTY; position: Position.EMPTY

    @property
    def key(self): return self.instrument, self.option, self.position


@dataclass(frozen=True, slots=True)
class Strategy(ConceptDataclass):
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


class FieldsError(Exception): pass
class FieldsMissingError(FieldsError): pass
class FieldsExcessiveError(FieldsError): pass


@dataclass(frozen=True, slots=True)
class FieldDataclass:
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


InstrumentField = FieldDataclass("instrument", enum_parser(Instrument), str)
OptionField = FieldDataclass("option", enum_parser(Option), str)
PositionField = FieldDataclass("position", enum_parser(Position), str)

TickerField = FieldDataclass("ticker", str, str)
DateField = FieldDataclass("date", date_parser, date_formatter)
ExpireField = FieldDataclass("expire", date_parser, date_formatter)
StrikeField = FieldDataclass("strike", decimal_parser, decimal_formatter)
PriceField = FieldDataclass("price", decimal_parser, decimal_formatter)
BidField = FieldDataclass("bid", decimal_parser, decimal_formatter)
AskField = FieldDataclass("ask", decimal_parser, decimal_formatter)


@dataclass(frozen=True, slots=True)
class QueryFields:
    name: str
    fields: tuple[FieldDataclass, ...]
    delimiter: str = "|"

    def create(self, values):
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
        return QueryContents(self, contents)


@dataclass(frozen=True, slots=True)
class QueryContents:
    key: QueryFields
    value: dict[str, Any]

    def __iter__(self): return iter([(field, self.value.get(field.name)) for field in self.key.fields])
    def __str__(self):
        strings = [field.format(content) for field, content in iter(self)]
        return self.key.delimiter.join(strings).rstrip(self.key.delimiter)

    def __getattr__(self, name):
        try: return self.value[name]
        except KeyError: raise AttributeError(name) from None

    def items(self): return self.value.items()
    def values(self): return self.value.values()
    def keys(self): return self.value.keys()


SymbolQuery = QueryFields("Symbol", fields=(TickerField,))
TradeQuery = QueryFields("Trade", fields=(TickerField, PriceField))
QuoteQuery = QueryFields("Quote", fields=(TickerField, BidField, AskField))
HistoryQuery = QueryFields("History", fields=(TickerField, DateField))
SettlementQuery = QueryFields("Settlement", fields=(TickerField, ExpireField))
ContractQuery = QueryFields("Contract", fields=(TickerField, ExpireField, OptionField, StrikeField))


class Registry(set):
    def __getitem__(self, value):
        if isinstance(value, ConceptDataclass): return self.bykey()[tuple(value)]
        elif isinstance(value, str): return self.bystr()[str(value.lower())]
        elif isinstance(value, tuple): return self.bykey()[tuple(value)]
        else: raise TypeError(type(value))

    def bystr(self): return {str(member): member for member in self}
    def bykey(self): return {member.key: member for member in self}


Securities = Registry([StockLongSecurity, StockShortSecurity, OptionPutLongSecurity, OptionPutShortSecurity, OptionCallLongSecurity, OptionCallShortSecurity])
Strategies = Registry([VerticalPutStrategy, VerticalCallStrategy, CondorLongStrategy, CondorShortStrategy,])


class OSIError(Exception): pass
class OSIParseError(OSIError): pass
class OSICreateError(OSIError): pass

@dataclass(frozen=True)
class OSI:
    ticker: str; expire: Date; option: Enum; strike: Decimal

    @classmethod
    def create(cls, contents):
        if isinstance(contents, QueryContents): contents = dict(contents.items())
        if isinstance(contents, pd.Series): contents = contents.to_dict()
        if isinstance(contents, dict): return cls(**{field.name: contents[field.name] for field in fields(cls)})
        elif isinstance(contents, str): return cls(*cls.parse(contents))
        elif isinstance(contents, (list, tuple)): return cls(*contents)
        else: raise OSICreateError()

    def __str__(self):
        ticker = self.ticker.upper()
        expire = self.expire.strftime("%y%m%d")
        option = str(self.option).upper()[0]
        strike = int((self.strike * Decimal("1000")).to_integral_value())
        strike = f"{strike:08d}"
        return f"{ticker}{expire}{option}{strike}"

    def items(self): return asdict(self).items()
    def values(self): return asdict(self).values()
    def keys(self): return asdict(self).keys()

    @classmethod
    def parse(cls, contents):
        pattern = r"^(?P<ticker>[A-Z]+)(?P<expire>\d{6})(?P<option>[PC])(?P<strike>\d{8})$"
        match = re.search(pattern, contents)
        if not match: raise OSIParseError()
        values = match.groupdict()
        ticker = values["ticker"].upper()
        expire = Datetime.strptime(values["expire"], "%y%m%d").date()
        option = {"P": Option.PUT, "C": Option.CALL}[values["option"]]
        strike = Decimal(values["strike"]) / Decimal("1000")
        return [ticker, expire, option, strike]


class Alerting(Logging):
    @Dispatchers.Value(locator="instrument")
    def alert(self, dataframe, *args, title, instrument, **kwargs): raise ValueError(instrument)

    @alert.register(Instrument.SPREAD)
    def spread(self, collection, *args, title, instrument, **kwargs):
        if not isinstance(collection, list): collection = [collection]
        tickers = "|".join(list({content.ticker for content in collection}))
        previous, post = kwargs.get("previous", None), kwargs.get("post", len(collection))
        sizes = f"{int(previous):.0f}|{int(post):.0f}" if previous is not None else f"{len(collection):.0f}"
        self.console(str(title), f"{str(instrument).title()}[{str(tickers)}, {str(sizes)}]")

    @alert.register(Instrument.STOCK)
    def stock(self, dataframe, *args, title, instrument, **kwargs):
        tickers = "|".join(list(dataframe["ticker"].unique()))
        previous, post = kwargs.get("previous", None), kwargs.get("post", len(dataframe))
        sizes = f"{int(previous):.0f}|{int(post):.0f}" if previous is not None else f"{len(dataframe):.0f}"
        self.console(str(title), f"{str(instrument).title()}[{str(tickers)}, {str(sizes)}]")

    @alert.register(Instrument.OPTION)
    def option(self, dataframe, *args, title, instrument, **kwargs):
        tickers = "|".join(list(dataframe["ticker"].unique()))
        expires = DateRange.create(list(dataframe["expire"].unique()))
        expires = f"{expires.minimum.strftime('%Y%m%d')}->{expires.maximum.strftime('%Y%m%d')}"
        previous, post = kwargs.get("previous", None), kwargs.get("post", len(dataframe))
        sizes = f"{int(previous):.0f}|{int(post):.0f}" if previous is not None else f"{len(dataframe):.0f}"
        self.console(str(title), f"{str(instrument).title()}[{str(tickers)}, {str(expires)}, {str(sizes)}]")


Querys = SimpleNamespace(symbol=SymbolQuery, trade=TradeQuery, quote=QuoteQuery, history=HistoryQuery, settlement=SettlementQuery, contract=ContractQuery)
Concepts = SimpleNamespace(technical=Technical, spread=Spread, instrument=Instrument, option=Option, position=Position, terms=Terms, action=Action)
Variables = SimpleNamespace(securities=Securities, strategies=Strategies)



