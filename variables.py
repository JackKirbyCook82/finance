# -*- coding: utf-8 -*-
"""
Created on Weds May 27 2026
@name:   Finance Concept Objects
@author: Jack Kirby Cook

"""

import regex as re
import pandas as pd
from abc import ABC
from decimal import Decimal
from enum import Enum, IntEnum
from typing import Callable, Any
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


string_formatter = lambda value: str(value)
string_parser = lambda value: str(value)
decimal_formatter = lambda value: f"{Decimal(str(value)):.02}"
decimal_parser = lambda value: Decimal(str(value))
enum_formatter = lambda value: str(value)
enum_parser = lambda concept: lambda value: value if isinstance(value, concept) else concept(value)
date_formatter = lambda value: value.strftime("%Y%m%d")


def date_parser(value):
    assert isinstance(value, (str, Date, Datetime))
    if isinstance(value, Date): return value
    elif isinstance(value, Datetime): return value.date()
    else: return Datetime.strptime(str(value), "%Y%m%d").date()


class ConceptEnum(IntEnum):
    def __str__(self): return str(self.name.lower())

    @classmethod
    def create(cls, value):
        if isinstance(value, cls): return value
        else: return cls.parse(value)

    @classmethod
    def parse(cls, string):
        string = string.upper().replace(" ", "").replace("_", "")
        if string.lstrip("-").isdigit(): return cls(int(string))
        else: return cls(string)


class Technical(ConceptEnum): BARS, STATS, SMA, EMA, MACD, RSI, BB, ATR, MFI, CMF, OBV = range(11)
class Spread(ConceptEnum): EMPTY, VERTICAL, COLLAR, FLY, CALENDAR, CONDOR = range(6)
class Instrument(ConceptEnum): EMPTY, STOCK, OPTION, SPREAD = range(4)
class Option(ConceptEnum): PUT, EMPTY, CALL = range(-1, 1, 1)
class Position(ConceptEnum): SHORT, EMPTY, LONG = range(-1, 1, 1)
class Terms(ConceptEnum): MARKET, LIMIT, STOP = range(3)
class Action(ConceptEnum): BUY, SELL = range(2)


@dataclass(frozen=True, slots=True)
class ConceptDataclass(ABC):
    delimiter: str = "|"

    def __iter__(self): return iter(fields(self))
    def __str__(self):
        strings = list(map(str, self))
        return str(self.delimiter).join(strings)


class Security(ConceptDataclass): instrument: Instrument; option: Option.EMPTY; position: Position.EMPTY
class Strategy(ConceptDataclass): pass
class Vertical(Strategy): option: Option
class Condor(Strategy): position: Position


@dataclass(frozen=True, slots=True)
class FieldDataclass:
    name: str
    parser: Callable[[Any], Any] = lambda value: value
    formatter: Callable[[Any], str] = str
    optional: bool = False

    def format(self, value): return self.formatter(value)
    def parse(self, string): return self.parser(string)


@dataclass(frozen=True, slots=True)
class QueryFields:
    name: str
    fields: tuple[FieldDataclass, ...]
    delimiter: str = "|"


@dataclass(frozen=True, slots=True)
class QueryContents:
    key: QueryFields
    value: dict[str, Any]

    def __iter__(self): return iter([(field, self.value.get(field)) for field in self.key.fields])
    def __str__(self):
        strings = [field.format(content) for field, content in iter(self)]
        self.key.delimiter.join(strings).rstrip(self.key.delimiter)


class Variables(set):
    def __getitem__(self, value):
        if isinstance(value, ConceptDataclass): return {tuple(member): member for member in self}[tuple(value)]
        elif isinstance(value, str): return {str(member.name.lower()): member for member in self}[str(value.lower())]
        elif isinstance(value, tuple): return {tuple(member): member for member in self}[tuple(value)]
        else: raise TypeError(type(value))


StockLongSecurity = Security(Instrument.STOCK, Option.EMPTY, Position.LONG)
StockShortSecurity = Security(Instrument.STOCK, Option.EMPTY, Position.SHORT)
OptionPutLongSecurity = Security(Instrument.OPTION, Option.PUT, Position.LONG)
OptionPutShortSecurity = Security(Instrument.OPTION, Option.PUT, Position.SHORT)
OptionCallLongSecurity = Security(Instrument.OPTION, Option.CALL, Position.LONG)
OptionCallShortSecurity = Security(Instrument.OPTION, Option.CALL, Position.SHORT)

InstrumentField = FieldDataclass("instrument", enum_parser(Instrument), enum_formatter)
OptionField = FieldDataclass("option", enum_parser(Option), enum_formatter)
PositionField = FieldDataclass("position", enum_parser(Position), enum_formatter)

TickerField = FieldDataclass("ticker", string_parser, string_formatter)
DateField = FieldDataclass("date", date_parser, date_formatter)
ExpireField = FieldDataclass("expire", date_parser, date_formatter)
StrikeField = FieldDataclass("strike", decimal_parser, decimal_formatter)
PriceField = FieldDataclass("price", decimal_parser, decimal_formatter)
BidField = FieldDataclass("bid", decimal_parser, decimal_formatter)
AskField = FieldDataclass("ask", decimal_parser, decimal_formatter)

SymbolQuery = QueryFields("Symbol", fields=(TickerField,))
TradeQuery = QueryFields("Trade", fields=(TickerField, PriceField))
QuoteQuery = QueryFields("Quote", fields=(TickerField, BidField, AskField))
HistoryQuery = QueryFields("History", fields=(TickerField, DateField))
SettlementQuery = QueryFields("Settlement", fields=(TickerField, ExpireField))
ContractQuery = QueryFields("Contract", fields=(TickerField, ExpireField, OptionField, StrikeField))

Securities = Variables([StockLongSecurity, StockShortSecurity, OptionPutLongSecurity, OptionPutShortSecurity, OptionCallLongSecurity, OptionCallShortSecurity])
Strategies = Variables([Vertical, Condor])

Querys = SimpleNamespace(symbol=SymbolQuery, trade=TradeQuery, quote=QuoteQuery, history=HistoryQuery, settlement=SettlementQuery, contract=ContractQuery)
Concepts = SimpleNamespace(technical=Technical, spread=Spread, instrument=Instrument, option=Option, position=Position, terms=Terms, action=Action)
Variables = SimpleNamespace(securities=Securities, strategies=Strategies)


class OSIError(Exception): pass
class OSIParseError(OSIError): pass
class OSICreateError(OSIError): pass

@dataclass(frozen=True)
class OSI:
    ticker: str; expire: Date; option: Enum; strike: float

    @classmethod
    def create(cls, contents):
        if isinstance(contents, Querys.Contract): contents = dict(contents.items())
        if isinstance(contents, pd.Series): contents = contents.to_dict()
        if isinstance(contents, dict): return cls(**{field.name: contents[field.name] for field in fields(cls)})
        elif isinstance(contents, str): return cls(*cls.parse(contents))
        elif isinstance(contents, (list, tuple)): return cls(*contents)
        else: raise OSICreateError()

    def __str__(self):
        ticker = self.ticker.upper()
        expire = self.expire.strftime("%y%m%d")
        option = str(self.option).upper()[0]
        strike_int = int(round(self.strike * 1000))
        strike = f"{strike_int:08d}"
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
        option = {str(option).upper()[0]: option for option in Concepts.Option}[values["option"]]
        strike = int(values["strike"]) / 1000.0
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

