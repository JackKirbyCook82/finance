# -*- coding: utf-8 -*-
"""
Created on Weds May 27 2026
@name:   Finance Querys Objects
@author: Jack Kirby Cook

"""

from decimal import Decimal
from typing import Callable, Any
from datetime import date as Date
from datetime import datetime as Datetime
from dataclasses import dataclass

from finance.enumerations import Instrument, Option, Position

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Symbol", "Trade", "Quote", "History", "Settlement", "Contract"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


decimal_formatter = lambda value: f"{Decimal(str(value)):.2f}"
decimal_parser = lambda value: Decimal(str(value))
enum_parser = lambda concept: lambda value: concept(value)
date_formatter = lambda value: value.strftime("%Y%m%d")


def date_parser(value):
    assert isinstance(value, (str, Date, Datetime))
    if isinstance(value, Datetime): return value.date()
    elif isinstance(value, Date): return value
    else: return Datetime.strptime(str(value), "%Y%m%d").date()


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

    def __iter__(self): return iter([(field, self.data.get(field.name)) for field in self.record.fields])
    def __hash__(self): return hash((self.record, tuple(self.data.get(field.name) for field in self.record.fields)))

    def __str__(self):
        strings = [field.format(content) for field, content in iter(self)]
        return self.record.delimiter.join(strings).rstrip(self.record.delimiter)

    def __eq__(self, other):
        assert isinstance(other, Query)
        return self.record == other.record and all(self.data.get(field.name) == other.data.get(field.name) for field in self.record.fields)

    def __getattr__(self, name):
        try: return self.data[name]
        except KeyError: raise AttributeError(name) from None

    def items(self): return self.data.items()
    def values(self): return self.data.values()
    def keys(self): return self.data.keys()


Symbol = Record("Symbol", fields=(TickerField,))
Trade = Record("Trade", fields=(TickerField, PriceField))
Quote = Record("Quote", fields=(TickerField, BidField, AskField))
History = Record("History", fields=(TickerField, DateField))
Settlement = Record("Settlement", fields=(TickerField, ExpireField))
Contract = Record("Contract", fields=(TickerField, ExpireField, OptionField, StrikeField))
