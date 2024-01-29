# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 2024
@name:   Finance Variable Objects
@author: Jack Kirby Cook

"""

from enum import IntEnum
from support.dispatchers import typedispatcher
from datetime import date as Date
from collections import namedtuple as ntuple

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["DateRange", "Contract", "Securities", "Instruments", "Positions", "Strategies", "Valuations"]
__copyright__ = "Copyright 2023,SE Jack Kirby Cook"
__license__ = ""


Contract = ntuple("Contract", "ticker expire")
Instruments = IntEnum("Instruments", ["PUT", "CALL", "STOCK"], start=1)
Positions = IntEnum("Positions", ["LONG", "SHORT"], start=1)
Spreads = IntEnum("Strategy", ["STRANGLE", "COLLAR", "VERTICAL"], start=1)
Basis = IntEnum("Basis", ["ARBITRAGE"], start=1)
Scenarios = IntEnum("Scenarios", ["MINIMUM", "MAXIMUM", "CURRENT"], start=1)


class DateRange(ntuple("DateRange", "minimum maximum")):
    def __new__(cls, dates):
        assert isinstance(dates, list)
        assert all([isinstance(date, Date) for date in dates])
        dates = [date if isinstance(date, Date) else date.date() for date in dates]
        return super().__new__(cls, min(dates), max(dates)) if dates else None

    def __contains__(self, date): return self.minimum <= date <= self.maximum
    def __repr__(self): return f"{self.__class__.__name__}({repr(self.minimum)}, {repr(self.maximum)})"
    def __str__(self): return f"{str(self.minimum)}|{str(self.maximum)}"
    def __bool__(self): return self.minimum < self.maximum
    def __len__(self): return (self.maximum - self.minimum).days


class Security(ntuple("Security", "instrument position")):
    def __new__(cls, instrument, position, *args, **kwargs): return super().__new__(cls, instrument, position)
    def __str__(self): return "|".join([str(value.name).lower() for value in self if bool(value)])
    def __int__(self): return int(self.instrument) * 10 + int(self.position) * 1

    @property
    def title(self): return "|".join([str(string).title() for string in str(self).split("|")])
    @property
    def payoff(self): return self.__payoff


class Strategy(ntuple("Strategy", "spread instrument position")):
    def __new__(cls, spread, instrument, position, *args, **kwargs): return super().__new__(cls, spread, instrument, position)
    def __init__(self, *args, **kwargs): self.__securities = kwargs["securities"]
    def __str__(self): return "|".join([str(value.name).lower() for value in self if bool(value)])
    def __int__(self): return int(self.spread) * 100 + int(self.instrument) * 10 + int(self.position) * 1

    @property
    def title(self): return "|".join([str(string).title() for string in str(self).split("|")])
    @property
    def securities(self): return self.__securities


class Valuation(ntuple("Valuation", "basis scenario")):
    def __str__(self): return "|".join([str(value.name).lower() for value in self if bool(value)])
    def __int__(self): return int(self.basis) * 10 + int(self.scenario) * 1

    @property
    def title(self): return "|".join([str(string).title() for string in str(self).split("|")])


StockLong = Security(Instruments.STOCK, Positions.LONG)
StockShort = Security(Instruments.STOCK, Positions.SHORT)
PutLong = Security(Instruments.PUT, Positions.LONG)
PutShort = Security(Instruments.PUT, Positions.SHORT)
CallLong = Security(Instruments.CALL, Positions.LONG,)
CallShort = Security(Instruments.CALL, Positions.SHORT)
StrangleLong = Strategy(Spreads.STRANGLE, 0, Positions.LONG, securities=[PutLong, CallLong])
CollarLong = Strategy(Spreads.COLLAR, 0, Positions.LONG, securities=[PutLong, CallShort, StockLong])
CollarShort = Strategy(Spreads.COLLAR, 0, Positions.SHORT, securities=[CallLong, PutShort, StockShort])
VerticalPut = Strategy(Spreads.VERTICAL, Instruments.PUT, 0, securities=[PutLong, PutShort])
VerticalCall = Strategy(Spreads.VERTICAL, Instruments.CALL, 0, securities=[CallLong, CallShort])
MinimumArbitrage = Valuation(Basis.ARBITRAGE, Scenarios.MINIMUM)
MaximumArbitrage = Valuation(Basis.ARBITRAGE, Scenarios.MAXIMUM)
CurrentArbitrage = Valuation(Basis.ARBITRAGE, Scenarios.CURRENT)


class SecuritiesMeta(type):
    def __iter__(cls): return iter([StockLong, StockShort, PutLong, PutShort, CallLong, CallShort])
    def __getitem__(cls, indexkey): return cls.retrieve(indexkey)

    @typedispatcher
    def retrieve(cls, indexkey): raise TypeError(type(indexkey).__name__)
    @retrieve.register(int)
    def integer(cls, index): return {int(security): security for security in iter(cls)}[index]
    @retrieve.register(str)
    def string(cls, string): return {str(security): security for security in iter(cls)}[str(string).lower()]
    @retrieve.register(tuple)
    def value(cls, value): return {(security.instrument, security.postion): security for security in iter(cls)}[value]

    @property
    def Stocks(cls): return iter([StockLong, StockShort])
    @property
    def Options(cls): return iter([PutLong, PutShort, CallLong, CallShort])
    @property
    def Puts(cls): return iter([PutLong, PutShort])
    @property
    def Calls(cls): return iter([CallLong, CallShort])

    class Stock:
        Long = StockLong
        Short = StockShort
    class Option:
        class Put:
            Long = PutLong
            Short = PutShort
        class Call:
            Long = CallLong
            Short = CallShort


class StrategiesMeta(type):
    def __iter__(cls): return iter([StrangleLong, CollarLong, CollarShort, VerticalPut, VerticalCall])
    def __getitem__(cls, indexkey): return cls.retrieve(indexkey)

    @typedispatcher
    def retrieve(cls, indexkey): raise TypeError(type(indexkey).__name__)
    @retrieve.register(int)
    def integer(cls, index): return {int(strategy): strategy for strategy in iter(cls)}[index]
    @retrieve.register(str)
    def string(cls, string): return {str(strategy): strategy for strategy in iter(cls)}[str(string).lower()]
    @retrieve.register(tuple)
    def value(cls, value): return {str(strategy): strategy for strategy in iter(cls)}[value]
    @retrieve.register(tuple)
    def value(cls, value): return {(strategy.spread, strategy.instrument, strategy.postion): strategy for strategy in iter(cls)}[value]

    class Strangle:
        Long = StrangleLong
    class Collar:
        Long = CollarLong
        Short = CollarShort
    class Vertical:
        Put = VerticalPut
        Call = VerticalCall


class ValuationsMeta(type):
    def __iter__(cls): return iter([MinimumArbitrage, MaximumArbitrage, CurrentArbitrage])
    def __getitem__(cls, indexkey): return cls.retrieve(indexkey)

    @typedispatcher
    def retrieve(cls, indexkey): raise TypeError(type(indexkey).__name__)
    @retrieve.register(int)
    def integer(cls, index): return {int(valuation): valuation for valuation in iter(cls)}[index]
    @retrieve.register(str)
    def string(cls, string): return {int(valuation): valuation for valuation in iter(cls)}[str(string).lower()]
    @retrieve.register(tuple)
    def value(cls, value): return {(valuation.basis, valuation.scenario): valuation for valuation in iter(cls)}[value]

    @property
    def Arbitrages(cls): return iter([MinimumArbitrage, MaximumArbitrage, CurrentArbitrage])

    class Arbitrage:
        Minimum = MinimumArbitrage
        Maximum = MaximumArbitrage
        Current = CurrentArbitrage


class Securities(object, metaclass=SecuritiesMeta): pass
class Strategies(object, metaclass=StrategiesMeta): pass
class Valuations(object, metaclass=ValuationsMeta): pass


