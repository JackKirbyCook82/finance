# -*- coding: utf-8 -*-
"""
Created on Weds May 27 2026
@name:   Finance Specifications Objects
@author: Jack Kirby Cook

"""

from typing import ClassVar
from dataclasses import dataclass
from abc import ABC, abstractmethod

from finance.enumerations import Spread, Instrument, Option, Position

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Securities", "Strategies"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


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
    instrument: Instrument
    option: Option = Option.EMPTY
    position: Position = Position.EMPTY

    @property
    def key(self): return self.instrument, self.option, self.position


@dataclass(frozen=True, slots=True)
class Strategy(Specification):
    spread: Spread
    option: Option = Option.EMPTY
    position: Position = Position.EMPTY

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


