# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 2024
@name:   Finance Variable Objects
@author: Jack Kirby Cook

"""

import numbers
import datetime
import regex as re
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod

from support.variables import Category, Variables, Variable
from support.querys import Field, Query
from support.meta import MappingMeta
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = []
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class OSI(ABC):
    def toOSI(self):
        ticker = str(self.ticker).upper()
        expire = str(self.expire.strftime("%y%m%d"))
        option = str(self.option).upper()[0]
        strike = str(self.strike).split(".")
        right = lambda value: str(value).rjust(5, "0")
        left = lambda value: str(value.ljust(3, "0"))
        strike = [function(value) for function, value in zip([right, left], strike)]
        return "".join([str(ticker), str(expire), str(option)] + strike)

    @classmethod
    def fromOSI(cls, string):
        pattern = "^(?P<ticker>[A-Z]*)(?P<expire>[0-9]*)(?P<option>[PC]{1})(?P<strike>[0-9]*)$"
        values = re.match(pattern).groupdict()
        ticker = str(["ticker"]).upper()
        expire = np.datetime64(datetime.datetime.strptime(str(values["expire"]), "%y%m%d").date(), "D")
        option = {str(option).upper()[0]: option for option in Variables.Options}[str(values["option"])]
        strike = np.float32(".".join([str(values["strike"]), str(values["strike"])]))
        return cls(ticker, expire, option, strike)

    @property
    @abstractmethod
    def ticker(self): pass
    @property
    @abstractmethod
    def expire(self): pass
    @property
    @abstractmethod
    def option(self): pass
    @property
    @abstractmethod
    def strike(self): pass


ThetaVariable = Variable("Theta", ["PUT", "NEUTRAL", "CALL"], start=-1)
PhiVariable = Variable("Phi", ["SHORT", "NEUTRAL", "LONG"], start=-1)
OmegaVariable = Variable("Omega", ["BEAR", "NEUTRAL", "BULL"], start=-1)

StatusVariable = Variable("Status", ["PROSPECT", "PENDING", "OBSOLETE", "ABANDONED", "REJECTED", "ACCEPTED"], start=1)
ScenarioVariable = Variable("Scenario", ["MINIMUM", "MAXIMUM"], start=1)
ValuationVariable = Variable("Valuation", ["ARBITRAGE"], start=1)

TermsVariable = Variable("Terms", ["MARKET", "LIMIT", "STOP", "STOPLIMIT", "LIMITDEBIT", "LIMITCREDIT"], start=1)
TechnicalVariable = Variable("Technical", ["STATISTIC", "STOCHASTIC"], start=1)
ActionVariable = Variable("Action", ["BUY", "SELL"], start=1)

MarketVariable = Variable("Market", ["EMPTY", "BEAR", "BULL"], start=0)
InstrumentVariable = Variable("Instrument", ["EMPTY", "STOCK", "OPTION"], start=0)
OptionVariable = Variable("Option", ["EMPTY", "PUT", "CALL"], start=0)
PositionVariable = Variable("Position", ["EMPTY", "LONG", "SHORT"], start=0)
SpreadVariable = Variable("Spread", ["STRANGLE", "COLLAR", "VERTICAL"], start=1)

SecurityVariables = Variables("Security", ["instrument", "option", "position"])
StrategyVariables = Variables("Strategy", ["spread", "option", "position"], {"stocks", "options"})

StockLongSecurity = SecurityVariables("StockLong", [InstrumentVariable.STOCK, OptionVariable.EMPTY, PositionVariable.LONG])
StockShortSecurity = SecurityVariables("StockShort", [InstrumentVariable.STOCK, OptionVariable.EMPTY, PositionVariable.SHORT])
OptionPutLongSecurity = SecurityVariables("OptionPutLong", [InstrumentVariable.OPTION, OptionVariable.PUT, PositionVariable.LONG])
OptionPutShortSecurity = SecurityVariables("OptionPutShort", [InstrumentVariable.OPTION, OptionVariable.PUT, PositionVariable.SHORT])
OptionCallLongSecurity = SecurityVariables("OptionCallLong", [InstrumentVariable.OPTION, OptionVariable.CALL, PositionVariable.LONG])
OptionCallShortSecurity = SecurityVariables("OptionCallShort", [InstrumentVariable.OPTION, OptionVariable.CALL, PositionVariable.SHORT])
VerticalPutSecurity = SecurityVariables("VerticalPut", [SpreadVariable.VERTICAL, OptionVariable.PUT, PositionVariable.EMPTY], options=[OptionPutLongSecurity, OptionPutShortSecurity], stocks=[])
VerticalCallSecurity = SecurityVariables("VerticalCall", [SpreadVariable.VERTICAL, OptionVariable.CALL, PositionVariable.EMPTY], options=[OptionCallLongSecurity, OptionCallShortSecurity], stocks=[])
CollarLongSecurity = SecurityVariables("CollarLong", [SpreadVariable.COLLAR, OptionVariable.EMPTY, PositionVariable.LONG], options=[OptionPutLongSecurity, OptionCallShortSecurity], stocks=[StockLongSecurity])
CollarShortSecurity = SecurityVariables("CollarShort", [SpreadVariable.COLLAR, OptionVariable.EMPTY, PositionVariable.SHORT], options=[OptionCallLongSecurity, OptionPutShortSecurity], stocks=[StockShortSecurity])

TickerField = Field("ticker", str)
ExpireField = Field("expire", datetime.date, format="%Y%m%d")
StrikeField = Field("strike", numbers.Number, digits=2)
OptionField = Field("option", Enum, variable=OptionVariable)

SymbolQuery = Query("Symbol", fields=[TickerField], delimiter="|")
SettlementQuery = Query("Future", fields=[TickerField, ExpireField], delimiter="|")
ContractQuery = Query("Contract", bases=[OSI], fields=[TickerField, ExpireField, OptionField, StrikeField], delimiter="|")


class Parameters(metaclass=MappingMeta):
    types = {key: np.float32 for key in ("strike", "price", "bid", "ask", "demand", "supply", "open", "close", "high", "low", "trend", "volatility", "oscillator")}
    types = dict(ticker=str, volume=np.int64) | dict(types)
    formatters = dict(instrument=int, option=int, position=int)
    parsers = dict(instrument=InstrumentVariable, option=OptionVariable, position=PositionVariable)
    dates = dict(date="Y%m%d", expire="Y%m%d", current="%Y%m%d-%H%M")

class StockTradeFile(File, variable=(STOCK, TRADE), **dict(Parameters), order=["ticker", "current", "price"]): pass
class StockQuoteFile(File, variable=(STOCK, QUOTE), **dict(Parameters), order=["ticker", "current", "bid", "ask", "demand", "supply"]): pass
class StockBarsFile(File, variable=(STOCK, BARS), **dict(Parameters), order=["ticker", "date", "open", "close", "high", "low", "price"]): pass
class StockStatisticFile(File, variable=(STOCK, STATISTIC), **dict(Parameters), order=["ticker", "date", "price", "trend", "volatility"]): pass
class StockStochasticFile(File, variable=(STOCK, STOCHASTIC), **dict(Parameters), order=["ticker", "date", "price", "oscillator"]): pass
class OptionTradeFile(File, variable=(OPTION, TRADE), **dict(Parameters), order=["ticker", "expire", "strike", "option", "current", "price", "underlying"]): pass
class OptionQuoteFile(File, variable=(OPTION, QUOTE), **dict(Parameters), order=["ticker", "expire", "strike", "option", "current", "bid", "ask", "demand", "supply", "underlying"]): pass


class Files(Category):
    class Stocks(Category): Trade = StockTradeFile, Quote = StockQuoteFile, Bars = StockBarsFile
    class Options(Category): Trade = OptionTradeFile, Quote = OptionQuoteFile

class Securities(Category):
    class Stocks(Category): Long = StockLongSecurity; Short = StockShortSecurity
    class Options(Category):
        class Puts(Category): Long = OptionPutLongSecurity; Short = OptionPutShortSecurity
        class Calls(Category): Long = OptionCallLongSecurity; Short = OptionCallShortSecurity

class Strategies(Category):
    class Verticals(Category): Put = VerticalPutSecurity; Call = VerticalCallSecurity
    class Collars(Category): Long = CollarLongSecurity; Short = CollarShortSecurity











