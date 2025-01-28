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
__all__ = ["Variables", "Querys", "Files", "Securities", "Strategies"]
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


TechnicalVariable = Variable("Technical", ["TRADE", "QUOTE", "BARS", "STATISTIC", "STOCHASTIC"], start=1)
InstrumentVariable = Variable("Instrument", ["EMPTY", "STOCK", "OPTION"], start=0)
OptionVariable = Variable("Option", ["EMPTY", "PUT", "CALL"], start=0)
PositionVariable = Variable("Position", ["EMPTY", "LONG", "SHORT"], start=0)
SpreadVariable = Variable("Spread", ["STRANGLE", "COLLAR", "VERTICAL"], start=1)
ValuationVariable = Variable("Valuation", ["ARBITRAGE"], start=1)
ScenarioVariable = Variable("Scenario", ["MINIMUM", "MAXIMUM"], start=1)
StatusVariable = Variable("Status", ["PROSPECT", "PENDING", "OBSOLETE", "ABANDONED", "REJECTED", "ACCEPTED"], start=1)
TermsVariable = Variable("Terms", ["MARKET", "LIMIT", "STOP", "STOPLIMIT", "LIMITDEBIT", "LIMITCREDIT"], start=1)
ActionVariable = Variable("Action", ["BUY", "SELL"], start=1)
ThetaVariable = Variable("Theta", ["PUT", "NEUTRAL", "CALL"], start=-1)
PhiVariable = Variable("Phi", ["SHORT", "NEUTRAL", "LONG"], start=-1)
OmegaVariable = Variable("Omega", ["BEAR", "NEUTRAL", "BULL"], start=-1)

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
DateField = Field("date", datetime.date, format="%Y%m%d")
ExpireField = Field("expire", datetime.date, format="%Y%m%d")
StrikeField = Field("strike", numbers.Number, digits=2)
OptionField = Field("option", Enum, variable=OptionVariable)

SymbolQuery = Query("Symbol", fields=[TickerField], delimiter="|")
HistoryQuery = Query("History", fields=[TickerField, DateField], delimiter="|")
SettlementQuery = Query("Future", fields=[TickerField, ExpireField], delimiter="|")
ContractQuery = Query("Contract", bases=[OSI], fields=[TickerField, ExpireField, OptionField, StrikeField], delimiter="|")


class Parameters(metaclass=MappingMeta):
    types = ("strike", "price", "quantity") + ("trend", "volatility", "oscillator") + ("spot", "minimum", "maximum") + ("open", "close", "high", "low") + ("bid", "ask", "demand", "supply")
    types = {key: np.float32 for key in types.items()}
    types = dict(ticker=str, volume=np.int64) | dict(types)
    formatters = dict(instrument=int, option=int, position=int)
    parsers = dict(instrument=InstrumentVariable, option=OptionVariable, position=PositionVariable)
    dates = dict(date="Y%m%d", expire="Y%m%d", current="%Y%m%d-%H%M")

class StockTradeFile(File, order=["ticker", "current", "price"], **dict(Parameters)): pass
class StockQuoteFile(File, order=["ticker", "current", "bid", "ask", "demand", "supply"], **dict(Parameters)): pass
class StockBarsFile(File, order=["ticker", "date", "open", "close", "high", "low", "price"], **dict(Parameters)): pass
class StockStatisticFile(File, order=["ticker", "date", "price", "trend", "volatility"], **dict(Parameters)): pass
class StockStochasticFile(File, order=["ticker", "date", "price", "oscillator"], **dict(Parameters)): pass
class OptionTradeFile(File, order=["ticker", "expire", "strike", "option", "current", "price", "underlying"], **dict(Parameters)): pass
class OptionQuoteFile(File, order=["ticker", "expire", "strike", "option", "current", "bid", "ask", "demand", "supply", "underlying"], **dict(Parameters)): pass
class HoldingsFile(File, order=["ticker", "expire", "strike", "instrument", "option", "position", "quantity"], **dict(Parameters)): pass


class Querys(Category): Symbol, History, Settlement, Contract = SymbolQuery, HistoryQuery, SettlementQuery, ContractQuery
class Variables(Category):
    class Securities(Category): Security, Instrument, Option, Position = SecurityVariables, InstrumentVariable, OptionVariable, PositionVariable
    class Strategies(Category): Strategy, Spread = StrategyVariables, SpreadVariable
    class Valuations(Category): Valuation, Scenario = ValuationVariable, ScenarioVariable
    class Markets(Category): Status, Terms, Action = StatusVariable, TermsVariable, ActionVariable
    class Greeks(Category): Theta, Phi, Omega = ThetaVariable, PhiVariable, OmegaVariable
    class Analysis(Category): Technical = TechnicalVariable

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












