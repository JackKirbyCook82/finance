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
from collections import namedtuple as ntuple

from support.variables import Category, Variables, Variable
from support.decorators import TypeDispatcher
from support.querys import Field, Query
from support.meta import MappingMeta
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Variables", "Querys", "Files", "Securities", "Strategies", "OSI"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


TechnicalVariable = Variable("Technical", ["SECURITY", "TRADE", "QUOTE", "BARS", "STATISTIC", "STOCHASTIC"], start=1)
InstrumentVariable = Variable("Instrument", ["EMPTY", "STOCK", "OPTION"], start=0)
OptionVariable = Variable("Option", ["EMPTY", "PUT", "CALL"], start=0)
PositionVariable = Variable("Position", ["EMPTY", "LONG", "SHORT"], start=0)
SpreadVariable = Variable("Spread", ["STRANGLE", "COLLAR", "VERTICAL"], start=1)
ValuationVariable = Variable("Valuation", ["ARBITRAGE"], start=1)
ScenarioVariable = Variable("Scenario", ["MINIMUM", "MAXIMUM"], start=1)
StatusVariable = Variable("Status", ["PROSPECT", "PENDING", "OBSOLETE", "ABANDONED", "REJECTED", "ACCEPTED"], start=1)
TermsVariable = Variable("Terms", ["MARKET", "LIMIT", "STOP", "STOPLIMIT", "LIMITDEBIT", "LIMITCREDIT"], start=1)
ActionVariable = Variable("Action", ["BUY", "SELL"], start=1)
PricingVariable = Variable("Pricing", ["MARKET", "LIMIT"], start=1)
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
VerticalPutStrategy = StrategyVariables("VerticalPut", [SpreadVariable.VERTICAL, OptionVariable.PUT, PositionVariable.EMPTY], options=[OptionPutLongSecurity, OptionPutShortSecurity], stocks=[])
VerticalCallStrategy = StrategyVariables("VerticalCall", [SpreadVariable.VERTICAL, OptionVariable.CALL, PositionVariable.EMPTY], options=[OptionCallLongSecurity, OptionCallShortSecurity], stocks=[])
CollarLongStrategy = StrategyVariables("CollarLong", [SpreadVariable.COLLAR, OptionVariable.EMPTY, PositionVariable.LONG], options=[OptionPutLongSecurity, OptionCallShortSecurity], stocks=[StockLongSecurity])
CollarShortStrategy = StrategyVariables("CollarShort", [SpreadVariable.COLLAR, OptionVariable.EMPTY, PositionVariable.SHORT], options=[OptionCallLongSecurity, OptionPutShortSecurity], stocks=[StockShortSecurity])


TickerField = Field("ticker", str)
DateField = Field("date", datetime.date, formatting="%Y%m%d")
ExpireField = Field("expire", datetime.date, formatting="%Y%m%d")
StrikeField = Field("strike", numbers.Number, digits=2)
PriceField = Field("price", numbers.Number, digits=2)
OptionField = Field("option", Enum, variable=OptionVariable)

SymbolQuery = Query("Symbol", fields=[TickerField], delimiter="|")
TradeField = Query("Trade", fields=[TickerField, PriceField], delimiter="|")
ProductQuery = Query("Anchor", fields=[TickerField, ExpireField, PriceField], delimiter="|")
HistoryQuery = Query("History", fields=[TickerField, DateField], delimiter="|")
SettlementQuery = Query("Future", fields=[TickerField, ExpireField], delimiter="|")
ContractQuery = Query("Contract", fields=[TickerField, ExpireField, OptionField, StrikeField], delimiter="|")


class Parameters(metaclass=MappingMeta):
    types = {"ticker": str, "strike price underlying bid ask size supply demand open close high low": np.float32, "trend volatility oscillator": np.float32, "volume": np.int64}
    types = {key: value for keys, value in types.items() for key in str(keys).split(" ")}
    parsers = dict(instrument=InstrumentVariable, option=OptionVariable, position=PositionVariable)
    formatters = dict(instrument=int, option=int, position=int)
    dates = dict(date="%Y%m%d", expire="%Y%m%d", current="%Y%m%d-%H%M")

class StockTradeFile(File, order=["ticker", "current", "price"], **dict(Parameters)): pass
class StockQuoteFile(File, order=["ticker", "current", "bid", "ask", "demand", "supply"], **dict(Parameters)): pass
class StockSecurityFile(File, order=["ticker", "position", "current", "price", "size"], **dict(Parameters)): pass
class StockBarsFile(File, order=["ticker", "date", "open", "close", "high", "low", "price"], **dict(Parameters)): pass
class StockStatisticFile(File, order=["ticker", "date", "price", "trend", "volatility"], **dict(Parameters)): pass
class StockStochasticFile(File, order=["ticker", "date", "price", "oscillator"], **dict(Parameters)): pass
class OptionTradeFile(File, order=["ticker", "expire", "strike", "option", "current", "price", "underlying"], **dict(Parameters)): pass
class OptionQuoteFile(File, order=["ticker", "expire", "strike", "option", "current", "bid", "ask", "demand", "supply"], **dict(Parameters)): pass
class OptionSecurityFile(File, order=["ticker", "expire", "strike", "instrument", "option", "position", "current", "price", "underlying", "size"], **dict(Parameters)): pass


class Files(Category):
    class Stocks(Category): Trade, Quote, Bars, Statistic, Stochastic = StockTradeFile, StockQuoteFile, StockBarsFile, StockStatisticFile, StockStochasticFile
    class Options(Category): Trade, Quote, Security = OptionTradeFile, OptionQuoteFile, OptionSecurityFile

class Querys(Category): Symbol, Trade, Product, History, Settlement, Contract = SymbolQuery, TradeField, ProductQuery, HistoryQuery, SettlementQuery, ContractQuery
class Variables(Category):
    class Securities(Category): Security, Instrument, Option, Position = SecurityVariables, InstrumentVariable, OptionVariable, PositionVariable
    class Strategies(Category): Strategy, Spread = StrategyVariables, SpreadVariable
    class Valuations(Category): Valuation, Scenario = ValuationVariable, ScenarioVariable
    class Markets(Category): Status, Terms, Action, Pricing = StatusVariable, TermsVariable, ActionVariable, PricingVariable
    class Greeks(Category): Theta, Phi, Omega = ThetaVariable, PhiVariable, OmegaVariable
    class Analysis(Category): Technical = TechnicalVariable

class Securities(Category):
    class Stocks(Category): Long = StockLongSecurity; Short = StockShortSecurity
    class Options(Category):
        class Puts(Category): Long = OptionPutLongSecurity; Short = OptionPutShortSecurity
        class Calls(Category): Long = OptionCallLongSecurity; Short = OptionCallShortSecurity

class Strategies(Category):
    class Verticals(Category): Put = VerticalPutStrategy; Call = VerticalCallStrategy
    class Collars(Category): Long = CollarLongStrategy; Short = CollarShortStrategy


class OSIMeta(type):
    def __call__(cls, content):
        content = cls.parse(content)
        assert isinstance(content, list) and len(content) == 4
        instance = super(OSIMeta, cls).__call__(*content)
        return instance

    @TypeDispatcher(locator=0)
    def parse(cls, content): raise TypeDispatcher(type(content))
    @parse.register(dict)
    def __mapping(cls, content): return [content[field] for field in cls._fields]
    @parse.register(list)
    def __collection(cls, content): return content
    @parse.register(str)
    def __string(cls, content):
        pattern = "^(?P<ticker>[A-Z]*)(?P<expire>[0-9]*)(?P<option>[PC]{1})(?P<strike>[0-9]*)$"
        values = re.search(pattern, content).groupdict()
        ticker = str(values["ticker"]).upper()
        expire = datetime.datetime.strptime(str(values["expire"]), "%y%m%d")
        option = {str(option).upper()[0]: option for option in OptionVariable}[str(values["option"])]
        strike = float(".".join([str(values["strike"])[:5], str(values["strike"])[5:]]))
        return [ticker, expire, option, strike]

class OSI(ntuple("OSI", "ticker expire option strike"), metaclass=OSIMeta):
    def __str__(self):
        ticker = str(self.ticker).upper()
        expire = str(self.expire.strftime("%y%m%d"))
        option = str(self.option).upper()[0]
        strike = str(self.strike).split(".")
        right = lambda value: str(value).rjust(5, "0")
        left = lambda value: str(value.ljust(3, "0"))
        strike = [function(value) for function, value in zip([right, left], strike)]
        return "".join([str(ticker), str(expire), str(option)] + strike)



