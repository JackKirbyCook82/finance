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

from support.variables import Category, Variables, Variable
from support.querys import Field, Query
from support.mixins import Naming

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Variables", "Querys", "Securities", "Strategies", "OSI"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


TechnicalVariable = Variable("Technical", ["SECURITY", "TRADE", "QUOTE", "BARS", "STATISTIC", "STOCHASTIC"], start=1)
InstrumentVariable = Variable("Instrument", ["EMPTY", "STOCK", "OPTION"], start=0)
OptionVariable = Variable("Option", ["PUT", "EMPTY", "CALL"], start=-1)
PositionVariable = Variable("Position", ["SHORT", "EMPTY", "LONG"], start=-1)
SpreadVariable = Variable("Spread", ["STRANGLE", "COLLAR", "VERTICAL"], start=1)
ValuationVariable = Variable("Valuation", ["ARBITRAGE"], start=1)
ScenarioVariable = Variable("Scenario", ["MINIMUM", "MAXIMUM"], start=1)
StatusVariable = Variable("Status", ["PROSPECT", "PENDING", "OBSOLETE", "ABANDONED", "REJECTED", "ACCEPTED"], start=1)
TermVariable = Variable("Terms", ["MARKET", "LIMIT", "STOP", "STOPLIMIT", "LIMITDEBIT", "LIMITCREDIT"], start=1)
TenureVariable = Variable("Tenure", ["DAY", "STANDING", "OPENING", "CLOSING", "IMMEDIATE", "FILLKILL"], start=1)
ActionVariable = Variable("Action", ["BUY", "SELL"], start=1)
PricingVariable = Variable("Pricing", ["AGGRESSIVE", "PASSIVE", "MODERATE"], start=1)

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
AskField = Field("ask", numbers.Number, digits=2)
BidField = Field("bid", numbers.Number, digits=2)
OptionField = Field("option", Enum, variable=OptionVariable)

SymbolQuery = Query("Symbol", fields=[TickerField], delimiter="|")
TradeQuery = Query("Trade", fields=[TickerField, PriceField], delimiter="|")
QuoteQuery = Query("Quote", fields=[TickerField, BidField, AskField], delimiter="|")
ProductQuery = Query("Anchor", fields=[TickerField, ExpireField, PriceField], delimiter="|")
HistoryQuery = Query("History", fields=[TickerField, DateField], delimiter="|")
SettlementQuery = Query("Settlement", fields=[TickerField, ExpireField], delimiter="|")
ContractQuery = Query("Contract", fields=[TickerField, ExpireField, OptionField, StrikeField], delimiter="|")


class Querys(Category): Symbol, Settlement, Contract = SymbolQuery, SettlementQuery, ContractQuery
class Variables(Category):
    class Securities(Category): Security, Instrument, Option, Position = SecurityVariables, InstrumentVariable, OptionVariable, PositionVariable
    class Strategies(Category): Strategy, Spread = StrategyVariables, SpreadVariable
    class Valuations(Category): Valuation, Scenario = ValuationVariable, ScenarioVariable
    class Markets(Category): Status, Term, Tenure, Action, Pricing = StatusVariable, TermVariable, TenureVariable, ActionVariable, PricingVariable
    class Analysis(Category): Technical = TechnicalVariable

class Securities(Category):
    class Stocks(Category): Long = StockLongSecurity; Short = StockShortSecurity
    class Options(Category):
        class Puts(Category): Long = OptionPutLongSecurity; Short = OptionPutShortSecurity
        class Calls(Category): Long = OptionCallLongSecurity; Short = OptionCallShortSecurity

class Strategies(Category):
    class Verticals(Category): Put = VerticalPutStrategy; Call = VerticalCallStrategy
    class Collars(Category): Long = CollarLongStrategy; Short = CollarShortStrategy


class OSI(Naming, fields=["ticker", "expire", "option", "strike"]):
    def __new__(cls, contents):
        if isinstance(contents, list): mapping = {field: content for field, content in zip(cls.fields, contents)}
        elif isinstance(contents, dict): mapping = {field: contents[field] for field in cls.fields}
        elif isinstance(contents, ContractQuery): mapping = dict(list(contents))
        elif isinstance(contents, str): mapping = cls.parse(contents)
        else: raise TypeError(type(contents))
        return super().__new__(cls, **mapping)

    def __str__(self):
        ticker = str(self.ticker).upper()
        expire = str(self.expire.strftime("%y%m%d"))
        option = str(self.option).upper()[0]
        strike = str(self.strike).split(".")
        right = lambda value: str(value).rjust(5, "0")
        left = lambda value: str(value.ljust(3, "0"))
        strike = [function(value) for function, value in zip([right, left], strike)]
        return "".join([str(ticker), str(expire), str(option)] + strike)

    @classmethod
    def parse(cls, string):
        pattern = "^(?P<ticker>[A-Z]*)(?P<expire>[0-9]*)(?P<option>[PC]{1})(?P<strike>[0-9]*)$"
        values = re.search(pattern, string).groupdict()
        ticker = str(values["ticker"]).upper()
        expire = datetime.datetime.strptime(str(values["expire"]), "%y%m%d").date()
        option = {str(option).upper()[0]: option for option in OptionVariable}[str(values["option"])]
        strike = float(".".join([str(values["strike"])[:5], str(values["strike"])[5:]]))
        strike = np.round(float(strike), 3)
        return dict(ticker=ticker, expire=expire, option=option, strike=strike)


