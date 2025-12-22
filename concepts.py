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

from support.concepts import Assembly, Concepts, Concept
from support.querys import Field, Query
from support.mixins import Naming

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Concepts", "Querys", "Securities", "Strategies", "OSI"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


AppraisalConcept = Concept("Appraisal", ["BLACKSCHOLES", "GREEKS"], start=1)
InstrumentConcept = Concept("Instrument", ["EMPTY", "STOCK", "OPTION"], start=0)
OptionConcept = Concept("Option", ["PUT", "EMPTY", "CALL"], start=-1)
PositionConcept = Concept("Position", ["SHORT", "EMPTY", "LONG"], start=-1)
SpreadConcept = Concept("Spread", ["EMPTY", "STRANGLE", "COLLAR", "VERTICAL"], start=0)
ValuationConcept = Concept("Valuation", ["ARBITRAGE", "RISKY", "WORTHLESS"], start=1)
ScenarioConcept = Concept("Scenario", ["MINIMUM", "MAXIMUM"], start=1)
StatusConcept = Concept("Status", ["PROSPECT", "PENDING", "OBSOLETE", "ABANDONED", "REJECTED", "ACCEPTED"], start=1)
TermConcept = Concept("Terms", ["MARKET", "LIMIT", "STOP", "STOPLIMIT", "LIMITDEBIT", "LIMITCREDIT"], start=1)
TenureConcept = Concept("Tenure", ["DAY", "STANDING", "OPENING", "CLOSING", "IMMEDIATE", "FILLKILL"], start=1)
PricingConcept = Concept("Pricing", ["AGGRESSIVE", "PASSIVE", "MODERATE"], start=1)
QuotingConcept = Concept("Quoting", ["CLOSING", "DELAYED", "REALTIME"], start=1)
MarketConcept = Concept("Market", ["BEAR", "NEUTRAL", "BULL"], start=-1)
ActionConcept = Concept("Action", ["BUY", "SELL"], start=1)

SecurityConcepts = Concepts("Security", ["instrument", "option", "position"])
StrategyConcepts = Concepts("Strategy", ["spread", "option", "position"], {"stocks", "options"})

TickerField = Field("ticker", str)
DateField = Field("date", datetime.date, formatting="%Y%m%d")
ExpireField = Field("expire", datetime.date, formatting="%Y%m%d")
StrikeField = Field("strike", numbers.Number, digits=2)
PriceField = Field("price", numbers.Number, digits=2)
AskField = Field("ask", numbers.Number, digits=2)
BidField = Field("bid", numbers.Number, digits=2)
InstrumentField = Field("instrument", Enum, variable=InstrumentConcept)
OptionField = Field("option", Enum, variable=OptionConcept)
PositionField = Field("position", Enum, variable=PositionConcept)

SymbolQuery = Query("Symbol", fields=[TickerField], delimiter="|")
TradeQuery = Query("Trade", fields=[TickerField, PriceField], delimiter="|")
QuoteQuery = Query("Quote", fields=[TickerField, BidField, AskField], delimiter="|")
ProductQuery = Query("Anchor", fields=[TickerField, ExpireField, PriceField], delimiter="|")
HistoryQuery = Query("History", fields=[TickerField, DateField], delimiter="|")
SettlementQuery = Query("Settlement", fields=[TickerField, ExpireField], delimiter="|")
ContractQuery = Query("Contract", fields=[TickerField, ExpireField, OptionField, StrikeField], delimiter="|")

StockLongSecurity = SecurityConcepts("StockLong", [InstrumentConcept.STOCK, OptionConcept.EMPTY, PositionConcept.LONG])
StockShortSecurity = SecurityConcepts("StockShort", [InstrumentConcept.STOCK, OptionConcept.EMPTY, PositionConcept.SHORT])
OptionPutLongSecurity = SecurityConcepts("OptionPutLong", [InstrumentConcept.OPTION, OptionConcept.PUT, PositionConcept.LONG])
OptionPutShortSecurity = SecurityConcepts("OptionPutShort", [InstrumentConcept.OPTION, OptionConcept.PUT, PositionConcept.SHORT])
OptionCallLongSecurity = SecurityConcepts("OptionCallLong", [InstrumentConcept.OPTION, OptionConcept.CALL, PositionConcept.LONG])
OptionCallShortSecurity = SecurityConcepts("OptionCallShort", [InstrumentConcept.OPTION, OptionConcept.CALL, PositionConcept.SHORT])

VerticalPutStrategy = StrategyConcepts("VerticalPut", [SpreadConcept.VERTICAL, OptionConcept.PUT, PositionConcept.EMPTY], options=[OptionPutLongSecurity, OptionPutShortSecurity], stocks=[])
VerticalCallStrategy = StrategyConcepts("VerticalCall", [SpreadConcept.VERTICAL, OptionConcept.CALL, PositionConcept.EMPTY], options=[OptionCallLongSecurity, OptionCallShortSecurity], stocks=[])
CollarLongStrategy = StrategyConcepts("CollarLong", [SpreadConcept.COLLAR, OptionConcept.EMPTY, PositionConcept.LONG], options=[OptionPutLongSecurity, OptionCallShortSecurity], stocks=[StockLongSecurity])
CollarShortStrategy = StrategyConcepts("CollarShort", [SpreadConcept.COLLAR, OptionConcept.EMPTY, PositionConcept.SHORT], options=[OptionCallLongSecurity, OptionPutShortSecurity], stocks=[StockShortSecurity])


class Querys(Assembly): Symbol, Settlement, Contract = SymbolQuery, SettlementQuery, ContractQuery
class Concepts(Assembly):
    class Securities(Assembly): Security, Instrument, Option, Position = SecurityConcepts, InstrumentConcept, OptionConcept, PositionConcept
    class Strategies(Assembly): Strategy, Spread = StrategyConcepts, SpreadConcept
    class Markets(Assembly): Status, Term, Tenure, Action, Quoting = StatusConcept, TermConcept, TenureConcept, ActionConcept, QuotingConcept
    Pricing = PricingConcept
    Appraisal = AppraisalConcept
    Technical = TechnicalConcept
    Valuation = ValuationConcept
    Scenario = ScenarioConcept
    Market = MarketConcept

class Securities(Assembly):
    class Stocks(Assembly): Long = StockLongSecurity; Short = StockShortSecurity
    class Options(Assembly):
        class Puts(Assembly): Long = OptionPutLongSecurity; Short = OptionPutShortSecurity
        class Calls(Assembly): Long = OptionCallLongSecurity; Short = OptionCallShortSecurity

class Strategies(Assembly):
    class Verticals(Assembly): Put = VerticalPutStrategy; Call = VerticalCallStrategy
    class Collars(Assembly): Long = CollarLongStrategy; Short = CollarShortStrategy


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
        option = {str(option).upper()[0]: option for option in OptionConcept}[str(values["option"])]
        strike = float(".".join([str(values["strike"])[:5], str(values["strike"])[5:]]))
        strike = np.round(float(strike), 3)
        return dict(ticker=ticker, expire=expire, option=option, strike=strike)


