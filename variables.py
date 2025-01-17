# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 2024
@name:   Finance Variable Objects
@author: Jack Kirby Cook

"""

import numbers
import datetime

from support.variables import Category, Variables, Variable
from support.querys import Field, Query

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Categories", "Variables", "Querys", "Scenarios"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


Theta = Variable("Theta", ["PUT", "NEUTRAL", "CALL"], start=-1)
Phi = Variable("Phi", ["SHORT", "NEUTRAL", "LONG"], start=-1)
Omega = Variable("Omega", ["BEAR", "NEUTRAL", "BULL"], start=-1)

Status = Variable("Status", ["PROSPECT", "PENDING", "OBSOLETE", "ABANDONED", "REJECTED", "ACCEPTED"])
Technicals = Variable("Technicals", ["HISTORY", "STATISTIC", "STOCHASTIC"])
Scenarios = Variable("Scenarios", ["MINIMUM", "MAXIMUM"])
Valuations = Variable("Valuations", ["ARBITRAGE"])

Terms = Variable("Terms", ["MARKET", "LIMIT", "STOP", "STOPLIMIT", "LIMITDEBIT", "LIMITCREDIT"], start=1)
Actions = Variable("Action", ["BUY", "SELL"], start=1)
Pricing = Variable("Pricing", ["BLACKSCHOLES"])

Markets = Variable("Markets", ["EMPTY", "BEAR", "BULL"], start=0)
Instruments = Variable("Instruments", ["EMPTY", "STOCK", "OPTION"], start=0)
Options = Variable("Options", ["EMPTY", "PUT", "CALL"], start=0)
Positions = Variable("Positions", ["EMPTY", "LONG", "SHORT"], start=0)
Spreads = Variable("Spreads", ["STRANGLE", "COLLAR", "VERTICAL"], start=1)

Security = Variables("Security", ["instrument", "option", "position"])
Strategy = Variables("Strategy", ["spread", "option", "position"], {"stocks", "options"})

Ticker = Field("ticker", str)
Option = Field("option", Options)
Date = Field("date", datetime.date, format="%Y%m%d")
Expire = Field("expire", datetime.date, format="%Y%m%d")
Strike = Field("strike", numbers.Number, digits=2)

Symbol = Query("Symbol", [Ticker], delimiter="|")
History = Query("History", [Ticker, Date], delimiter="|")
Contract = Query("Contract", [Ticker, Expire], delimiter="|")
Product = Query("Product", [Ticker, Expire, Strike, Option], delimiter="|")

StockLong = Security("StockLong", [Instruments.STOCK, Options.EMPTY, Positions.LONG])
StockShort = Security("StockShort", [Instruments.STOCK, Options.EMPTY, Positions.SHORT])
OptionPutLong = Security("OptionPutLong", [Instruments.OPTION, Options.PUT, Positions.LONG])
OptionPutShort = Security("OptionPutShort", [Instruments.OPTION, Options.PUT, Positions.SHORT])
OptionCallLong = Security("OptionCallLong", [Instruments.OPTION, Options.CALL, Positions.LONG])
OptionCallShort = Security("OptionCallShort", [Instruments.OPTION, Options.CALL, Positions.SHORT])
VerticalPut = Strategy("VerticalPut", [Spreads.VERTICAL, Options.PUT, Positions.EMPTY], options=[OptionPutLong, OptionPutShort], stocks=[])
VerticalCall = Strategy("VerticalCall", [Spreads.VERTICAL, Options.CALL, Positions.EMPTY], options=[OptionCallLong, OptionCallShort], stocks=[])
CollarLong = Strategy("CollarLong", [Spreads.COLLAR, Options.EMPTY, Positions.LONG], options=[OptionPutLong, OptionCallShort], stocks=[StockLong])
CollarShort = Strategy("CollarShort", [Spreads.COLLAR, Options.EMPTY, Positions.SHORT], options=[OptionCallLong, OptionPutShort], stocks=[StockShort])


class Securities(Category):
    class Stocks(Category): Long = StockLong; Short = StockShort
    class Options(Category):
        class Puts(Category): Long = OptionPutLong; Short = OptionPutShort
        class Calls(Category): Long = OptionCallLong; Short = OptionCallShort

class Strategies(Category):
    class Verticals(Category): Put = VerticalPut; Call = VerticalCall
    class Collars(Category): Long = CollarLong; Short = CollarShort


class Categories:
    Securities = Securities
    Strategies = Strategies

class Querys:
    Product = Product
    Contract = Contract
    Symbol = Symbol
    History = History

class Variables:
    Scenarios = Scenarios
    Security = Security
    Strategy = Strategy
    Markets = Markets
    Instruments = Instruments
    Options = Options
    Positions = Positions
    Pricing = Pricing
    Actions = Actions
    Terms = Terms
    Spreads = Spreads
    Valuations = Valuations
    Technicals = Technicals
    Status = Status
    Omega = Omega
    Theta = Theta
    Phi = Phi



