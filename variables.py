# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 2024
@name:   Finance Variable Objects
@author: Jack Kirby Cook

"""

import numbers
import datetime

from support.variables import VariablesMeta, Variables, Variable
from support.querys import Field, Query

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Variables", "Querys", "Scenarios"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


Ticker = Field("ticker", str)
Date = Field("date", datetime.date, format="%Y%m%d")
Expire = Field("expire", datetime.date, format="%Y%m%d")
Strike = Field("strike", numbers.Number, digits=2)

Symbol = Query("Symbol", [Ticker], delimiter="|")
History = Query("History", [Ticker, Date], delimiter="|")
Contract = Query("Contract", [Ticker, Expire], delimiter="|")
Product = Query("Product", [Ticker, Expire, Strike], delimiter="|")

Theta = Variable("Theta", ["PUT", "NEUTRAL", "CALL"], start=-1)
Phi = Variable("Phi", ["SHORT", "NEUTRAL", "LONG"], start=-1)
Omega = Variable("Omega", ["BEAR", "NEUTRAL", "BULL"], start=-1)
Status = Variable("Status", ["PROSPECT", "PENDING", "ABANDONED", "REJECTED", "ACCEPTED"])
Technicals = Variable("Technicals", ["BARS", "STATISTIC", "STOCHASTIC"])
Scenarios = Variable("Scenarios", ["MINIMUM", "MAXIMUM"])
Valuations = Variable("Valuations", ["ARBITRAGE"])
Pricing = Variable("Pricing", ["BLACKSCHOLES"])

Markets = Variable("Markets", ["EMPTY", "BEAR", "BULL"], start=0)
Instruments = Variable("Instruments", ["EMPTY", "STOCK", "OPTION"], start=0)
Options = Variable("Options", ["EMPTY", "PUT", "CALL"], start=0)
Positions = Variable("Positions", ["EMPTY", "LONG", "SHORT"], start=0)
Spreads = Variable("Spreads", ["STRANGLE", "COLLAR", "VERTICAL"], start=1)

Security = Variables("Security", ["instrument", "option", "position"])
Strategy = Variables("Strategy", ["spread", "option", "position"], {"stocks", "options"})

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


class Securities(contents=[StockLong, StockShort, OptionPutLong, OptionPutShort, OptionCallLong, OptionCallShort], metaclass=VariablesMeta):
    Options = [OptionPutLong, OptionCallLong, OptionPutShort, OptionCallShort]
    Puts = [OptionPutLong, OptionPutShort]
    Calls = [OptionCallLong, OptionCallShort]
    Stocks = [StockLong, StockShort]

    class Stock: Long = StockLong; Short = StockShort
    class Option:
        class Put: Long = OptionPutLong; Short = OptionPutShort
        class Call: Long = OptionCallLong; Short = OptionCallShort

class Strategies(contents=[VerticalPut, VerticalCall, CollarLong, CollarShort], metaclass=VariablesMeta):
    Verticals = [VerticalPut, VerticalCall]
    Collars = [CollarLong, CollarShort]

    class Vertical: Put = VerticalPut; Call = VerticalCall
    class Collar: Long = CollarLong; Short = CollarShort


class Querys:
    Product = Product
    Contract = Contract
    Symbol = Symbol
    History = History

class Variables:
    Scenarios = Scenarios
    Securities = Securities
    Strategies = Strategies
    Security = Security
    Strategy = Strategy
    Markets = Markets
    Pricing = Pricing
    Instruments = Instruments
    Options = Options
    Positions = Positions
    Spreads = Spreads
    Valuations = Valuations
    Technicals = Technicals
    Status = Status
    Omega = Omega
    Theta = Theta
    Phi = Phi



