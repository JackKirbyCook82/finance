# -*- coding: utf-8 -*-
"""
Created on Weds May 27 2026
@name:   Finance Enumerations Objects
@author: Jack Kirby Cook

"""

from enum import Enum

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Technical", "Spread", "Instrument", "Status", "Website", "Option", "Position", "Terms", "Tenure", "Intent", "Action"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class Enumeration(Enum):
    def __str__(self): return str(self.name).lower()
    def __int__(self): return int(self.value)

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            string = value.upper().replace(" ", "").replace("_", "")
            if str(string).lstrip("-").isdigit():
                return cls(int(string))
            for member in cls:
                if member.name.replace("_", "") == string:
                    return member
        return None


class Technical(Enumeration): BARS, STATS, SMA, EMA, MACD, RSI, BB, ATR, MFI, CMF, OBV = range(11)
class Spread(Enumeration): EMPTY, VERTICAL, COLLAR, FLY, CALENDAR, CONDOR = range(6)
class Instrument(Enumeration): EMPTY, STOCK, OPTION, SPREAD, CONTRACT = range(5)
class Status(Enumeration): NEW, PARTIAL, FILLED, CANCELED, EXPIRED = range(5)
class Movement(Enumeration): GAIN, STAGNANT, LOSS = range(-1, 2)
class Website(Enumeration): ETRADE, ALPACA, INTERACTIVE = range(3)
class Option(Enumeration): PUT, EMPTY, CALL = range(-1, 2)
class Position(Enumeration): SHORT, EMPTY, LONG = range(-1, 2)
class Terms(Enumeration): MARKET, LIMIT, STOP = range(3)
class Tenure(Enumeration): DAY, GTC, FOK = range(3)
class Intent(Enumeration): OPEN, CLOSE = range(2)
class Action(Enumeration): BUY, SELL = range(2)



