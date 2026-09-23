# -*- coding: utf-8 -*-
"""
Created on Weds May 27 2026
@name:   Finance Enumerations Objects
@author: Jack Kirby Cook
@file:   finance/enumerations.py

"""

from enum import Enum

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Technical", "Spread", "Instrument", "Status", "Website", "Option", "Position", "Terms", "Tenure", "Intent", "Action", "Movement"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class Enumeration(Enum):
    def __str__(self): return str(self.name).lower()
    def __int__(self): return int(self.value)

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            normalized = (value.strip().upper().replace(" ", "").replace("_", ""))
            if normalized.lstrip("-").isdigit(): return cls(int(normalized))
            for member in cls:
                member_name = (member.name.upper().replace(" ", "").replace("_", ""))
                if member_name == normalized: return member
        return None

class Status(Enumeration): ACCEPTED, REJECTED, EXECUTING, PARTIAL, FILLED, CANCELED, EXPIRED = range(7)
class Technical(Enumeration): BARS, STATS, SMA, EMA, MACD, RSI, BB, ATR, MFI, CMF, OBV = range(11)
class Instrument(Enumeration): EMPTY, STOCK, OPTION, SPREAD, CONTRACT = range(5)
class Spread(Enumeration): EMPTY, FLY, CALENDAR, VERTICAL, COLLAR = range(5)
class Website(Enumeration): ETRADE, ALPACA, INTERACTIVE = range(3)
class Terms(Enumeration): MARKET, LIMIT, STOP = range(3)
class Tenure(Enumeration): DAY, GTC, FOK = range(3)
class Movement(Enumeration): LOSS, STAGNANT, GAIN = (-1, 0, +1)
class Position(Enumeration): SHORT, LONG = (-1, +1)
class Intent(Enumeration): CLOSE, OPEN = (-1, +1)
class Option(Enumeration): PUT, CALL = (-1, +1)
class Action(Enumeration): SELL, BUY = (-1, +1)



