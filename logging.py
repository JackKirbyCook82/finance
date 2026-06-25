# -*- coding: utf-8 -*-
"""
Created on Weds May 27 2026
@name:   Finance Variable Objects
@author: Jack Kirby Cook

"""

from finance.variables import Enumerations
from support.decorators import Dispatchers
from support.custom import DateRange
from support.mixins import Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Logging"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class Logging(Logging):
    @Dispatchers.Value(locator="instrument")
    def results(self, dataframe, *args, title, instrument, **kwargs): raise ValueError(instrument)

    @results.register(Enumerations.Instrument.STOCK)
    def stock(self, dataframe, *args, title, **kwargs):
        tickers = "|".join(list(dataframe["ticker"].unique()))
        previous, post = kwargs.get("previous", None), kwargs.get("post", len(dataframe))
        sizes = f"{int(previous):.0f}|{int(post):.0f}, {post / previous * 100:.0f}%" if previous is not None else f"{len(dataframe):.0f}"
        self.console(str(title), f"Stocks[{str(tickers)}, {str(sizes)}]")

    @results.register(Enumerations.Instrument.OPTION)
    def option(self, dataframe, *args, title, **kwargs):
        tickers = "|".join(list(dataframe["ticker"].unique()))
        expires = DateRange.create(list(dataframe["expire"].unique()))
        expires = f"{expires.minimum.strftime('%Y%m%d')}->{expires.maximum.strftime('%Y%m%d')}"
        previous, post = kwargs.get("previous", None), kwargs.get("post", len(dataframe))
        sizes = f"{int(previous):.0f}|{int(post):.0f}, {post / previous * 100:.0f}%" if previous is not None else f"{len(dataframe):.0f}"
        self.console(str(title), f"Options[{str(tickers)}, {str(expires)}, {str(sizes)}]")

    @results.register(Enumerations.Instrument.SPREAD)
    def spread(self, collection, *args, title, **kwargs):
        if not isinstance(collection, list): collection = [collection]
        tickers = "|".join(list({content.ticker for content in collection}))
        previous, post = kwargs.get("previous", None), kwargs.get("post", len(collection))
        sizes = f"{int(previous):.0f}|{int(post):.0f}, {post / previous * 100:.0f}%" if previous is not None else f"{len(collection):.0f}"
        self.console(str(title), f"Spreads[{str(tickers)}, {str(sizes)}]")

    @results.register(Enumerations.Instrument.CONTRACT)
    def contracts(self, collection, *args, title, **kwargs):
        if not isinstance(collection, list): collection = [collection]
        tickers = "|".join(list({content.ticker for content in collection}))
        previous, post = kwargs.get("previous", None), kwargs.get("post", len(collection))
        sizes = f"{int(previous):.0f}|{int(post):.0f}, {post / previous * 100:.0f}%" if previous is not None else f"{len(collection):.0f}"
        self.console(str(title), f"Spreads[{str(tickers)}, {str(sizes)}]")



