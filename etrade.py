# -*- coding: utf-8 -*-
"""
Created on Sun May 15 2022
@name:   ETrade Finance  Application
@author: Jack Kirby Cook

"""

import sys
import warnings
import logging

from utilities.inputs import InputParser
from webscraping.webtimers import WebDelayer
from webscraping.webreaders import WebReader, Retrys
from webscraping.weburl import WebURL
from webscraping.webpages import WebRequestPage

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = []
__copyright__ = "Copyright 2022, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


_aslist = lambda x: list(x) if isinstance(x, (tuple, list, set)) else [x]
_filter = lambda x: [i for i in x if i is not None]


class ETrade_WebDelayer(WebDelayer): pass
class ETrade_WebReader(WebReader, retrys=Retrys(retries=3, backoff=0.3, httpcodes=(500, 502, 504)), authenticate=None): pass


class ETrade_Request_WebURL(WebURL, protocol="https", domain="api.etrade.com"):
    @staticmethod
    def path(*args, **kwargs): return ["oauth", "request_token"]
    @staticmethod
    def parm(*args, **kwargs): pass


class ETrade_Authorize_WebURL(WebURL, protocol="https", domain="api.etrade.com"):
    @staticmethod
    def path(*args, **kwargs): return ["e", "t", "etws", "authorize"]
    @staticmethod
    def parm(*args, **kwargs): pass


class ETrade_Access_WebURL(WebURL, protocol="https", domain="api.etrade.com"):
    @staticmethod
    def path(*args, renew=False, **kwargs): return ["oauth", "access_token" if not renew else "renew_access_token"]
    @staticmethod
    def parm(*args, **kwargs): pass


class ETrade_Ticker_WebURL(WebURL, protocol="https", domain="api.etrade.com"):
    @staticmethod
    def path(*args, tickers=[], ticker=None, **kwargs):
        return ["v1", "market", "quote", ",".join(_aslist(tickers) + _aslist(ticker))]


class ETrade_Options_WebURL(WebURL, protocol="https", domain="api.etrade.com"):
    @staticmethod
    def path(*args, **kwargs):
        return ["v1", "market", "optionchains"]

    @staticmethod
    def parm(*args, ticker, year, month, strike, size, **kwargs):
        return {"symbol": ticker, "expiryYear": str(year), "expiryMonth": str(month).zfill(2), "strikePriceNear": str(strike), "noOfStrikes": str(size)}


class USCensus_WebPage(WebRequestPage):
    def execute(self, *args, **kwargs):
        pass


def main(*args, **kwargs):
    delayer = ETrade_WebDelayer(name="ETradeDelayer", method="constant", wait=10)
    reader = ETrade_WebReader(name="ETradeReader")


if __name__ == "__main__":
    sys.argv += []
    logging.basicConfig(level="INFO", format="[%(levelname)s, %(threadName)s]:  %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    parsers = {}
    inputparser = InputParser(proxys={"assign": "=", "space": "_"}, parsers=parsers, default=str)
    inputparser(*sys.argv[1:])
    main(*inputparser.arguments, **inputparser.parameters)



