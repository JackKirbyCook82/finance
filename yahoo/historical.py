# -*- coding: utf-8 -*-
"""
Created on Thurs Apr 21 2022
@name:   Yahoo Finance Download Application
@author: Jack Kirby Cook

"""

import sys
import os.path
import warnings
import logging
import traceback
import pandas as pd
from abc import ABC
from datetime import datetime as Datetime

MAIN_DIR = os.path.dirname(os.path.realpath(__file__))
MODULE_DIR = os.path.abspath(os.path.join(MAIN_DIR, os.pardir))
ROOT_DIR = os.path.abspath(os.path.join(MODULE_DIR, os.pardir))
RESOURCE_DIR = os.path.join(ROOT_DIR, "resources")
SAVE_DIR = os.path.join(ROOT_DIR, "save")
REPOSITORY_DIR = os.path.join(SAVE_DIR, "yahoo")
NORDVPN_EXE = os.path.join("C:/", "Program Files", "NordVPN", "NordVPN.exe")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utilities.iostream import InputParser
from webscraping.webtimers import WebDelayer
from webscraping.webloaders import WebLoader
from webscraping.webreaders import WebReader, Retrys
from webscraping.webquerys import WebQuery, WebDataset
from webscraping.weburl import WebURL
from webscraping.webdata import WebTable
from webscraping.webpages import WebRequestPage, DataframeMixin
from webscraping.webdownloaders import WebDownloader

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Yahoo_Finance_WebDelayer", "Yahoo_Finance_WebReader", "Yahoo_Finance_WebDownloader"]
__copyright__ = "Copyright 2022, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


historical_xpath = r"//table[@data-test='historical-prices']"
historical_webloader = WebLoader(xpath=historical_xpath)
datetime_parser = lambda x, **kw: {int: Datetime.fromtimestamp(x), str: Datetime.strptime(x, kw.get("format", "%d/%m/%Y"))}[type(x)](x) if not isinstance(x, Datetime) else x
filter_parser = lambda x, j: [i for i in x if i is not j]
ticker_parser = lambda x: str(x).upper()
date_parser = lambda x: Datetime.strptime(x, "%b %d, %Y")
price_parser = lambda x: float(x)
volume_parser = lambda x: int(str(x).replace(",", ""))


def historical_parser(dataframe, *args, ticker, **kwargs):
    columns = {"date": date_parser, "volume": volume_parser, "ticker": ticker_parser}
    dataframe.rename(columns={"Adj Close**": "Adjusted"}, inplace=True)
    dataframe.columns = [column.lower() for column in dataframe.columns]
    dataframe["ticker"] = ticker
    for column in dataframe.columns:
        dataframe[column] = dataframe[column].apply(columns.get(column, price_parser))
    return dataframe


class Yahoo_Finance_Historical(WebTable, loader=historical_webloader, parsers={"table": historical_parser}, optional=False): pass
class Yahoo_Finance_WebDelayer(WebDelayer): pass
class Yahoo_Finance_WebReader(WebReader, retrys=Retrys(retries=3, backoff=0.3, httpcodes=(500, 502, 504)), authenticate=None): pass
class Yahoo_Finance_WebQuery(WebQuery, fields=["ticker"]): pass
class Yahoo_Finance_WebDatasets(WebDataset[pd.DataFrame], ABC, fields=["historical"]): pass


class Yahoo_Finance_WebURL(WebURL, protocol="https", domain="www.finance.yahoo.com"):
    @staticmethod
    def path(*args, ticker, **kwargs): return ["quote", ticker_parser(ticker), "history"]

    @staticmethod
    def parm(*args, fromdate, todate, **kwargs):
        parm = {"interval": "1d", "filter": "history", "frequency": "1d", "includeAdjustedClose": "true"}
        return {"period1": datetime_parser(fromdate, **kwargs), "period2": datetime_parser(todate, **kwargs), **parm}


class Yahoo_Finance_WebPage(DataframeMixin, WebRequestPage):
    def execute(self, *args, ticker, **kwargs):
        dataframe = Yahoo_Finance_Historical(self.source).data(*args, ticker, **kwargs)
        return {"ticker": ticker}, "historical", dataframe


class Yahoo_Finance_WebDownloader(WebDownloader):
    def execute(self, *args, reader, delayer, **kwargs):
        tickers = filter_parser([kwargs.pop("ticker", None), *kwargs.pop("tickers", [])], None)
        with reader() as session:
            page = Yahoo_Finance_WebPage(session, name="YahooPage", delayer=delayer)
            for ticker in tickers:
                url = Yahoo_Finance_WebURL(*args, ticker=ticker, **kwargs)
                page.load(str(url), referer=None)
                page.setup(*args, **kwargs)
                query, dataset, data = page(*args, ticker=ticker, **kwargs)
                query = Yahoo_Finance_WebQuery(query, name="YahooQuery")
                dataset = Yahoo_Finance_WebDatasets({dataset: data}, name="YahooDataset")
                yield query, dataset


def main(*args, **kwargs):
    delayer = Yahoo_Finance_WebDelayer(name="YahooDelayer", method="constant", wait=10)
    reader = Yahoo_Finance_WebReader(name="YahooReader")
    downloader = Yahoo_Finance_WebDownloader(name="YahooDownloader", timeout=30)
    downloader(*args, reader=reader, delayer=delayer, **kwargs)
    downloader.start()
    downloader.join()
    for query, results in downloader.results.items():
        LOGGER.info(str(query))
        LOGGER.info(str(results))
    if bool(downloader.error):
        traceback.print_exception(*downloader.error)


if __name__ == "__main__":
    sys.argv += ["fromdate=01/01/2021", "todate=31/12/2021", "ticker=", "tickers=", "format=%d/%m/%Y"]
    logging.basicConfig(level="INFO", format="[%(levelname)s, %(threadName)s]:  %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    parsers = {"from": date_parser, "to": date_parser, "ticker": ticker_parser, "tickers": [ticker_parser(i) for i in lambda x: x.split(",")]}
    inputparser = InputParser(proxys={"assign": "=", "space": "_"}, parsers=parsers, default=str)
    inputparser(*sys.argv[1:])
    main(*inputparser.arguments, **inputparser.parameters)









