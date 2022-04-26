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
REPORT_FILE = os.path.join(REPOSITORY_DIR, "historical.csv")
DRIVER_EXE = os.path.join(RESOURCE_DIR, "chromedriver.exe")
NORDVPN_EXE = os.path.join("C:/", "Program Files", "NordVPN", "NordVPN.exe")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utilities.iostream import InputParser
from webscraping.webtimers import WebDelayer
from webscraping.webloaders import WebLoader
from webscraping.webdrivers import WebBrowser
from webscraping.webquerys import WebQuery, WebDataset
from webscraping.webqueues import WebScheduler, WebQueueable, WebQueue
from webscraping.weburl import WebURL
from webscraping.webdata import WebTable
from webscraping.webpages import WebRequestPage, DataframeMixin
from webscraping.webdownloaders import WebDownloader

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Yahoo_Finance_WebDelayer", "Yahoo_Finance_WebBrowser", "Yahoo_Finance_WebDownloader", "Yahoo_Finance_WebScheduler"]
__copyright__ = "Copyright 2022, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


historical_xpath = r"//table[@data-test='historical-prices']"
historical_webloader = WebLoader(xpath=historical_xpath)
datetime_parser = lambda x, y, z: Datetime(x, y, z)
filter_parser = lambda x, j: [i for i in x if i is not j]
date_parser = lambda x: Datetime.strptime(x, "%b %d, %Y")
volume_parser = lambda x: int(str(x).replace(",", ""))
ticker_parser = lambda x: str(x).upper()


def table_parser(dataframe, *args, ticker, **kwargs):
    dataframe.drop(index=dataframe.index[-1], axis=0, inplace=True)
    dataframe.columns = [str(column).replace("*", "") for column in dataframe.columns]
    dataframe.rename(columns={"Adj Close": "Adjusted"}, inplace=True)
    dataframe.columns = [column.lower() for column in dataframe.columns]
    dataframe["ticker"] = ticker_parser(ticker)
    dataframe["date"] = dataframe["date"].apply(date_parser)
    return dataframe


def price_parser(dataframe, *args, **kwargs):
    dataframe["volume"] = dataframe["volume"].apply(volume_parser)
    for column in ["open", "close", "high", "low", "adjusted"]:
        dataframe[column] = dataframe[column].apply(float)
    dataframe = dataframe[["date", "ticker", "open", "close", "high", "low", "adjusted"]]
    return dataframe


def dividend_parser(dataframe, *args, **kwargs):
    dataframe["dividend"] = dataframe["adjusted"].apply(lambda x: float(str(x).split(" ")[0]))
    dataframe = dataframe[["date", "ticker", "dividend"]]
    return dataframe


def split_parser(dataframe, *args, **kwargs):
    dataframe["split"] = dataframe["adjusted"].apply(lambda x: str(x).split(" ")[0])
    dataframe = dataframe[["date", "ticker", "split"]]
    return dataframe


def data_parser(dataframe, *args, **kwargs):
    dividend = bool(str(dataframe["adjusted"]).find("Dividend"))
    split = bool(str(dataframe["adjusted"]).find("Stock Split"))
    bar = not dividend and not split
    return {"price": price_parser(dataframe[bar], *args, **kwargs), "dividend": dividend_parser(dataframe[dividend], *args, **kwargs), "split": split_parser(dataframe, *args, **kwargs)}


class Yahoo_Finance_Historical(WebTable, loader=historical_webloader, parsers={"table": table_parser, "data": data_parser}, optional=False): pass
class Yahoo_Finance_WebDelayer(WebDelayer): pass
class Yahoo_Finance_WebBrowser(WebBrowser, files={"chrome": DRIVER_EXE}, options={"headless": False, "images": True, "incognito": False}): pass
class Yahoo_Finance_WebQueue(WebQueue): pass
class Yahoo_Finance_WebQuery(WebQuery, WebQueueable, fields=["ticker", "date"]): pass
class Yahoo_Finance_WebDatasets(WebDataset[pd.DataFrame], ABC, fields=["historical.csv"]): pass


class Yahoo_Finance_WebScheduler(WebScheduler, fields=["ticker", "date"]):
    @staticmethod
    def ticker(*args, ticker=None, tickers=[], **kwargs): return list(set([ticker_parser(x) for x in filter_parser([ticker, *tickers], None)]))
    @staticmethod
    def date(*args, year=None, years=[], **kwargs): return list(set([int(x) for x in filter_parser([year, *years], None)]))

    @staticmethod
    def execute(querys, *args, **kwargs):
        queueables = [Yahoo_Finance_WebQuery(query, name="YahooQuery") for query in querys]
        queue = Yahoo_Finance_WebQueue(queueables, *args, name="YahooQueue", **kwargs)
        return queue


class Yahoo_Finance_WebURL(WebURL, protocol="https", domain="finance.yahoo.com"):
    @staticmethod
    def path(*args, ticker, **kwargs): return ["quote", ticker_parser(ticker), "history"]

    @staticmethod
    def parm(*args, date, **kwargs):
        parm = {"interval": "1d", "filter": "history", "frequency": "1d", "includeAdjustedClose": "true"}
        start = datetime_parser(int(date), 1, 1)
        end = datetime_parser(int(date), 12, 31)
        return {"period1": str(int(start.timestamp())), "period2": str(int(end.timestamp())), **parm}


class Yahoo_Finance_WebPage(DataframeMixin, WebRequestPage):
    def execute(self, *args, ticker, date, **kwargs):
        query = {"ticker": ticker, "date": date}

#        dataframe = Yahoo_Finance_Historical(self.source).data(*args, ticker=ticker, **kwargs)
#        return query, "historical", dataframe


class Yahoo_Finance_WebDownloader(WebDownloader):
    def execute(self, *args, scheduler, browser, delayer, **kwargs):
        with browser() as driver:
            page = Yahoo_Finance_WebPage(driver, name="YahooPage", delayer=delayer)
            with scheduler(*args, **kwargs) as queue:
                with queue:
                    for query in queue:
                        url = Yahoo_Finance_WebURL(**query.todict())
                        page.load(str(url), referer=None)
                        page.setup()
                        fields, dataset, data = page(**query.todict())
                        yield query, Yahoo_Finance_WebDatasets({dataset: data}, name="YahooDataset")


def main(*args, **kwargs):
    delayer = Yahoo_Finance_WebDelayer(name="YahooDelayer", method="constant", wait=10)
    browser = Yahoo_Finance_WebBrowser(name="YahooBrowser", browser="chrome", timeout=60, wait=15)
    scheduler = Yahoo_Finance_WebScheduler(name="YahooScheduler", randomize=False, size=None, file=REPORT_FILE)
    downloader = Yahoo_Finance_WebDownloader(name="YahooDownloader", repository=REPOSITORY_DIR, timeout=60)
    downloader(*args, scheduler=scheduler, browser=browser, delayer=delayer, **kwargs)
    downloader.start()
    downloader.join()
    for query, results in downloader.results.items():
        LOGGER.info(str(query))
        LOGGER.info(str(results))
    if bool(downloader.error):
        traceback.print_exception(*downloader.error)


if __name__ == "__main__":
    sys.argv += ["tickers=APPL,TSLA,SPY,QQQ", "years=2021,2020"]
    logging.basicConfig(level="INFO", format="[%(levelname)s, %(threadName)s]:  %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    logging.getLogger("seleniumwire").setLevel(logging.ERROR)
    parsers = {"ticker": ticker_parser, "tickers": lambda x: [ticker_parser(i) for i in x.split(",")], "year": int, "years": lambda x: [int(i) for i in x.split(",")]}
    inputparser = InputParser(proxys={"assign": "=", "space": "_"}, parsers=parsers, default=str)
    inputparser(*sys.argv[1:])
    main(*inputparser.arguments, **inputparser.parameters)









