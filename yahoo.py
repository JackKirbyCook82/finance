# -*- coding: utf-8 -*-
"""
Created on Thurs Apr 21 2022
@name:   Yahoo Finance Download History Application
@author: Jack Kirby Cook

"""

import sys
import os.path
import warnings
import logging
import traceback
import pandas as pd
from abc import ABC
import regex as re
from datetime import date as Date
from datetime import datetime as Datetime
from datetime import timedelta as Timedelta
from collections import namedtuple as ntuple

MAIN_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(MAIN_DIR, os.pardir))
RESOURCE_DIR = os.path.join(ROOT_DIR, "resources")
SAVE_DIR = os.path.join(ROOT_DIR, "save")
REPOSITORY_DIR = os.path.join(SAVE_DIR, "yahoo")
REPORT_FILE = os.path.join(REPOSITORY_DIR, "history.csv")
DRIVER_EXE = os.path.join(RESOURCE_DIR, "chromedriver.exe")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utilities.inputs import InputParser
from utilities.dispatchers import keywordDispatcher as keydispatcher
from utilities.dispatchers import typeDispatcher as typedispatcher
from utilities.parsers import parmParser as parmparser
from webscraping.webtimers import WebDelayer
from webscraping.webloaders import WebLoader
from webscraping.webdrivers import WebBrowser
from webscraping.webquerys import WebQuery, WebDataset
from webscraping.webqueues import WebScheduler, WebQueueable, WebQueue
from webscraping.weburl import WebURL
from webscraping.webdata import WebTable
from webscraping.webpages import ContentMixin, WebBrowserPage, DataframeMixin, GeneratorMixin, WebData
from webscraping.webdownloaders import WebDownloader, CacheMixin

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Yahoo_WebDelayer", "Yahoo_WebBrowser", "Yahoo_WebDownloader", "Yahoo_WebScheduler"]
__copyright__ = "Copyright 2022, Jack Kirby Cook"
__license__ = ""


LOGGER = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


QUERYS = ["ticker", "date"]
DATASETS = ["history.csv", "options.csv", "dividend.csv", "split.csv"]
INTERVALS = {"day": "1d", "week": "1wk", "month": "1m"}
DURATIONS = {"day": 1, "week": 7, "month": 365/12, "year": 365}
FORMATS = ["%Y-%m-%d %I:%M%p %Z", "%Y/%m/%d %H:%m:%S", "%Y/%m/%d", "%b %d, %Y", "%m/%d/%Y"]
FILTERS = {"history": "history", "dividend": "div", "split": "split"}


history_xpath = r"//table[@data-test='historical-prices']"
option_xpath = r"//table[contains(@class, 'list-options')]"
history_webloader = WebLoader(xpath=history_xpath)
option_webloader = WebLoader(xpath=option_xpath)
filter_parser = lambda x, j: [i for i in x if i is not j]
common_parser = lambda x, y: [i for i in x if i in y]
price_parser = lambda x: float(x)
volume_parser = lambda x: int(str(x).replace(",", ""))
ticker_parser = lambda x: str(x).upper()
tickers_parsers = lambda x: [ticker_parser(i) for i in x.split(",")]
Duration = ntuple("Duration", "key value")


def strptime(txt, fmts):
    for fmt in fmts:
        try:
            return Datetime.strptime(txt, fmt)
        except ValueError:
            pass
    raise ValueError(txt)


@typedispatcher
def date_parser(x): raise TypeError(type(x).__name__)
@date_parser.register(Datetime)
def date_parser_datetime(x): return x
@date_parser.register(Date)
def date_parser_date(x): return Datetime(x.year, x.month, x.day)
@date_parser.regsiter(str)
def date_parser_str(x): return strptime(x, fmts=FORMATS)


@typedispatcher
def duration_parser(x): raise TypeError(type(x).__name__)
@typedispatcher.register(Timedelta)
def duration_parser(x): return x
@typedispatcher.register(Duration)
def duration_parser(x): return Timedelta(**{x.key: int(DURATIONS[x.key]) * int(x.value)})
@typedispatcher.register(str)
def duration_parser(x): return Timedelta(**{str(re.findall("[a-rt-z]+", x)[0]): int(re.findall("^-?\d+", x)[0])})


def table_parser(dataframe, *args, ticker, **kwargs):
    rename = {"Contract Name": "Contract", "Last Trade Date": "Date", "Last Price": "Price", "Open Interest": "Interest", "Dividends": "Dividend", "Adj Close": "Adjusted"}
    prices = ["price", "open", "high", "low", "close", "adjusted", "strike", "bid", "ask"]
    volumes = ["volume", "interest"]
    dataframe.columns = [str(column).replace("*", "") for column in dataframe.columns]
    dataframe.rename(columns=rename, inplace=True)
    dataframe.columns = [column.lower() for column in dataframe.columns]
    dataframe["ticker"] = ticker_parser(ticker)
    for column in common_parser(dataframe.columns, prices):
        dataframe[column] = dataframe[column].apply(price_parser)
    for column in common_parser(dataframe.columns, volumes):
        dataframe[column] = dataframe[column].apply(volume_parser)
    return dataframe


class Yahoo_History(WebTable, loader=history_webloader, parsers={}, optional=False): pass
class Yahoo_Dividend(WebTable, loader=history_webloader, parsers={}, optional=False): pass
class Yahoo_Split(WebTable, loader=history_webloader, parsers={}, optional=False): pass
class Yahoo_Option(WebTable, loader=option_webloader, parsers={}, optoinal=False): pass
class Yahoo_WebDelayer(WebDelayer): pass
class Yahoo_WebBrowser(WebBrowser, files={"chrome": DRIVER_EXE}, options={"headless": False, "images": True, "incognito": False}): pass
class Yahoo_WebQueue(WebQueue): pass
class Yahoo_WebQuery(WebQuery, WebQueueable, fields=QUERYS): pass
class Yahoo_WebDatasets(WebDataset[pd.DataFrame], ABC, fields=DATASETS): pass


class Yahoo_WebScheduler(WebScheduler, fields=QUERYS):
    @staticmethod
    def ticker(*args, ticker=None, tickers=[], **kwargs): return list(set([ticker_parser(x) for x in filter_parser([ticker, *tickers], None)]))
    @staticmethod
    def date(*args, year=None, years=[], **kwargs): return list(set([int(x) for x in filter_parser([year, *years], None)]))

    @staticmethod
    def execute(querys, *args, **kwargs):
        queueables = [Yahoo_WebQuery(query, name="YahooQuery") for query in querys]
        queue = Yahoo_WebQueue(queueables, *args, name="YahooQueue", **kwargs)
        return queue


class Yahoo_WebURL(WebURL, protocol="https", domain="www.finance.yahoo.com"):
    @staticmethod
    def path(*args, ticker, dataset, **kwargs): return ["quote", ticker_parser(ticker), dataset]
    @staticmethod
    @keydispatcher("dataset")
    def parm(*args, dataset, **kwargs): raise KeyError(dataset)

    @staticmethod
    @parm.register("history", "dividend", "split")
    @parmparser(date=date_parser, duration=duration_parser)
    def history(*args, dataset, date, interval, duration, **kwargs):
        try:
            interval = INTERVALS[interval]
        except KeyError:
            raise ValueError(interval)
        start, end = date.timestamp(), (date + duration.timedelta()).timestamp()
        start, end = min(start, end), max(start, end)
        return {"period1": start, "period2": end, "interval": interval, "filter": FILTERS[dataset], "includeAdjustedClose": "true"}

    @staticmethod
    @parm.regsiter("options")
    @parmparser(date=date_parser)
    def options(*args, ticker, date, **kwargs):
        return {"date": date.timestamp(), "p": ticker_parser(ticker), "includeAdjustedClose": "true"}


class Yahoo_WebData(WebData):
    HISTORY = Yahoo_History
    DIVIDEND = Yahoo_Dividend
    SPLIT = Yahoo_Split
    OPTION = Yahoo_Option


class Yahoo_WebPage(ContentMixin, DataframeMixin, GeneratorMixin, WebBrowserPage, contents=Yahoo_WebData):
    def execute(self, *args, ticker, date, **kwargs):
        query = {"ticker": ticker, "date": date}
#        data = self[Yahoo_WebData.TABLE].table(*args, ticker=ticker, **kwargs)
#        yield query, "", data


class Yahoo_WebDownloader(CacheMixin, WebDownloader):
    def execute(self, *args, scheduler, browser, delayer, **kwargs):
        with browser() as driver:
            page = Yahoo_WebPage(driver, name="YahooPage", delayer=delayer)
            with scheduler(*args, **kwargs) as queue:
                with queue:
                    for query in queue:
                        url = Yahoo_WebURL(**query.todict())
                        page.load(str(url), referer=None)
                        page.setup()
                        for fields, dataset, data in page(**query.todict()):
                            yield query, Yahoo_WebDatasets({dataset: data}, name="YahooDataset")
                        query.success()


def main(*args, **kwargs):
    delayer = Yahoo_WebDelayer(name="YahooDelayer", method="constant", wait=10)
    browser = Yahoo_WebBrowser(name="YahooBrowser", browser="chrome", timeout=60)
    scheduler = Yahoo_WebScheduler(name="YahooScheduler", randomize=False, size=None, file=REPORT_FILE)
    downloader = Yahoo_WebDownloader(name="YahooDownloader", repository=REPOSITORY_DIR, timeout=60)
    downloader(*args, scheduler=scheduler, browser=browser, delayer=delayer, **kwargs)
    downloader.start()
    downloader.join()
    for query, results in downloader.results.items():
        LOGGER.info(str(query))
        LOGGER.info(str(results))
    if bool(downloader.error):
        traceback.print_exception(*downloader.error)


if __name__ == "__main__":
    sys.argv += ["tickers=TSLA,AAPL,SPY,QQQ", "date=20222/07/01", "interval=day", "duration=-10years"]
    logging.basicConfig(level="INFO", format="[%(levelname)s, %(threadName)s]:  %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    logging.getLogger("seleniumwire").setLevel(logging.ERROR)
    parsers = {"ticker": ticker_parser, "tickers": tickers_parsers, "date": date_parser, "duration": duration_parser}
    inputparser = InputParser(proxys={"assign": "=", "space": "_"}, parsers=parsers, default=str)
    inputparser(*sys.argv[1:])
    main(*inputparser.arguments, **inputparser.parameters)









