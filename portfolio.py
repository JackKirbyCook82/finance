# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Valuation Objects
@author: Jack Kirby Cook

"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime as Datetime

from support.processes import Calculator
from support.files import DataframeFile, Header
from support.processes import Saver, Loader, Filter, Axes

from finance.variables import Query, Contract

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["PortfolioCalculator", "PortfolioFilter", "PortfolioLoader", "PortfolioSaver", "PortfolioFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


INDEX = {"strike": np.float32}
COLUMNS = {"instrument": str, "position": str}
SCOPE = {"ticker": str, "expire": np.datetime64, "date": np.datetime64}
VALUES = {"quantity": np.int32}
HEADER = Header(INDEX | COLUMNS, VALUES | SCOPE)
AXES = Axes(INDEX, COLUMNS, VALUES, SCOPE)


class PortfolioCalculator(Calculator):
    def execute(self, query, *args, **kwargs):
        pass


class PortfolioFilter(Filter):
    def execute(self, query, *args, **kwargs):
        holdings = query.holdings
        prior = self.size(holdings["quantity"])
        mask = self.mask(holdings, *args, **kwargs)
        holdings = self.filter(holdings, *args, mask=mask, **kwargs)
        post = self.size(holdings["quantity"])
        query = query(holdings=holdings)
        __logger__.info(f"Filter: {repr(self)}|{str(query)}[{prior:.0f}|{post:.0f}]")
        yield query


class PortfolioFile(DataframeFile, header=HEADER): pass
class PortfolioSaver(Saver):
    def execute(self, query, *args, **kwargs):
        holdings = query.holdings
        assert isinstance(holdings, pd.DataFrame)
        if bool(holdings.empty):
            return
        inquiry_name = str(query.inquiry.strftime("%Y%m%d_%H%M%S"))
        inquiry_folder = self.path(inquiry_name)
        if not os.path.isdir(inquiry_folder):
            os.mkdir(inquiry_folder)
        contract_name = "_".join([str(query.contract.ticker), str(query.contract.expire.strftime("%Y%m%d"))])
        contract_folder = self.path(inquiry_name, contract_name)
        if not os.path.isdir(contract_folder):
            os.mkdir(contract_folder)
        holding_file = self.path(inquiry_name, contract_name, "holding.csv")
        holdings = self.parse(holdings, *args, **kwargs)
        self.write(holdings, file=holding_file, filemode="w")
        __logger__.info("Saved: {}[{}]".format(repr(self), str(holding_file)))


class PortfolioLoader(Loader):
    def execute(self, *args, **kwargs):
        function = lambda foldername: Datetime.strptime(foldername, "%Y%m%d_%H%M%S")
        inquiry_names = self.directory()
        for inquiry_name in sorted(inquiry_names, key=function, reverse=False):
            inquiry = function(inquiry_name)
            contract_names = self.directory(inquiry_name)
            for contract_name in contract_names:
                contract_name = os.path.splitext(contract_name)[0]
                ticker, expire = str(contract_name).split("_")
                ticker = str(ticker).upper()
                expire = Datetime.strptime(expire, "%Y%m%d").date()
                contract = Contract(ticker, expire)
                holding_file = self.path(inquiry_name, contract_name, "holding.csv")
                holdings = self.read(file=holding_file)
                holdings = self.parse(holdings, *args, **kwargs)
                yield Query(inquiry, contract, holdings=holdings)



