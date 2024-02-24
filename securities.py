# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime as Datetime

from support.files import DataframeFile
from support.processes import Saver, Loader, Filter, Parser

from finance.variables import Query, Contract

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityFilter", "SecurityParser", "SecurityLoader", "SecuritySaver", "SecurityFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


COLUMNS_VARS = {"price": np.float32, "underlying": np.float32, "size": np.int32, "volume": np.int64, "interest": np.int32, "quantity": np.int32}
INDEX_VARS = {"instrument": str, "position": str, "strike": np.float32}
SCOPE_VARS = {"ticker": str, "expire": np.datetime64, "date": np.datetime64}


class SecurityFilter(Filter):
    def execute(self, query, *args, **kwargs):
        securities = query.securities
        prior = len(securities.index)
        mask = self.mask(securities, *args, **kwargs)
        securities = self.filter(securities, *args, mask=mask, **kwargs)
        post = len(securities.index)
        query = query(securities=securities)
        __logger__.info(f"Filter: {repr(self)}|{str(query)}[{prior:.0f}|{post:.0f}]")
        yield query


class SecurityParser(Parser, index=list(INDEX_VARS.keys()), scope=list(SCOPE_VARS.keys()), columns=list(COLUMNS_VARS.keys())):
    def execute(self, query, *args, **kwargs):
        securities = query.securities
        securities = self.parse(securities, *args, **kwargs)
        query = query(securities=securities)
        yield query


class SecurityFile(DataframeFile, header=INDEX_VARS | SCOPE_VARS | COLUMNS_VARS): pass
class SecuritySaver(Saver):
    def execute(self, query, *args, **kwargs):
        securities = query.securities
        assert isinstance(securities, pd.DataFrame)
        if bool(securities.empty):
            return
        inquiry_name = str(query.inquiry.strftime("%Y%m%d_%H%M%S"))
        inquiry_folder = self.file.path(inquiry_name)
        if not os.path.isdir(inquiry_folder):
            os.mkdir(inquiry_folder)
        contract_name = "_".join([str(query.contract.ticker), str(query.contract.expire.strftime("%Y%m%d"))])
        contract_file = self.file.path(inquiry_name, contract_name + ".csv")
        self.file.write(securities, file=contract_file, filemode="w")
        __logger__.info("Saved: {}[{}]".format(repr(self), str(contract_file)))


class SecurityLoader(Loader):
    def execute(self, *args, tickers=None, expires=None, **kwargs):
        function = lambda foldername: Datetime.strptime(foldername, "%Y%m%d_%H%M%S")
        inquiry_folders = self.file.directory()
        for inquiry_name in sorted(inquiry_folders, key=function, reverse=False):
            inquiry = function(inquiry_name)
            contract_filenames = self.file.directory(inquiry_name)
            for contract_filename in contract_filenames:
                contract_name = os.path.splitext(contract_filename)[0]
                ticker, expire = str(contract_name).split("_")
                ticker = str(ticker).upper()
                if tickers is not None and ticker not in tickers:
                    continue
                expire = Datetime.strptime(expire, "%Y%m%d").date()
                if expires is not None and expire not in expires:
                    continue
                contract = Contract(ticker, expire)
                contract_file = self.file.path(inquiry_name, contract_filename)
                securities = self.file.read(file=contract_file)
                yield Query(inquiry, contract, securities=securities)



