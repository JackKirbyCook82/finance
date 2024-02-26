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


INDEX = {"instrument": str, "position": str, "ticker": str, "expire": np.datetime64, "strike": np.float32, "date": np.datetime64}
COLUMNS = {"price": np.float32, "underlying": np.float32, "size": np.int32, "volume": np.int64, "interest": np.int32, "quantity": np.int32}


class SecurityFilter(Filter, index=INDEX, columns=COLUMNS):
    def execute(self, query, *args, **kwargs):
        securities = query.securities
        prior = len(securities.index)
        mask = self.mask(securities, *args, **kwargs)
        securities = self.filter(securities, *args, mask=mask, **kwargs)
        post = len(securities.index)
        query = query(securities=securities)
        __logger__.info(f"Filter: {repr(self)}|{str(query)}[{prior:.0f}|{post:.0f}]")
        yield query


class SecurityParser(Parser, index=INDEX, columns=COLUMNS):
    def execute(self, query, *args, **kwargs):
        securities = query.securities
        securities = self.parse(securities, *args, **kwargs)
        query = query(securities=securities)
        yield query


class SecurityFile(DataframeFile, index=INDEX, columns=COLUMNS): pass
class SecuritySaver(Saver):
    def execute(self, query, *args, **kwargs):
        securities = query.securities
        assert isinstance(securities, pd.DataFrame)
        if bool(securities.empty):
            return
        inquiry_name = str(query.inquiry.strftime("%Y%m%d_%H%M%S"))
        inquiry_folder = self.path(inquiry_name)
        if not os.path.isdir(inquiry_folder):
            os.mkdir(inquiry_folder)
        contract_name = "_".join([str(query.contract.ticker), str(query.contract.expire.strftime("%Y%m%d"))])
        contract_folder = self.path(inquiry_name, contract_name)
        if not os.path.isdir(contract_folder):
            os.mkdir(contract_folder)
        security_file = self.path(inquiry_name, contract_name, "security.csv")
        securities = self.parse(securities, *args, **kwargs)
        self.write(securities, file=security_file, filemode="w")
        __logger__.info("Saved: {}[{}]".format(repr(self), str(security_file)))


class SecurityLoader(Loader):
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
                security_file = self.path(inquiry_name, contract_name, "security.csv")
                securities = self.read(file=security_file)
                securities = self.parse(securities, *args, **kwargs)
                yield Query(inquiry, contract, securities=securities)



