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
import xarray as xr
from datetime import datetime as Datetime

from support.files import DataframeFile
from support.pipelines import Producer, Processor, Consumer

from finance.variables import Query, Contract, Securities

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityFilter", "SecurityParser", "SecurityLoader", "SecuritySaver", "SecurityFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class SecurityFilter(Processor, title="Filtered"):
    def execute(self, query, *args, **kwargs):
        length = lambda dataframes: sum([len(dataframe.index) for dataframe in dataframes.values()])
        securities = query.securities
        assert isinstance(securities, dict)
        assert all([isinstance(dataframe, pd.DataFrame) for dataframe in securities.values()])
        securities = {security: dataframe for security, dataframe in securities.items() if not dataframe.empty}
        prior = length(securities)
        securities = {security: self.filter(dataframe, *args, **kwargs) for security, dataframe in securities.items()}
        post = length(securities)
        query = query(securities=securities)
        __logger__.info(f"Filter: {repr(self)}|{str(query)}[{prior:.0f}|{post:.0f}]")
        yield query

    def filter(self, dataframe, *args, **kwargs):
        index = ["security", "date", "ticker", "expire", "strike"]
        mask = self.mask(dataframe, *args, **kwargs)
        dataframe = dataframe.drop_duplicates(subset=index, keep="last", inplace=False)
        dataframe = dataframe.where(mask)
        dataframe = dataframe.dropna(axis=0, how="all")
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe

    @staticmethod
    def mask(dataframe, *args, size=None, interest=None, volume=None, **kwargs):
        mask = (dataframe["size"].notna() & dataframe["interest"].notna() & dataframe["volume"].notna())
        mask = (mask & (dataframe["size"] >= size)) if size is not None else mask
        mask = (mask & (dataframe["interest"] >= interest)) if interest is not None else mask
        mask = (mask & (dataframe["volume"] >= volume)) if volume is not None else mask
        return mask


class SecurityParser(Processor, title="Parsed"):
    def execute(self, query, *args, **kwargs):
        securities = query.securities
        assert isinstance(securities, dict)
        assert all([isinstance(dataframe, pd.DataFrame) for dataframe in securities.values()])
        securities = {security: dataframe for security, dataframe in securities.items() if not dataframe.empty}
        securities = {security: self.parse(dataframe, *args, **kwargs) for security, dataframe in securities.items()}
        query = query(securities=securities)
        yield query

    @staticmethod
    def parse(dataframe, *args, **kwargs):
        index = ["security", "date", "ticker", "expire", "strike"]
        dataframe = dataframe.drop_duplicates(subset=index, keep="last", inplace=False)
        dataframe = dataframe.set_index(index, inplace=False, drop=True)
        dataset = xr.Dataset.from_dataframe(dataframe)
        return dataset


class SecuritySaver(Consumer, title="Saved"):
    def __init__(self, *args, file, **kwargs):
        assert isinstance(file, SecurityFile)
        super().__init__(*args, **kwargs)
        self.securities = list(Securities.Options)
        self.file = file

    def execute(self, query, *args, **kwargs):
        securities = query.securities
        assert isinstance(securities, dict)
        assert all([isinstance(dataframe, pd.DataFrame) for dataframe in securities.values()])
        securities = {security: dataframe for security, dataframe in securities.items() if not dataframe.empty and security in self.securities}
        inquiry_name = str(query.inquiry.strftime("%Y%m%d_%H%M%S"))
        contract_name = "_".join([str(query.contract.ticker), str(query.contract.expire.strftime("%Y%m%d"))])
        inquiry_folder = self.file.path(inquiry_name)
        contract_folder = self.file.path(inquiry_name, contract_name)
        if bool(securities):
            if not os.path.isdir(inquiry_folder):
                os.mkdir(inquiry_folder)
            if not os.path.isdir(contract_folder):
                os.mkdir(contract_folder)
        for security, dataframe in securities.items():
            security_name = str(security).lower().replace("|", "_") + ".csv"
            securities_file = self.file.path(inquiry_name, contract_name, security_name)
            self.file.write(dataframe, file=securities_file, filemode="w")
            __logger__.info("Saved: {}[{}]".format(repr(self), str(securities_file)))


class SecurityQuery(Query, fields=["securities"]): pass
class SecurityLoader(Producer, title="Loaded"):
    def __init__(self, *args, file, **kwargs):
        assert isinstance(file, SecurityFile)
        super().__init__(*args, **kwargs)
        self.securities = list(Securities.Options)
        self.file = file

    def execute(self, *args, tickers=None, expires=None, **kwargs):
        function = lambda foldername: Datetime.strptime(foldername, "%Y%m%d_%H%M%S")
        inquiry_folders = self.file.directory()
        for inquiry_name in sorted(inquiry_folders, key=function, reverse=False):
            inquiry = function(inquiry_name)
            contract_names = self.file.directory(inquiry_name)
            for contract_name in contract_names:
                ticker, expire = str(os.path.splitext(contract_name)[0]).split("_")
                ticker = str(ticker).upper()
                if tickers is not None and ticker not in tickers:
                    continue
                expire = Datetime.strptime(expire, "%Y%m%d").date()
                if expires is not None and expire not in expires:
                    continue
                contract = Contract(ticker, expire)
                security_names = {security: str(security).lower().replace("|", "_") + ".csv" for security in self.securities}
                security_files = {security: self.file.path(inquiry_name, contract_name, security_name) for security, security_name in security_names.items()}
                securities = {security: self.file.read(file=security_file) for security, security_file in security_files.items() if os.path.isfile(security_file)}
                yield SecurityQuery(inquiry, contract, securities=securities)


class SecurityFile(DataframeFile):
    @staticmethod
    def dataheader(*args, **kwargs): return ["security", "ticker", "expire", "strike", "date", "price", "underlying", "size", "volume", "interest"]
    @staticmethod
    def datatypes(*args, **kwargs): return {"strike": np.float32, "price": np.float32, "underlying": np.float32, "size": np.int32, "volume": np.int64, "interest": np.int32}
    @staticmethod
    def datetypes(*args, **kwargs): return ["expire", "date"]



