# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import os
import logging
import numpy as np
import xarray as xr
from datetime import datetime as Datetime

from support.files import DataframeFile
from support.pipelines import Producer, Processor, Consumer

from finance.variables import Query, Contract

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityFilter", "SecurityParser", "SecurityLoader", "SecuritySaver", "SecurityFile"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class SecurityQuery(Query, fields=["securities"]): pass
class SecurityFilter(Processor, title="Filtered"):
    def execute(self, query, *args, **kwargs):
        securities = query.securities
        mask = self.mask(securities, *args, **kwargs)
        securities = securities.where(mask).dropna(axis=0, how="all")
        securities = securities.reset_index(drop=True, inplace=False)
        query = query(securities=securities)
        __logger__.info(f"Filter: {repr(self)}|{str(query)}[{len(securities.index):.0f}]")
        yield query

    @staticmethod
    def mask(dataframe, *args, size=0, interest=0, volume=0, **kwargs):
        mask = (dataframe["size"].notna() & dataframe["interest"].notna() & dataframe["volume"].notna())
        mask = (mask & dataframe["size"] >= size) if size is not None else mask
        mask = (mask & dataframe["interest"] >= interest) if interest is not None else mask
        mask = (mask & dataframe["volume"] >= volume) if volume is not None else volume
        return mask


class SecurityParser(Processor, title="Parsed"):
    def execute(self, query, *args, **kwargs):
        securities = query.securities
        securities = self.parse(securities, *args, **kwargs)
        query = query(securities=securities)
        yield query

    @staticmethod
    def parse(dataframe, *args, **kwargs):
        index = ["date", "ticker", "expire", "strike"]
        dataframe = dataframe.drop_duplicates(subset=["security", "date", "ticker", "expire", "strike"], keep="last", inplace=False)
        dataframe = {security: dataframe.drop("security", axis=1, inplace=False) for security, dataframe in iter(dataframe.groupby("security"))}
        dataframe = {security: dataframe.set_index(index, inplace=False, drop=True) for security, dataframe in dataframe.items()}
        dataframe = {security: xr.Dataset.from_dataframe(dataframe) for security, dataframe in dataframe.items()}
        return dataframe


class SecuritySaver(Consumer, title="Saved"):
    def __init__(self, *args, file, **kwargs):
        assert isinstance(file, SecurityFile)
        super().__init__(*args, **kwargs)
        self.__file = file

    def execute(self, query, *args, **kwargs):
        securities = query.securities
        inquiry_name = str(query.inquiry.strftime("%Y%m%d_%H%M%S"))
        inquiry_folder = self.file.path(inquiry_name)
        if not os.path.isdir(inquiry_folder):
            os.mkdir(inquiry_folder)
        contract_name = "_".join([str(query.contract.ticker), str(query.contract.expire.strftime("%Y%m%d"))])
        contract_folder = self.file.path(inquiry_name, contract_name)
        if not os.path.isdir(contract_folder):
            os.mkdir(contract_folder)
        securities_file = self.file.path(inquiry_name, contract_name, "securities.csv")
        if securities is not None and not securities.empty:
            self.file.write(securities, file=securities_file, filemode="w")
            __logger__.info("Saved: {}[{}]".format(repr(self), str(securities_file)))


class SecurityLoader(Producer, title="Loaded"):
    def __init__(self, *args, file, **kwargs):
        assert isinstance(file, SecurityFile)
        super().__init__(*args, **kwargs)
        self.__file = file

    def execute(self, *args, tickers=None, expires=None, **kwargs):
        function = lambda foldername: Datetime.strptime(foldername, "%Y%m%d_%H%M%S")
        inquiry_folders = self.file.directory()
        for inquiry_name in sorted(inquiry_folders, key=function, reverse=False):
            inquiry = function(inquiry_name)
            contract_names = self.file.directory(inquiry_name)
            for contract_name in contract_names:
                contract = Contract(*str(contract_name).split("_"))
                ticker = str(contract.ticker).upper()
                if tickers is not None and ticker not in tickers:
                    continue
                expire = Datetime.strptime(os.path.splitext(contract.expire)[0], "%Y%m%d").date()
                if expires is not None and expire not in expires:
                    continue
                securities_file = self.file.path(inquiry_name, contract_name, "securities.csv")
                securities = self.file.read(file=securities_file) if os.path.isfile(securities_file) else None
                yield SecurityQuery(inquiry, contract, securities=securities)


class SecurityFile(DataframeFile):
    @staticmethod
    def dataheader(*args, **kwargs): return ["security", "ticker", "expire", "strike", "date", "price", "underlying", "size", "volume", "interest", "entry", "quantity"]
    @staticmethod
    def datatypes(*args, **kwargs): return {"strike": np.float32, "price": np.float3, "underlying": np.float32, "size": np.int32, "volume": np.int64, "interest": np.int32, "entry": np.float32, "quantity": np.int32}
    @staticmethod
    def datetypes(*args, **kwargs): return ["expire", "date"]



