# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np

from support.cleaning import Filter

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["SecurityFilter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


stocks_index = {"instrument": str, "position": str, "ticker": str, "date": np.datetime64}
stocks_columns = {"price": np.float32, "size": np.float32, "volume": np.float32}
options_index = {"instrument": str, "position": str, "strike": np.float32, "ticker": str, "expire": np.datetime64, "date": np.datetime64}
options_columns = {"price": np.float32, "underlying": np.float32, "size": np.float32, "volume": np.float32, "interest": np.float32}


# stocks_axes = Axes.Dataframe(index=stocks_index, columns=stocks_columns)
# stocks_data = FileData.Dataframe(header=stocks_index | stocks_columns)
# options_axes = Axes.Dataframe(index=options_index, columns=options_columns)
# options_data = FileData.Dataframe(header=options_index | options_columns)
# contract_query = FileQuery("contract", Contract.tostring, Contract.fromstring)


# class StockFile(FileDirectory, variable="stocks", query=contract_query, data=stocks_data): pass
# class OptionFile(FileDirectory, variable="options", query=contract_query, data=options_data): pass
# class SecurityHeader(Header, variables={"stocks": stocks_axes, "options": options_axes}): pass


class SecurityFilter(Filter, variables=["stocks", "options"], query="contract"):
    pass



