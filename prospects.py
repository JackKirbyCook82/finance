# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 21 2024
@name:   Prospect Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from collections import namedtuple as ntuple

from finance.concepts import Concepts, Querys, Securities
from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["ProspectCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class ProspectCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, liquidity, priority, **kwargs):
        super().__init__(*args, **kwargs)
        assert callable(liquidity) and callable(priority)
        self.__liquidity = liquidity
        self.__priority = priority

    def execute(self, securities, valuations, /, **kwargs):
        assert isinstance(securities, pd.DataFrame) and isinstance(valuations, pd.DataFrame)
        if self.empty(valuations): return

        Columns = ntuple("Columns", "contract security option valuation")
        columns = Columns(list(map(str, Querys.Contract)), list(map(str, Concepts.Securities.Security)), list(map(str, Securities.Options)), [])
        columns = Columns(columns.contract, columns.security, columns.option, [column for column in valuations.columns if column not in columns.option])
        dataframe = valuations.melt(id_vars=columns.valuation, value_vars=columns.option, var_name="security", value_name="strike")
        dataframe[columns.security] = dataframe["security"].str.split("|", expand=True)
        dataframe = dataframe.drop(columns=["security"]).dropna(subset=["strike"])
        dataframe = dataframe[columns.security + ["strike"] + columns.contract]
        dataframe = dataframe.loc[:, ~dataframe.columns.duplicated()]
        dataframe[columns.security] = dataframe[columns.security].astype(str)
        securities = securities.assign(**securities[columns.security].astype(str).to_dict())
        overlap = list(set(securities.columns) & set(dataframe.columns))
        securities = securities.merge(dataframe[overlap], on=overlap, how="inner")

        print(valuations, "\n")
        print(securities)
        raise Exception()

        settlements = self.keys(valuations, by=Querys.Settlement)
        settlements = ",".join(list(map(str, settlements)))
        prospects = self.calculate(valuations, **kwargs)
        size = self.size(prospects)
        self.console(f"{str(settlements)}[{int(size):.0f}]")
        if self.empty(prospects): return
        yield prospects

    def calculate(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        valuations["liquidity"] = valuations.apply(self.liquidity, axis=1).apply(np.floor)
        valuations["priority"] = valuations.apply(self.priority, axis=1)
        valuations = valuations.where(valuations["liquidity"] >= 1).dropna(how="all", inplace=False)
        valuations = valuations.where(valuations["priority"] >= 0).dropna(how="all", inplace=False)
        valuations = valuations.sort_values("priority", axis=0, ascending=False, inplace=False, ignore_index=False)
        valuations = valuations.reset_index(drop=True, inplace=False)
        valuations["status"] = Concepts.Markets.Status.PROSPECT
        valuations["quantity"] = 1
        return valuations

    @property
    def liquidity(self): return self.__liquidity
    @property
    def priority(self): return self.__priority







