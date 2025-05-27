# -*- coding: utf-8 -*-
"""
Created on Tues Mar 18 2025
@name:   Market Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC

from finance.variables import Variables, Querys, Securities
from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["MarketCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class MarketCalculator(Sizing, Emptying, Partition, Logging, ABC, title="Calculated"):
    def __init__(self, *args, liquidity, priority, **kwargs):
        super().__init__(*args, **kwargs)
        assert callable(liquidity) and callable(priority)
        self.__liquidity = liquidity
        self.__priority = priority

    def execute(self, valuations, options, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame) and isinstance(options, pd.DataFrame)
        if self.empty(valuations): return
        settlements = self.keys(valuations, by=Querys.Settlement)
        settlements = ",".join(list(map(str, settlements)))
        valuations = self.calculate(valuations, options, *args, **kwargs)
        size = self.size(valuations)
        self.console(f"{str(settlements)}[{int(size):.0f}]")
        if self.empty(valuations): return
        yield valuations

    def calculate(self, valuations, options, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame) and isinstance(options, pd.DataFrame)
        valuations["priority"] = valuations.apply(self.priority, axis=1)
        valuations["liquidity"] = valuations.apply(self.liquidity, axis=1).apply(np.floor).astype(np.int32)
        options["liquidity"] = options.apply(self.liquidity, axis=1).apply(np.floor).astype(np.int32)
        options = options.where(options["size"] >= 1).dropna(how="all", inplace=False)
        valuations = valuations.sort_values("priority", axis=0, ascending=False, inplace=False, ignore_index=False)
        valuations = valuations.reset_index(drop=True, inplace=False)
        interest = self.interest(valuations, *args, **kwargs)
        available = self.available(options, *args, **kwargs)
        available = available[available.index.isin(interest)]
        header = list(Querys.Settlement) + list(map(str, Securities.Options))
        valuations["quantity"] = valuations[header].apply(self.converge, axis=1, available=available)
        mask = valuations[("quantity", "")] > 0
        valuations = valuations.where(mask).dropna(how="all", inplace=False)
        valuations = valuations.reset_index(drop=True, inplace=False)
        valuations.columns.names = ["axis", "scenario"]
        return valuations

    @staticmethod
    def interest(valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        parameters = dict(id_vars=list(Querys.Settlement), value_name="strike", var_name="security")
        header = list(Querys.Settlement) + list(map(str, Securities.Options))
        valuations = valuations[header].droplevel(level=1, axis=1)
        interest = pd.melt(valuations, **parameters)
        mask = interest["strike"].isna()
        interest = interest.where(~mask).dropna(how="all", inplace=False)
        interest = pd.MultiIndex.from_frame(interest)
        return interest

    @staticmethod
    def available(options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        function = lambda cols: str(Securities([cols["instrument"], cols["option"], cols["position"]]))
        header = list(Querys.Settlement) + list(Variables.Securities.Security) + ["strike"]
        options = options[header + ["liquidity"]]
        try: options["security"] = options.apply(function, axis=1)
        except ValueError: options = pd.DataFrame(columns=list(options.columns) + ["security"])
        index = list(Querys.Settlement) + ["security", "strike"]
        options = options[index + ["liquidity"]].set_index(index, drop=True, inplace=False)
        return options

    @staticmethod
    def converge(valuation, *arg, available, **kwargs):
        parameters = dict(id_vars=list(Querys.Settlement), value_name="strike", var_name="security")
        valuation = valuation.droplevel(level=1)
        valuation = valuation.to_frame().transpose()
        valuation = pd.melt(valuation, **parameters)
        mask = valuation["strike"].isna()
        valuation = valuation.where(~mask).dropna(how="all", inplace=False)
        index = pd.MultiIndex.from_frame(valuation)
        try: quantity = available.loc[index]
        except KeyError: return 0
        quantity["liquidity"] = np.min(quantity["liquidity"].values).min()
        available["liquidity"] = available["liquidity"].subtract(quantity["liquidity"], fill_value=0)
        return quantity.loc[index, "liquidity"].min().astype(np.int32)

    @property
    def liquidity(self): return self.__liquidity
    @property
    def priority(self): return self.__priority


