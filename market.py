# -*- coding: utf-8 -*-
"""
Created on Tues Mar 18 2025
@name:   Market Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC
from datetime import datetime as Datetime
from collections import namedtuple as ntuple

from finance.variables import Variables, Querys, Securities, Strategies
from support.mixins import Emptying, Sizing, Partition, Logging
from support.meta import ParameterMeta
from support.files import Saver

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["AcquisitionCalculator", "AcquisitionSaver", "AcquisitionParameters"]
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
        dataframe = self.calculate(valuations, options, *args, **kwargs)
        settlements = self.groups(valuations, by=Querys.Settlement)
        settlements = ",".join(list(map(str, settlements)))
        size = self.size(dataframe)
        self.console(f"{str(settlements)}[{int(size):.0f}]")
        if self.empty(dataframe): return
        yield dataframe

    def calculate(self, valuations, options, *args, **kwargs):
        valuations["priority"] = valuations.apply(self.priority, axis=1)
        valuations["size"] = valuations.apply(self.liquidity, axis=1).apply(np.floor).astype(np.int32)
        options["size"] = options.apply(self.liquidity, axis=1).apply(np.floor).astype(np.int32)
        options = options.where(options["size"] >= 1).dropna(how="all", inplace=False)
        valuations = valuations.sort_values("priority", axis=0, ascending=False, inplace=False, ignore_index=False)
        valuations = valuations.reset_index(drop=True, inplace=False)
        interest = self.interest(valuations, *args, **kwargs)
        available = self.available(options, *args, **kwargs)
        available = available[available.index.isin(interest)]
        header = list(Querys.Settlement) + list(map(str, Securities.Options))
        valuations["size"] = valuations[header].apply(self.converge, axis=1, available=available)
        mask = valuations[("size", "")] > 0
        valuations = valuations.where(mask).dropna(how="all", inplace=False)
        valuations = valuations.reset_index(drop=True, inplace=False)
        return valuations

    @staticmethod
    def interest(valuations, *args, **kwargs):
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
        function = lambda cols: str(Securities([cols["instrument"], cols["option"], cols["position"]]))
        header = list(Querys.Settlement) + list(Variables.Securities.Security) + ["strike"]
        options = options[header + ["size"]]
        try: options["security"] = options.apply(function, axis=1)
        except ValueError: options = pd.DataFrame(columns=list(options.columns) + ["security"])
        index = list(Querys.Settlement) + ["security", "strike"]
        options = options[index + ["size"]].set_index(index, drop=True, inplace=False)
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
        quantity["size"] = np.min(quantity["size"].values).min()
        available["size"] = available["size"].subtract(quantity["size"], fill_value=0)
        return quantity.loc[index, "size"].min().astype(np.int32)

    @property
    def liquidity(self): return self.__liquidity
    @property
    def priority(self): return self.__priority


class AcquisitionParameters(metaclass=ParameterMeta):
    order = ["valuation", "scenario", "strategy"] + list(Querys.Settlement) + list(map(str, Securities.Options)) + ["underlying", "tau", "size", "revenue", "expense", "purchase", "borrow", "spot", "future", "npv"]
    types = {"ticker": str, " ".join(map(str, Securities.Options)): str, "underlying": np.float32, "tau size": np.float32, "revenue expense": np.float32, "purchase borrow": np.float32, "spot future": np.float32, "npv": np.float32}
    parsers = dict(valuation=Variables.Valuations.Valuation, scenario=Variables.Valuations.Scenario, strategy=Strategies)
    formatters = dict(valuation=str, scenario=str, strategy=str)
    dates = dict(expire="%Y%m%d")


class AcquisitionCalculator(MarketCalculator): pass
class AcquisitionSaver(Saver):
    def categorize(self, dataframe, *args, **kwargs):
        Header = ntuple("Header", "axis scenario")
        headers = {scenario: [Header(axis, scenario) for (axis, scenario) in dataframe.columns] for scenario in list(Variables.Valuations.Scenario)}
        headers = {scenario: [header for header in contents if not bool(header.scenario) or header.scenario == scenario] for scenario, contents in headers.items()}
        dataframes = [dataframe[header].assign(scenario=scenario).droplevel(level=1, axis=1) for scenario, header in headers.items()]
        dataframe = pd.concat(dataframes, axis=0)
        current = ".".join([Datetime.now().strftime("%Y%m%d"), "csv"])
        yield current, dataframe


