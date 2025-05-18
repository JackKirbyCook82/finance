# -*- coding: utf-8 -*-
"""
Created on Weds Apr 9 2025
@name:   Payoff Strategy Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from enum import Enum
from itertools import product

from finance.variables import Querys, Variables, Securities
from support.calculations import Calculation, Equation, Variable
from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["PayoffCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class PayoffEquation(Equation, datatype=pd.Series, vectorize=True):
    wlτ = Variable.Dependent("wlτ", "minimum", np.float32, function=lambda yτn, *, ε: np.min(yτn) * 100 - ε)
    whτ = Variable.Dependent("whτ", "maximum", np.float32, function=lambda yτn, *, ε: np.max(yτn) * 100 - ε)

    ypα = Variable.Dependent("ypα", "payoff", np.float32, function=lambda kpα, xτn: + np.maximum(kpα - xτn, 0) if not np.isnan(kpα) else np.zeros_like(xτn))
    ypβ = Variable.Dependent("ypβ", "payoff", np.float32, function=lambda kpβ, xτn: - np.maximum(kpβ - xτn, 0) if not np.isnan(kpβ) else np.zeros_like(xτn))
    ycα = Variable.Dependent("ycα", "payoff", np.float32, function=lambda kcα, xτn: + np.maximum(xτn - kcα, 0) if not np.isnan(kcα) else np.zeros_like(xτn))
    ycβ = Variable.Dependent("ycβ", "payoff", np.float32, function=lambda kcβ, xτn: - np.maximum(xτn - kcβ, 0) if not np.isnan(kcβ) else np.zeros_like(xτn))
    xτ = Variable.Dependent("xτ", "payoff", np.float32, function=lambda xτn, so: xτn * int(so.position))

    yτn = Variable.Dependent("yτn", "payoff", np.float32, function=lambda ypα, ypβ, ycα, ycβ, xτ: ypα + ypβ + ycα + ycβ + xτ)
    xτn = Variable.Dependent("xτn", "underlying", np.float32, function=lambda xτi, xτj: np.arange(xτi, xτj, 1))

    xτj = Variable.Dependent("xτj", "upper", np.float32, function=lambda xo, kpα, kpβ, kcα, kcβ: (np.nanmax([0, xo, kpα, kpβ, kcα, kcβ]) * 1.1).astype(np.int32))
    xτi = Variable.Dependent("xτi", "lower", np.float32, function=lambda xo, kpα, kpβ, kcα, kcβ: (np.nanmin([0, xo, kpα, kpβ, kcα, kcβ]) * 0.9).astype(np.int32))

    kpα = Variable.Independent("kpα", "strike", np.float32, locator=str(Securities.Options.Puts.Long))
    kpβ = Variable.Independent("kpβ", "strike", np.float32, locator=str(Securities.Options.Puts.Short))
    kcα = Variable.Independent("kcα", "strike", np.float32, locator=str(Securities.Options.Calls.Long))
    kcβ = Variable.Independent("kcβ", "strike", np.float32, locator=str(Securities.Options.Calls.Short))

    xo = Variable.Independent("xo", "underlying", np.float32, locator="underlying")
    so = Variable.Independent("so", "strategy", Enum, locator="strategy")
    ε = Variable.Constant("ε", "fees", np.float32, locator="fees")

    def execute(self, *args, **kwargs):
        yield self.wlτ()
        yield self.whτ()


class PayoffCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__calculation = Calculation[pd.Series](*args, equation=PayoffEquation, **kwargs)

    def execute(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        if self.empty(valuations): return
        settlements = self.keys(valuations, by=Querys.Settlement)
        settlements = ",".join(list(map(str, settlements)))
        valuations = self.calculate(valuations, *args, **kwargs)
        size = self.size(valuations)
        self.console(f"{str(settlements)}[{int(size):.0f}]")
        if self.empty(valuations): return

        print(valuations)
        raise Exception()

        yield valuations

    def calculate(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        header = list(Querys.Settlement) + list(map(str, Securities.Options)) + ["underlying", "strategy"]
        options = valuations[header].droplevel(1, axis=1)
        payoffs = self.calculation(options, *args, **kwargs)
        assert isinstance(payoffs, pd.DataFrame)
        payoffs = payoffs.rename(columns={str(scenario).lower(): scenario for scenario in Variables.Scenario})
        columns = list(product(["payoff"], payoffs.columns))
        payoffs.columns = pd.MultiIndex.from_tuples(columns)
        results = pd.concat([valuations, payoffs], axis=1)
        results.columns.names = results.columns.names
        return results

    @property
    def calculation(self): return self.__calculation


