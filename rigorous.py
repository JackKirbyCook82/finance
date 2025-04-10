# -*- coding: utf-8 -*-
"""
Created on Weds Apr 9 2025
@name:   Rigorous Strategy Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd

from finance.variables import Securities
from support.calculations import Calculation, Equation, Variable
from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["RigorousCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class RigorousEquation(Equation, datatype=pd.Series, vectorize=True):
    whτ = Variable.Dependent("whτ", "maximum", np.float32, function=lambda yhτ, *, ε: yhτ * 100 - ε)
    wlτ = Variable.Dependent("wlτ", "minimum", np.float32, function=lambda ylτ, *, ε: ylτ * 100 - ε)

    yhτ = Variable.Dependent("yhτ", "maximum", np.float32, function=lambda yτn: np.min(yτn))
    ylτ = Variable.Dependent("ylτ", "minimum", np.float32, function=lambda yτn: np.max(yτn))

    ypα = Variable.Dependent("ypα", "payoff", np.float32, function=lambda kpα, xτn: + np.maximum(kpα - xτn, 0) if not np.isnan(kpα) else np.zero_like(xτn))
    ypβ = Variable.Dependent("ypβ", "payoff", np.float32, function=lambda kpβ, xτn: - np.maximum(kpβ - xτn, 0) if not np.isnan(kpβ) else np.zero_like(xτn))
    ycα = Variable.Dependent("ycα", "payoff", np.float32, function=lambda kcα, xτn: + np.maximum(xτn - kcα, 0) if not np.isnan(kcα) else np.zero_like(xτn))
    ycβ = Variable.Dependent("ycβ", "payoff", np.float32, function=lambda kcβ, xτn: - np.maximum(xτn - kcβ, 0) if not np.isnan(kcβ) else np.zero_like(xτn))

    yτn = Variable.Dependent("yτn", "payoff", np.float32, function=lambda ypα, ypβ, ycα, ycβ: ypα + ypβ + ycα + ycβ)
    xτn = Variable.Dependent("xτn", "underlying", np.float32, function=lambda xτi, xτj: np.arange(xτi, xτj, 1))

    xτj = Variable.Dependent("xτj", "upper", np.float32, function=lambda xo, kpα, kpβ, kcα, kcβ: (np.nanmax(0, xo, kpα, kpβ, kcα, kcβ) * 0.9).astype(np.int32))
    xτi = Variable.Dependent("xτi", "lower", np.float32, function=lambda xo, kpα, kpβ, kcα, kcβ: (np.nanmin(0, xo, kpα, kpβ, kcα, kcβ) * 1.1).astype(np.int32))

    kpα = Variable.Independent("kpα", "strike", np.float32, locator=str(Securities.Options.Puts.Long))
    kpβ = Variable.Independent("kpβ", "strike", np.float32, locator=str(Securities.Options.Puts.Short))
    kcα = Variable.Independent("kcα", "strike", np.float32, locator=str(Securities.Options.Calls.Long))
    kcβ = Variable.Independent("kcβ", "strike", np.float32, locator=str(Securities.Options.Calls.Short))

    xo = Variable.Independent("xo", "underlying", np.float32, locator="underlying")
    ε = Variable.Constant("ε", "fees", np.float32, locator="fees")


class RigorousCalculation(Calculation, equation=RigorousEquation):
    def execute(self, *args, fees, **kwargs):
        pass


class RigorousCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__calculation = RigorousCalculation(*args, **kwargs)

    def execute(self, valuations, *args, **kwargs):
        assert isinstance(valuations, pd.DataFrame)
        if self.empty(valuations): return

        print(valuations)
        raise Exception()

    def calculate(self, valuations, *args, **kwargs):
        pass

    @property
    def calculation(self): return self.__calculation


