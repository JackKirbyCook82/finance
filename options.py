# -*- coding: utf-8 -*-
"""
Created on Tues May 13 2025
@name:   Options Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC

from finance.variables import Variables
from support.calculations import Calculation, Equation, Variable
from support.mixins import Emptying, Sizing, Partition, Logging

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["OptionCalculator"]
__copyright__ = "Copyright 2025, Jack Kirby Cook"
__license__ = "MIT License"


class OptionEquation(Equation, ABC, datatype=pd.Series, vectorize=True):
    τ = Variable.Dependent("τ", "tau", np.float32, function=lambda to, tτ: (np.datetime64(tτ, "ns") - np.datetime64(to, "ns")) / np.timedelta64(364, 'D'))

    tτ = Variable.Independent("tτ", "expire", np.datetime64, locator="expire")
    to = Variable.Constant("to", "date", np.datetime64, locator="date")

    x = Variable.Independent("x", "underlying", np.float32, locator="underlying")
    σ = Variable.Independent("σ", "volatility", np.float32, locator="volatility")
    μ = Variable.Independent("μ", "trend", np.float32, locator="trend")
    i = Variable.Independent("i", "option", Variables.Securities.Option, locator="option")
    j = Variable.Independent("j", "position", Variables.Securities.Position, locator="position")
    k = Variable.Independent("k", "strike", np.float32, locator="strike")
    r = Variable.Constant("r", "interest", np.float32, locator="interest")
    q = Variable.Constant("q", "dividend", np.float32, locator="dividend")


class OptionCalculation(Calculation, ABC):
    def execute(self, options, *args, **kwargs):
        with self.equation(*args, **kwargs) as equation:
            pass


class OptionCalculator(Sizing, Emptying, Partition, Logging, title="Calculated"):
    def execute(self, options, *args, **kwargs):
        assert isinstance(options, pd.DataFrame)
        if self.empty(options): return

        print(options)
        raise Exception()

