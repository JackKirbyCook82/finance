# -*- coding: utf-8 -*-
"""
Created on Fri May 17 2024
@name:   Stability Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd

from finance.variables import Variables
from finance.operations import Operations
from support.calculations import Variable, Equation, Calculation

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["StabilityCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class StabilityEquation(Equation):
    y = Variable("y", "value", np.float32, function=lambda q, Θ, Φ, Ω, Δ: q * Θ * Φ * Δ * (1 - Θ * Ω) / 2)
    m = Variable("m", "trend", np.float32, function=lambda q, Θ, Φ, Ω: q * Θ * Φ * Ω)
    Ω = Variable("Ω", "omega", np.int32, function=lambda x, k: np.sign(x / k - 1))
    Δ = Variable("Δ", "delta", np.int32, function=lambda x, k: np.subtract(x, k))
    Θ = Variable("Θ", "theta", np.int32, function=lambda i: + int(Variables.Theta(str(i))))
    Φ = Variable("Φ", "phi", np.int32, function=lambda j: + int(Variables.Phi(str(j))))

    x = Variable("x", "underlying", np.float32, position=0, locator="underlying")
    q = Variable("q", "quantity", np.int32, position=0, locator="quantity")
    i = Variable("i", "option", Variables.Options, position=0, locator="option")
    j = Variable("j", "position", Variables.Positions, position=0, locator="position")
    k = Variable("k", "strike", np.float32, position=0, locator="strike")


class StabilityCalculation(Calculation, equation=StabilityEquation):
    def execute(self, portfolios, *args, **kwargs):
        equation = self.equation(*args, **kwargs)
        yield equation.y(portfolios)
        yield equation.m(portfolios)


class StabilityCalculator(Operations.Processor):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.__calculation = StabilityCalculation(*args, **kwargs)

    def processor(self, contents, *args, **kwargs):
        exposures = contents[Variables.Datasets.EXPOSURE]
        allocations = contents[Variables.Datasets.ALLOCATION]
        assert isinstance(exposures, pd.DataFrame) and isinstance(allocations, pd.DataFrame)

        print(exposures)
        print(allocations)
        raise Exception()

    @property
    def calculation(self): return self.__calculation




