# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Security Objects
@author: Jack Kirby Cook

"""

import numpy as np
import pandas as pd
from abc import ABC

from finance.variables import Variables, Querys
from support.calculations import Calculation, Equation, Variable
from support.mixins import Emptying, Sizing, Partition, Logging
from support.meta import RegistryMeta

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["PricingCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class PricingEquation(Equation, ABC, datatype=pd.Series, vectorize=True):
    j = Variable.Constant("j", "position", Variables.Securities.Position, locator="position")
    qα = Variable.Independent("qα", "supply", np.float32, locator="supply")
    qβ = Variable.Independent("qβ", "demand", np.float32, locator="demand")
    yα = Variable.Independent("yα", "ask", np.float32, locator="ask")
    yβ = Variable.Independent("yβ", "bid", np.float32, locator="bid")

class MarketEquation(PricingEquation):
    y = Variable.Dependent("y", "price", np.float32, function=lambda yα, yβ, *, j: {Variables.Securities.Position.LONG: yα, Variables.Securities.Position.SHORT: yβ}[j])
    q = Variable.Dependent("q", "size", np.float32, function=lambda qα, qβ, *, j: {Variables.Securities.Position.LONG: qα, Variables.Securities.Position.SHORT: qβ}[j])

class LimitEquation(PricingEquation):
    y = Variable.Dependent("y", "price", np.float32, function=lambda yα, yβ, *, j: {Variables.Securities.Position.LONG: yβ, Variables.Securities.Position.SHORT: yα}[j])
    q = Variable.Dependent("q", "size", np.float32, function=lambda qα, qβ, *, j: {Variables.Securities.Position.LONG: qα, Variables.Securities.Position.SHORT: qβ}[j])

class CenteredEquation(PricingEquation):
    y = Variable.Dependent("y", "price", np.float32, function=lambda yα, yβ: (yα + yβ) / 2)
    q = Variable.Dependent("q", "size", np.float32, function=lambda qα, qβ, *, j: {Variables.Securities.Position.LONG: qα, Variables.Securities.Position.SHORT: qβ}[j])


class PricingCalculation(Calculation, ABC, metaclass=RegistryMeta):
    def execute(self, securities, *args, position, **kwargs):
        with self.equation(securities, position=position) as equation:
            yield equation.q()
            yield equation.y()

class AggressiveCalculation(PricingCalculation, equation=MarketEquation, register=Variables.Markets.Pricing.AGGRESSIVE): pass
class PassiveCalculation(PricingCalculation, equation=LimitEquation, register=Variables.Markets.Pricing.PASSIVE): pass
class ModerateCalculation(PricingCalculation, equation=CenteredEquation, register=Variables.Markets.Pricing.MODERATE): pass


class PricingCalculator(Sizing, Emptying, Partition, Logging, ABC, title="Calculated"):
    def __init__(self, *args, pricing, **kwargs):
        super().__init__(*args, **kwargs)
        self.__calculation = dict(PricingCalculation)[pricing](*args, **kwargs)
        self.__pricing = pricing

    def execute(self, securities, *args, **kwargs):
        assert isinstance(securities, pd.DataFrame)
        if self.empty(securities): return
        criteria = all([column in securities.columns for column in ("expire", "strike")])
        instrument = Variables.Securities.Instrument.OPTION if criteria else Variables.Securities.Instrument.STOCK
        query = {Variables.Securities.Instrument.STOCK: Querys.Symbol, Variables.Securities.Instrument.OPTION: Querys.Settlement}[instrument]
        securities = self.calculate(securities, *args, instrument=instrument, **kwargs)
        querys = self.groups(securities, by=query)
        querys = ",".join(list(map(str, querys)))
        size = self.size(securities)
        self.console(f"{str(querys)}[{int(size):.0f}]")
        if self.empty(securities): return
        yield securities

    def calculate(self, securities, *args, **kwargs):
        generator = self.calculator(securities, *args, **kwargs)
        results = pd.concat(list(generator), axis=0)
        results = results.reset_index(drop=True, inplace=False)
        return results

    def calculator(self, securities, *args, instrument, **kwargs):
        for position in list(Variables.Securities.Position):
            pricing = self.calculation(securities, *args, position=position, **kwargs)
            header = {Variables.Securities.Instrument.STOCK: list(Querys.Symbol), Variables.Securities.Instrument.OPTION: list(Querys.Contract)}[instrument]
            dataframe = pd.concat([securities[header], pricing], axis=1)
            if instrument == Variables.Securities.Instrument.STOCK:
                dataframe["option"] = Variables.Securities.Option.EMPTY
            dataframe["instrument"] = instrument
            dataframe["position"] = position
            yield dataframe

    @property
    def calculation(self): return self.__calculation
    @property
    def pricing(self): return self.__pricing



