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
__all__ = ["StockCalculator", "OptionCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class PricingEquation(Equation, ABC, datatype=pd.Series, vectorize=True):
    Θ = Variable.Constant("Θ", "position", Variables.Securities.Position, locator="position")
    qα = Variable.Independent("qα", "supply", np.float32, locator="supply")
    qβ = Variable.Independent("qβ", "demand", np.float32, locator="demand")
    yα = Variable.Independent("yα", "ask", np.float32, locator="ask")
    yβ = Variable.Independent("yβ", "bid", np.float32, locator="bid")

class MarketEquation(PricingEquation):
    y = Variable.Dependent("y", "price", np.float32, function=lambda yα, yβ, *, Θ: {Variables.Securities.Position.LONG: yα, Variables.Securities.Position.SHORT: yβ}[Θ])
    q = Variable.Dependent("q", "size", np.float32, function=lambda qα, qβ, *, Θ: {Variables.Securities.Position.LONG: qα, Variables.Securities.Position.SHORT: qβ}[Θ])

class LimitEquation(PricingEquation):
    y = Variable.Dependent("y", "price", np.float32, function=lambda yα, yβ, *, Θ: {Variables.Securities.Position.LONG: yβ, Variables.Securities.Position.SHORT: yα}[Θ])
    q = Variable.Dependent("q", "size", np.float32, function=lambda qα, qβ, *, Θ: {Variables.Securities.Position.LONG: qα, Variables.Securities.Position.SHORT: qβ}[Θ])

class CenteredEquation(PricingEquation):
    y = Variable.Dependent("y", "price", np.float32, function=lambda yα, yβ: (yα + yβ) / 2)
    q = Variable.Dependent("q", "size", np.float32, function=lambda qα, qβ, *, Θ: {Variables.Securities.Position.LONG: qα, Variables.Securities.Position.SHORT: qβ}[Θ])


class PricingCalculation(Calculation, ABC, metaclass=RegistryMeta):
    def execute(self, securities, *args, position, **kwargs):
        with self.equation(securities, position=position) as equation:
            yield equation.q()
            yield equation.y()

class AggressiveCalculation(PricingCalculation, equation=MarketEquation, register=Variables.Markets.Pricing.AGGRESSIVE): pass
class PassiveCalculation(PricingCalculation, equation=LimitEquation, register=Variables.Markets.Pricing.PASSIVE): pass
class ModerateCalculation(PricingCalculation, equation=CenteredEquation, register=Variables.Markets.Pricing.MODERATE): pass


class SecurityCalculator(Sizing, Emptying, Partition, Logging, ABC, title="Calculated"):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls.__instrument__ = kwargs.get("instrument", getattr(cls, "__instrument__", None))
        cls.__header__ = kwargs.get("header", getattr(cls, "__header__", None))
        cls.__query__ = kwargs.get("query", getattr(cls, "__query__", None))

    def __init__(self, *args, pricing, **kwargs):
        super().__init__(*args, **kwargs)
        self.__calculation = dict(PricingCalculation)[pricing](*args, **kwargs)
        self.__pricing = pricing

    def execute(self, securities, *args, **kwargs):
        assert isinstance(securities, pd.DataFrame)
        if self.empty(securities): return
        securities = self.calculate(securities, *args, **kwargs)
        querys = self.groups(securities, by=self.query)
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

    def calculator(self, securities, *args, **kwargs):
        for position in list(Variables.Securities.Position):
            contract = securities[self.header]
            pricing = self.calculation(securities, *args, position=position, **kwargs)
            dataframe = pd.concat([contract, pricing], axis=1)
            dataframe["instrument"] = self.instrument
            dataframe["position"] = position
            yield dataframe

    @property
    def instrument(self): return type(self).__instrument__
    @property
    def header(self): return type(self).__header__
    @property
    def query(self): return type(self).__query__

    @property
    def calculation(self): return self.__calculation
    @property
    def pricing(self): return self.__pricing


class StockCalculator(SecurityCalculator, instrument=Variables.Securities.Instrument.STOCK, query=Querys.Symbol, header=list(Querys.Symbol)): pass
class OptionCalculator(SecurityCalculator, instrument=Variables.Securities.Instrument.OPTION, query=Querys.Settlement, header=list(Querys.Contract)): pass


