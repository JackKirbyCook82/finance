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

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["PricingCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"


class PricingEquation(Equation, ABC, datatype=pd.Series, vectorize=True):
    qα = Variable.Independent("qα", "supply", np.float32, locator="supply")
    qβ = Variable.Independent("qβ", "demand", np.float32, locator="demand")
    yα = Variable.Independent("yα", "ask", np.float32, locator="ask")
    yβ = Variable.Independent("yβ", "bid", np.float32, locator="bid")
    jo = Variable.Constant("jo", "position", Variables.Securities.Position, locator="position")

    def execute(self, *args, **kwargs):
        yield self.q(*args, **kwargs)
        yield self.y(*args, **kwargs)


class AggressiveEquation(PricingEquation, register=Variables.Markets.Pricing.AGGRESSIVE):
    q = Variable.Dependent("q", "size", np.float32, function=lambda qα, qβ, *, jo: {Variables.Securities.Position.LONG: qα, Variables.Securities.Position.SHORT: qβ}[jo])
    y = Variable.Dependent("y", "price", np.float32, function=lambda yα, yβ, *, jo: {Variables.Securities.Position.LONG: yα, Variables.Securities.Position.SHORT: yβ}[jo])

class PassiveEquation(PricingEquation, register=Variables.Markets.Pricing.PASSIVE):
    q = Variable.Dependent("q", "size", np.float32, function=lambda qα, qβ, *, jo: {Variables.Securities.Position.LONG: qα, Variables.Securities.Position.SHORT: qβ}[jo])
    y = Variable.Dependent("y", "price", np.float32, function=lambda yα, yβ, *, jo: {Variables.Securities.Position.LONG: yβ, Variables.Securities.Position.SHORT: yα}[jo])

class ModerateEquation(PricingEquation, register=Variables.Markets.Pricing.MODERATE):
    q = Variable.Dependent("q", "size", np.float32, function=lambda qα, qβ, *, jo: {Variables.Securities.Position.LONG: qα, Variables.Securities.Position.SHORT: qβ}[jo])
    y = Variable.Dependent("y", "price", np.float32, function=lambda yα, yβ: (yα + yβ) / 2)


class PricingCalculator(Sizing, Emptying, Partition, Logging, ABC, title="Calculated"):
    def __init__(self, *args, pricing, **kwargs):
        assert pricing in list(Variables.Markets.Pricing)
        super().__init__(*args, **kwargs)
        self.__calculation = Calculation[pricing](*args, equation=PricingEquation[pricing], **kwargs)
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
        generator = list(self.calculator(securities, *args, **kwargs))
        securities = pd.concat(list(generator), axis=0)
        securities = securities.reset_index(drop=True, inplace=False)
        return securities

    def calculator(self, securities, *args, instrument, **kwargs):
        for position in list(Variables.Securities.Position):
            pricing = self.calculation(securities, *args, position=position, **kwargs)
            assert isinstance(pricing, pd.DataFrame)
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



