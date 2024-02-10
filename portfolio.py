# -*- coding: utf-8 -*-
"""
Created on Thurs Feb 8 2024
@name:   Portfolio Objects
@author: Jack Kirby Cook

"""

import numpy as np

from support.calculations import equation, source

from finance.variables import Securities
from finance.strategies import StrategyCalculation
from finance.valuations import ValuationCalculation

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = []
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


class StrategyCalculation(StrategyCalculation):
    pα = source("pα", str(Securities.Option.Put.Long), position=str(Securities.Option.Put.Long), variables={"ws": "entry", "Δo": "quantity"}, destination=True)
    pβ = source("pβ", str(Securities.Option.Put.Short), position=str(Securities.Option.Put.Short), variables={"ws": "entry", "Δo": "quantity"}, destination=True)
    cα = source("cα", str(Securities.Option.Call.Long), position=str(Securities.Option.Call.Long), variables={"ws": "entry", "Δo": "quantity"}, destination=True)
    cβ = source("cβ", str(Securities.Option.Call.Short), position=str(Securities.Option.Call.Short), variables={"ws": "entry", "Δo": "quantity"}, destination=True)

    def execute(self, feeds, *args, fees, **kwargs):
        yield self.qo(**feeds, fees=fees)
        yield self.Δo(**feeds, fees=fees)
        yield self.ws(**feeds, fees=fees)
        yield self.wo(**feeds, fees=fees)
        yield self.wτn(**feeds, fees=fees)
        yield self.wτx(**feeds, fees=fees)
        yield self.wτi(**feeds, fees=fees)


class VerticalCalculation(StrategyCalculation): pass
class CollarCalculation(StrategyCalculation): pass

class VerticalPutCalculation(StrategyCalculation.Vertical.Call, VerticalCalculation):
    ws = equation("ws", "entry", np.float32, domain=("pα.ws", "pβ.ws"), function=lambda wpα, wpβ: -(wpα + wpβ))
    Δo = equation("Δo", "quantity", np.int32, domain=("pα.Δo", "pβ.Δo"), function=lambda Δpα, Δpβ: np.minimum(Δpα, Δpβ))

class VerticalCallCalculation(StrategyCalculation.Vertical.Put, VerticalCalculation):
    ws = equation("ws", "entry", np.float32, domain=("cα.ws", "cβ.ws"), function=lambda wcα, wcβ: -(wcα + wcβ))
    Δo = equation("Δo", "quantity", np.int32, domain=("cα.Δo", "cβ.Δo"), function=lambda Δcα, Δcβ: np.minimum(Δcα, Δcβ))

class CollarLongCalculation(StrategyCalculation.Collar.Long, CollarCalculation):
    ws = equation("ws", "entry", np.float32, domain=("pα.ws", "cβ.ws"), function=lambda wpα, wcβ: -(wpα + wcβ))
    Δo = equation("Δo", "quantity", np.int32, domain=("pα.Δo", "cβ.Δo"), function=lambda Δpα, Δcβ: np.minimum(Δpα, Δcβ))

class CollarShortCalculation(StrategyCalculation.Collar.Short, CollarCalculation):
    ws = equation("ws", "entry", np.float32, domain=("cα.ws", "pβ.ws"), function=lambda wcα, wpβ: -(wcα + wpβ))
    Δo = equation("Δo", "quantity", np.int32, domain=("cα.Δo", "pβ.Δo"), function=lambda Δcα, Δpβ: np.minimum(Δcα, Δpβ))


class ValuationCalculation(ValuationCalculation):
    Λ = source("Λ", "valuation", position=0, variables={"ws": "entry", "Δo": "quantity"})




