# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Strategy Objects
@author: Jack Kirby Cook

"""

import numpy as np
import xarray as xr
from enum import IntEnum
from collections import namedtuple as ntuple

from support.pipelines import Calculator
from support.calculations import Calculation, equation

from finance.securities import Positions, Instruments, Securities

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Strategy", "Strategies", "Calculations", "StrategyCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


Spreads = IntEnum("Strategy", ["STRANGLE", "COLLAR", "VERTICAL", "CONDOR"], start=1)
class Strategy(ntuple("Strategy", "spread instrument position")):
    def __new__(cls, spread, instrument, position, *args, **kwargs): return super().__new__(cls, spread, instrument, position)
    def __init__(self, *args, **kwargs): self.__securities = kwargs["securities"]
    def __str__(self): return "|".join([str(value.name).lower() for value in self if bool(value)])

    @property
    def securities(self): return self.__securities

StrangleLong = Strategy(Spreads.STRANGLE, 0, Positions.LONG, securities=[Securities.Option.Put.Long, Securities.Option.Call.Long])
CollarLong = Strategy(Spreads.COLLAR, 0, Positions.LONG, securities=[Securities.Option.Put.Long, Securities.Option.Call.Short, Securities.Stock.Long])
CollarShort = Strategy(Spreads.COLLAR, 0, Positions.SHORT, securities=[Securities.Option.Call.Long, Securities.Option.Put.Short, Securities.Stock.Short])
VerticalPut = Strategy(Spreads.VERTICAL, Instruments.PUT, 0, securities=[Securities.Option.Put.Long, Securities.Option.Put.Short])
VerticalCall = Strategy(Spreads.VERTICAL, Instruments.CALL, 0, securities=[Securities.Option.Call.Long, Securities.Option.Call.Short])
Condor = Strategy(Spreads.CONDOR, 0, 0, securities=[Securities.Option.Put.Long, Securities.Option.Call.Long, Securities.Option.Put.Short, Securities.Option.Call.Short])

class Strategies:
    Condor = Condor
    class Strangle:
        Long = StrangleLong
    class Collar:
        Long = CollarLong
        Short = CollarShort
    class Vertical:
        Put = VerticalPut
        Call = VerticalCall


sources = {"pα": Securities.Option.Put.Long, "pβ": Securities.Option.Put.Short, "cα": Securities.Option.Call.Long, "cβ": Securities.Option.Call.Short, "sα": Securities.Stock.Long, "sβ": Securities.Stock.Short}
variables = {"τ": "tau", "w": "price", "k": "strike", "x": "time", "q": "size", "i": "interest"}
results = {"τ": "tau", "wo": "spot", "vmn": "future"}
constants = {"ε": "fees"}

class StrategyCalculation(Calculation, sources=sources, variables=variables, constants=constants, results=results): pass
class StrangleCalculation(StrategyCalculation): pass
class VerticalCalculation(StrategyCalculation): pass
class CollarCalculation(StrategyCalculation): pass

class StrangleLongCalculation(StrangleCalculation):
    τ = equation("tau", np.int16, domain=("pα.τ", "cα.τ"), function=lambda τpα, τcα: τpα)
    x = equation("time", np.datetime64, domain=("pα.x", "cα.x"), function=lambda xpα, xcα: np.maximum.outer(xpα, xcα))
    q = equation("size", np.float32, domain=("pα.q", "cα.q"), function=lambda qpα, qcα: np.minimum.outer(qpα, qcα))
    i = equation("interest", np.int32, domain=("pα.i", "cα.i"), function=lambda ipα, icα: np.minimum.outer(ipα, icα))

    wo = equation("spot", np.float32, domain=("pα.w", "cα.w", "ε"), function=lambda wpα, wcα, ε: - np.add.outer(wpα, wcα) * 100 - ε)
    vmn = equation("minimum", np.float32, domain=("pα.k", "cα.k", "ε"), function=lambda kpα, kcα, ε: + np.maximum(np.add.outer(kpα, -kcα), 0) * 100 - ε)
    vmx = equation("maximum", np.float32, domain=("pα.k", "cα.k", "ε"), function=lambda kpα, kcα, ε: + np.ones((kpα.shape, kcα.shape)) * np.inf * 100 - ε)

class VerticalPutCalculation(VerticalCalculation):
    τ = equation("tau", np.int16, domain=("pα.τ", "pβ.τ"), function=lambda τpα, τpβ: τpα)
    x = equation("time", np.datetime64, domain=("pα.x", "pβ.x"), function=lambda xpα, xpβ: np.maximum.outer(xpα, xpβ))
    q = equation("size", np.float32, domain=("pα.q", "pβ.q"), function=lambda qpα, qpβ: np.minimum.outer(qpα, qpβ))
    i = equation("interest", np.int32, domain=("pα.i", "pβ.i"), function=lambda ipα, ipβ: np.minimum.outer(ipα, ipβ))

    wo = equation("spot", np.float32, domain=("pα.w", "pβ.w", "ε"), function=lambda wpα, wpβ, ε: - np.add.outer(wpα, -wpβ) * 100 - ε)
    vmn = equation("minimum", np.float32, domain=("pα.k", "pβ.k", "ε"), function=lambda kpα, kpβ, ε: + np.minimum(np.add.outer(kpα, -kpβ), 0) * 100 - ε)
    vmx = equation("maximum", np.float32, domain=("pα.k", "pβ.k", "ε"), function=lambda kpα, kpβ, ε: + np.maximum(np.add.outer(kpα, -kpβ), 0) * 100 - ε)

class VerticalCallCalculation(VerticalCalculation):
    τ = equation("tau", np.int16, domain=("cα.τ", "cβ.τ"), function=lambda τcα, τcβ: τcα)
    x = equation("time", np.datetime64, domain=("cα.x", "cβ.x"), function=lambda xcα, xcβ: np.maximum.outer(xcα, xcβ))
    q = equation("size", np.float32, domain=("cα.q", "cβ.q"), function=lambda qcα, qcβ: np.minimum.outer(qcα, qcβ))
    i = equation("interest", np.int32, domain=("cα.i", "cβ.i"), function=lambda icα, icβ: np.minimum.outer(icα, icβ))

    wo = equation("spot", np.float32, domain=("cα.w", "cβ.w", "ε"), function=lambda wcα, wcβ, ε: - np.add.outer(wcα, -wcβ) * 100 - ε)
    vmn = equation("minimum", np.float32, domain=("cα.k", "cβ.k", "ε"), function=lambda kcα, kcβ, ε: + np.minimum(np.add.outer(-kcα, kcβ), 0) * 100 - ε)
    vmx = equation("maximum", np.float32, domain=("cα.k", "cβ.k", "ε"), function=lambda kcα, kcβ, ε: + np.maximum(np.add.outer(-kcα, kcβ), 0) * 100 - ε)

class CollarLongCalculation(CollarCalculation):
    τ = equation("tau", np.int16, domain=("pα.τ", "cβ.τ"), function=lambda τpα, τcβ: τpα)
    x = equation("time", np.datetime64, domain=("pα.x", "cβ.x"), function=lambda xpα, xcβ: np.maximum.outer(xpα, xcβ))
    q = equation("size", np.float32, domain=("pα.q", "cβ.q"), function=lambda qpα, qcβ: np.minimum.outer(qpα, qcβ))
    i = equation("interest", np.int32, domain=("pα.i", "cβ.i"), function=lambda ipα, icβ: np.minimum.outer(ipα, icβ))

    wo = equation("spot", np.float32, domain=("pα.w", "cβ.w", "ε"), function=lambda wpα, wcβ, wsα, ε: (-np.add.outer(wpα, -wcβ) - wsα) * 100 - ε)
    vmn = equation("minimum", np.float32, domain=("pα.k", "cβ.k", "ε"), function=lambda kpα, kcβ, ε: + np.minimum.outer(kpα, kcβ) * 100 - ε)
    vmx = equation("maximum", np.float32, domain=("pα.k", "cβ.k", "ε"), function=lambda kpα, kcβ, ε: + np.maximum.outer(kpα, kcβ) * 100 - ε)

class CollarShortCalculation(CollarCalculation):
    τ = equation("tau", np.int16, domain=("pβ.τ", "cα.τ"), function=lambda τpβ, τcα: τpβ)
    x = equation("time", np.datetime64, domain=("pβ.x", "cα.x"), function=lambda xpβ, xcα: np.maximum.outer(xpβ, xcα))
    q = equation("size", np.float32, domain=("pβ.q", "cα.q"), function=lambda qpβ, qcα: np.minimum.outer(qpβ, qcα))
    i = equation("interest", np.int32, domain=("pβ.i", "cα.i"), function=lambda ipβ, icα: np.minimum.outer(ipβ, icα))

    wo = equation("spot", np.float32, domain=("pβ.w", "cα.w", "ε"), function=lambda wpβ, wcα, wsβ, ε: (-np.add.outer(-wpβ, wcα) + wsβ) * 100 - ε)
    vmn = equation("minimum", np.float32, domain=("pβ.k", "cα.k", "ε"), function=lambda kpβ, kcα, ε: + np.minimum.outer(-kpβ, -kcα) * 100 - ε)
    vmx = equation("maximum", np.float32, domain=("pβ.k", "cα.k", "ε"), function=lambda kpβ, kcα, ε: + np.maximum.outer(-kpβ, -kcα) * 100 - ε)

class CondorCalculation(StrategyCalculation):
    pass

class Calculations:
    Condor = CondorCalculation
    class Strangle:
        Long = StrangleLongCalculation
    class Collar:
        Long = CollarLongCalculation
        Short = CollarShortCalculation
    class Vertical:
        Put = VerticalPutCalculation
        Call = VerticalCallCalculation


calculations = {Strategies.Strangle.Long: Calculations.Strangle.Long, Strategies.Strangle.Short: Calculations.Strangle.Short}
calculations.update({Strategies.Vertical.Put: Calculations.Vertical.Put, Strategies.Vertical.Call: Calculations.Vertical.Call})
calculations.update({Strategies.Collar.Long: Calculations.Collar.Long, Strategies.Collar.Short: Calculations.Collar.Short})
calculations.update({Strategies.Condor: Calculations.Condor})
class StrategyCalculator(Calculator, calculations=calculations):
    def execute(self, contents, *args, **kwargs):
        ticker, expire, datasets = contents
        assert isinstance(datasets, dict)
        assert all([isinstance(security, xr.Dataset) for security in datasets.values()])
        for strategy, calculation in self.calculations.items():
            strategies = calculation(*args, **datasets, **kwargs)
            yield ticker, expire, strategy, strategies




