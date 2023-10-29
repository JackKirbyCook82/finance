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


variables = {"tτ": "tau", "w": "price", "k": "strike", "x": "time", "q": "size", "i": "interest"}
calculation = ["τ", "wo", "vmn"]
class StrategyCalculation(Calculation, variables=variables, calculation=calculation): pass
class StrangleCalculation(StrategyCalculation): pass
class VerticalCalculation(StrategyCalculation): pass
class CollarCalculation(StrategyCalculation): pass

class StrangleLongCalculation(StrangleCalculation, sources={"pα": Securities.Option.Put.Long, "cα": Securities.Option.Call.Long}):
    τ = equation("tau", np.int16, domain=("pα.τ", "cα.τ"), function=lambda τpα, τcα: τpα)
    x = equation("time", np.datetime64, domain=("pα.x", "cα.x"), function=lambda xpα, xcα: np.maximum.outer(xpα, xcα))
    q = equation("size", np.float32, domain=("pα.q", "cα.q"), function=lambda qpα, qcα: np.minimum.outer(qpα, qcα))
    i = equation("interest", np.int32, domain=("pα.i", "cα.i"), function=lambda ipα, icα: np.minimum.outer(ipα, icα))

    wo = equation("spot", np.float32, domain=("pα.w", "cα.w"), function=lambda wpα, wcα: - np.add.outer(wpα, wcα))
    vo = equation("current", np.float32, domain=("pα.w", "cα.w"), function=lambda : )
    vmn = equation("minimum", np.float32, domain=("pα.k", "cα.k"), function=lambda kpα, kcα: + np.maximum(np.add.outer(kpα, -kcα), 0))
    vmx = equation("maximum", np.float32, domain=("pα.k", "cα.k"), function=lambda kpα, kcα: + np.ones((kpα.shape, kcα.shape)) * np.inf)

class VerticalPutCalculation(VerticalCalculation, sources={"pα": Securities.Option.Put.Long, "pβ": Securities.Option.Put.Short}):
    τ = equation("tau", np.int16, domain=("pα.τ", "pβ.τ"), function=lambda τpα, τpβ: τpα)
    x = equation("time", np.datetime64, domain=("pα.x", "pβ.x"), function=lambda xpα, xpβ: np.maximum.outer(xpα, xpβ))
    q = equation("size", np.float32, domain=("pα.q", "pβ.q"), function=lambda qpα, qpβ: np.minimum.outer(qpα, qpβ))
    i = equation("interest", np.int32, domain=("pα.i", "pβ.i"), function=lambda ipα, ipβ: np.minimum.outer(ipα, ipβ))

    wo = equation("spot", np.float32, domain=("pα.w", "pβ.w"), function=lambda wpα, wpβ: - np.add.outer(wpα, -wpβ))
    vo = equation("current", np.float32, domain=("pα.w", "pβ.w"), function=lambda : )
    vmn = equation("minimum", np.float32, domain=("pα.k", "pβ.k"), function=lambda kpα, kpβ: + np.minimum(np.add.outer(kpα, -kpβ), 0))
    vmx = equation("maximum", np.float32, domain=("pα.k", "pβ.k"), function=lambda kpα, kpβ: + np.maximum(np.add.outer(kpα, -kpβ), 0))

class VerticalCallCalculation(VerticalCalculation, sources={"cα": Securities.Option.Call.Long, "cβ": Securities.Option.Call.Short}):
    τ = equation("tau", np.int16, domain=("cα.τ", "cβ.τ"), function=lambda τcα, τcβ: τcα)
    x = equation("time", np.datetime64, domain=("cα.x", "cβ.x"), function=lambda xcα, xcβ: np.maximum.outer(xcα, xcβ))
    q = equation("size", np.float32, domain=("cα.q", "cβ.q"), function=lambda qcα, qcβ: np.minimum.outer(qcα, qcβ))
    i = equation("interest", np.int32, domain=("cα.i", "cβ.i"), function=lambda icα, icβ: np.minimum.outer(icα, icβ))

    wo = equation("spot", np.float32, domain=("cα.w", "cβ.w"), function=lambda wcα, wcβ: - np.add.outer(wcα, -wcβ))
    vo = equation("current", np.float32, domain=("cα.w", "cβ.w"), function=lambda : )
    vmn = equation("minimum", np.float32, domain=("cα.k", "cβ.k"), function=lambda kcα, kcβ: + np.minimum(np.add.outer(-kcα, kcβ), 0))
    vmx = equation("maximum", np.float32, domain=("cα.k", "cβ.k"), function=lambda kcα, kcβ: + np.maximum(np.add.outer(-kcα, kcβ), 0))

class CollarLongCalculation(CollarCalculation, sources={"pα": Securities.Option.Put.Long, "cβ": Securities.Option.Call.Short, "sα": Securities.Stock.Long}):
    τ = equation("tau", np.int16, domain=("pα.τ", "cβ.τ"), function=lambda τpα, τcβ: τpα)
    x = equation("time", np.datetime64, domain=("pα.x", "cβ.x"), function=lambda xpα, xcβ: np.maximum.outer(xpα, xcβ))
    q = equation("size", np.float32, domain=("pα.q", "cβ.q"), function=lambda qpα, qcβ: np.minimum.outer(qpα, qcβ))
    i = equation("interest", np.int32, domain=("pα.i", "cβ.i"), function=lambda ipα, icβ: np.minimum.outer(ipα, icβ))

    wo = equation("spot", np.float32, domain=("pα.w", "cβ.w"), function=lambda wpα, wcβ, wsα: - np.add.outer(wpα, -wcβ) - wsα)
    vo = equation("current", np.float32, domain=("pα.w", "cβ.w"), function=lambda : )
    vmn = equation("minimum", np.float32, domain=("pα.k", "cβ.k"), function=lambda kpα, kcβ: + np.minimum.outer(kpα, kcβ))
    vmx = equation("maximum", np.float32, domain=("pα.k", "cβ.k"), function=lambda kpα, kcβ: + np.maximum.outer(kpα, kcβ))

class CollarShortCalculation(CollarCalculation, sources={"pβ": Securities.Option.Put.Short, "cα": Securities.Option.Call.Long, "sβ": Securities.Stock.Short}):
    τ = equation("tau", np.int16, domain=("pβ.τ", "cα.τ"), function=lambda τpβ, τcα: τpβ)
    x = equation("time", np.datetime64, domain=("pβ.x", "cα.x"), function=lambda xpβ, xcα: np.maximum.outer(xpβ, xcα))
    q = equation("size", np.float32, domain=("pβ.q", "cα.q"), function=lambda qpβ, qcα: np.minimum.outer(qpβ, qcα))
    i = equation("interest", np.int32, domain=("pβ.i", "cα.i"), function=lambda ipβ, icα: np.minimum.outer(ipβ, icα))

    wo = equation("spot", np.float32, domain=("pβ.w", "cα.w"), function=lambda wpβ, wcα, wsβ: - np.add.outer(-wpβ, wcα) + wsβ)
    vo = equation("current", np.float32, domain=("pβ.w", "cα.w"), function=lambda : )
    vmn = equation("minimum", np.float32, domain=("pβ.k", "cα.k"), function=lambda kpβ, kcα: + np.minimum.outer(-kpβ, -kcα))
    vmx = equation("maximum", np.float32, domain=("pβ.k", "cα.k"), function=lambda kpβ, kcα: + np.maximum.outer(-kpβ, -kcα))

class CondorCalculation(StrategyCalculation, sources={"pα": Securities.Option.Put.Long, "pβ": Securities.Option.Put.Short, "cα": Securities.Option.Call.Long, "cβ": Securities.Option.Call.Short}):
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
    def execute(self, contents, *args, partition=None, **kwargs):
        ticker, expire, datasets = contents
        assert isinstance(datasets, dict)
        assert all([isinstance(security, xr.Dataset) for security in datasets.values()])

        ###


