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
from collections import OrderedDict as ODict

from support.pipelines import Calculator
from support.calculations import Calculation, equation, source, constant
from support.dispatchers import typedispatcher

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
    def __int__(self): return int(self.spread) * 100 + int(self.instrument) * 10 + int(self.position) * 1

    @property
    def securities(self): return self.__securities

StrangleLong = Strategy(Spreads.STRANGLE, 0, Positions.LONG, securities=[Securities.Option.Put.Long, Securities.Option.Call.Long])
CollarLong = Strategy(Spreads.COLLAR, 0, Positions.LONG, securities=[Securities.Option.Put.Long, Securities.Option.Call.Short, Securities.Stock.Long])
CollarShort = Strategy(Spreads.COLLAR, 0, Positions.SHORT, securities=[Securities.Option.Call.Long, Securities.Option.Put.Short, Securities.Stock.Short])
VerticalPut = Strategy(Spreads.VERTICAL, Instruments.PUT, 0, securities=[Securities.Option.Put.Long, Securities.Option.Put.Short])
VerticalCall = Strategy(Spreads.VERTICAL, Instruments.CALL, 0, securities=[Securities.Option.Call.Long, Securities.Option.Call.Short])
Condor = Strategy(Spreads.CONDOR, 0, 0, securities=[Securities.Option.Put.Long, Securities.Option.Call.Long, Securities.Option.Put.Short, Securities.Option.Call.Short])

class StrategiesMeta(type):
    def __iter__(cls): return iter([StrangleLong, CollarLong, CollarShort, VerticalPut, VerticalCall, Condor])
    def __getitem__(cls, indexkey): return cls.retrieve(indexkey)

    @typedispatcher
    def retrieve(cls, indexkey): raise TypeError(type(indexkey).__name__)
    @retrieve.register(int)
    def integer(cls, index): return {int(strategy): strategy for strategy in iter(cls)}[index]
    @retrieve.register(str)
    def string(cls, string): return {int(strategy): strategy for strategy in iter(cls)}[str(string).lower()]

    Condor = Condor
    class Strangle:
        Long = StrangleLong
    class Collar:
        Long = CollarLong
        Short = CollarShort
    class Vertical:
        Put = VerticalPut
        Call = VerticalCall

class Strategies(object, metaclass=StrategiesMeta):
    pass


class StrategyCalculation(Calculation):
    pα = source("pα", str(Securities.Option.Put.Long), position=str(Securities.Option.Put.Long), variables={"τ": "tau", "w": "price", "k": "strike", "q": "size", "i": "interest"})
    pβ = source("pβ", str(Securities.Option.Put.Short), position=str(Securities.Option.Put.Short), variables={"τ": "tau", "w": "price", "k": "strike", "q": "size", "i": "interest"})
    cα = source("cα", str(Securities.Option.Call.Long), position=str(Securities.Option.Call.Long), variables={"τ": "tau", "w": "price", "k": "strike", "q": "size", "i": "interest"})
    cβ = source("cβ", str(Securities.Option.Call.Short), position=str(Securities.Option.Call.Short), variables={"τ": "tau", "w": "price", "k": "strike", "q": "size", "i": "interest"})
    sα = source("sα", str(Securities.Stock.Long), position=str(Securities.Stock.Long), variables={"w": "price", "q": "size"})
    sβ = source("sβ", str(Securities.Stock.Short), position=str(Securities.Stock.Short), variables={"w": "price", "q": "size"})
    ε = constant("ε", "fees", position="fees")

    def execute(self, dataset, *args, **kwargs):
        yield self.τ(*args, **kwargs)
        yield self.wo(*args, **kwargs)
        yield self.vmn(*args, **kwargs)
        yield self.vmx(*args, **kwargs)

class StrangleCalculation(StrategyCalculation): pass
class VerticalCalculation(StrategyCalculation): pass
class CollarCalculation(StrategyCalculation): pass

class StrangleLongCalculation(StrangleCalculation):
    τ = equation("τ", "tau", np.int16, domain=("pα.τ", "cα.τ"), function=lambda τpα, τcα: τpα)
    q = equation("q", "size", np.float32, domain=("pα.q", "cα.q"), function=lambda qpα, qcα: np.minimum.outer(qpα, qcα))
    i = equation("i", "interest", np.int32, domain=("pα.i", "cα.i"), function=lambda ipα, icα: np.minimum.outer(ipα, icα))

    wo = equation("wo", "spot", np.float32, domain=("pα.w", "cα.w", "ε"), function=lambda wpα, wcα, ε: - np.add.outer(wpα, wcα) * 100 - ε)
    vmn = equation("vmn", "minimum", np.float32, domain=("pα.k", "cα.k", "ε"), function=lambda kpα, kcα, ε: + np.maximum(np.add.outer(kpα, -kcα), 0) * 100 - ε)
    vmx = equation("vmx", "maximum", np.float32, domain=("pα.k", "cα.k", "ε"), function=lambda kpα, kcα, ε: + np.ones((kpα.shape, kcα.shape)) * np.inf * 100 - ε)

class VerticalPutCalculation(VerticalCalculation):
    τ = equation("τ", "tau", np.int16, domain=("pα.τ", "pβ.τ"), function=lambda τpα, τpβ: τpα)
    q = equation("q", "size", np.float32, domain=("pα.q", "pβ.q"), function=lambda qpα, qpβ: np.minimum.outer(qpα, qpβ))
    i = equation("i", "interest", np.int32, domain=("pα.i", "pβ.i"), function=lambda ipα, ipβ: np.minimum.outer(ipα, ipβ))

    wo = equation("wo", "spot", np.float32, domain=("pα.w", "pβ.w", "ε"), function=lambda wpα, wpβ, ε: - np.add.outer(wpα, -wpβ) * 100 - ε)
    vmn = equation("vmn", "minimum", np.float32, domain=("pα.k", "pβ.k", "ε"), function=lambda kpα, kpβ, ε: + np.minimum(np.add.outer(kpα, -kpβ), 0) * 100 - ε)
    vmx = equation("vmx", "maximum", np.float32, domain=("pα.k", "pβ.k", "ε"), function=lambda kpα, kpβ, ε: + np.maximum(np.add.outer(kpα, -kpβ), 0) * 100 - ε)

class VerticalCallCalculation(VerticalCalculation):
    τ = equation("τ", "tau", np.int16, domain=("cα.τ", "cβ.τ"), function=lambda τcα, τcβ: τcα)
    q = equation("q", "size", np.float32, domain=("cα.q", "cβ.q"), function=lambda qcα, qcβ: np.minimum.outer(qcα, qcβ))
    i = equation("i", "interest", np.int32, domain=("cα.i", "cβ.i"), function=lambda icα, icβ: np.minimum.outer(icα, icβ))

    wo = equation("wo", "spot", np.float32, domain=("cα.w", "cβ.w", "ε"), function=lambda wcα, wcβ, ε: - np.add.outer(wcα, -wcβ) * 100 - ε)
    vmn = equation("vmn", "minimum", np.float32, domain=("cα.k", "cβ.k", "ε"), function=lambda kcα, kcβ, ε: + np.minimum(np.add.outer(-kcα, kcβ), 0) * 100 - ε)
    vmx = equation("vmx", "maximum", np.float32, domain=("cα.k", "cβ.k", "ε"), function=lambda kcα, kcβ, ε: + np.maximum(np.add.outer(-kcα, kcβ), 0) * 100 - ε)

class CollarLongCalculation(CollarCalculation):
    τ = equation("τ", "tau", np.int16, domain=("pα.τ", "cβ.τ"), function=lambda τpα, τcβ: τpα)
    q = equation("q", "size", np.float32, domain=("pα.q", "cβ.q"), function=lambda qpα, qcβ: np.minimum.outer(qpα, qcβ))
    i = equation("i", "interest", np.int32, domain=("pα.i", "cβ.i"), function=lambda ipα, icβ: np.minimum.outer(ipα, icβ))

    wo = equation("wo", "spot", np.float32, domain=("pα.w", "cβ.w", "ε"), function=lambda wpα, wcβ, wsα, ε: (-np.add.outer(wpα, -wcβ) - wsα) * 100 - ε)
    vmn = equation("vmn", "minimum", np.float32, domain=("pα.k", "cβ.k", "ε"), function=lambda kpα, kcβ, ε: + np.minimum.outer(kpα, kcβ) * 100 - ε)
    vmx = equation("vmx", "maximum", np.float32, domain=("pα.k", "cβ.k", "ε"), function=lambda kpα, kcβ, ε: + np.maximum.outer(kpα, kcβ) * 100 - ε)

class CollarShortCalculation(CollarCalculation):
    τ = equation("τ", "tau", np.int16, domain=("pβ.τ", "cα.τ"), function=lambda τpβ, τcα: τpβ)
    q = equation("q", "size", np.float32, domain=("pβ.q", "cα.q"), function=lambda qpβ, qcα: np.minimum.outer(qpβ, qcα))
    i = equation("i", "interest", np.int32, domain=("pβ.i", "cα.i"), function=lambda ipβ, icα: np.minimum.outer(ipβ, icα))

    wo = equation("wo", "spot", np.float32, domain=("pβ.w", "cα.w", "ε"), function=lambda wpβ, wcα, wsβ, ε: (-np.add.outer(-wpβ, wcα) + wsβ) * 100 - ε)
    vmn = equation("vmn", "minimum", np.float32, domain=("pβ.k", "cα.k", "ε"), function=lambda kpβ, kcα, ε: + np.minimum.outer(-kpβ, -kcα) * 100 - ε)
    vmx = equation("vmx", "maximum", np.float32, domain=("pβ.k", "cα.k", "ε"), function=lambda kpβ, kcα, ε: + np.maximum.outer(-kpβ, -kcα) * 100 - ε)


class CalculationsMeta(type):
    def __iter__(cls):
        contents = {Strategies.Strangle.Long: StrangleLongCalculation}
        contents.update({Strategies.Vertical.Put: VerticalPutCalculation, Strategies.Vertical.Call: VerticalCallCalculation})
        contents.update({Strategies.Collar.Long: CollarLongCalculation, Strategies.Collar.Short: CollarShortCalculation})
        return ((key, value) for key, value in contents.items())

    class Strangle:
        Long = StrangleLongCalculation
    class Collar:
        Long = CollarLongCalculation
        Short = CollarShortCalculation
    class Vertical:
        Put = VerticalPutCalculation
        Call = VerticalCallCalculation

class Calculations(object, metaclass=CalculationsMeta):
    pass


class StrategyCalculator(Calculator, calculations=ODict(list(iter(Calculations)))):
    def execute(self, contents, *args, **kwargs):
        current, ticker, expire, datasets = contents
        assert isinstance(datasets, dict)
        assert all([isinstance(security, xr.Dataset) for security in datasets.values()])
        feeds = {str(security): dataset for security, dataset in datasets.items()}
        for strategy, calculation in self.calculations.items():
            results = calculation(*args, **feeds, **kwargs)
            yield current, ticker, expire, strategy, results




