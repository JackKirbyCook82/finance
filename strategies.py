# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Strategy Objects
@author: Jack Kirby Cook

"""

import numpy as np
from enum import IntEnum
from collections import namedtuple as ntuple
from collections import OrderedDict as ODict

from support.pipelines import Calculator
from support.calculations import Calculation, equation, source, constant
from support.dispatchers import kwargsdispatcher, typedispatcher

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
    pα = source("pα", str(Securities.Option.Put.Long), position=str(Securities.Option.Put.Long), variables={"τ": "tau", "w": "price", "k": "strike", "x": "size"}, destination=True)
    pβ = source("pβ", str(Securities.Option.Put.Short), position=str(Securities.Option.Put.Short), variables={"τ": "tau", "w": "price", "k": "strike", "x": "size"}, destination=True)
    cα = source("cα", str(Securities.Option.Call.Long), position=str(Securities.Option.Call.Long), variables={"τ": "tau", "w": "price", "k": "strike", "x": "size"}, destination=True)
    cβ = source("cβ", str(Securities.Option.Call.Short), position=str(Securities.Option.Call.Short), variables={"τ": "tau", "w": "price", "k": "strike", "x": "size"}, destination=True)
    sα = source("sα", str(Securities.Stock.Long), position=str(Securities.Stock.Long), variables={"w": "price", "x": "size"}, destination=True)
    sβ = source("sβ", str(Securities.Stock.Short), position=str(Securities.Stock.Short), variables={"w": "price", "x": "size"}, destination=True)
    sμ = equation("sμ", "underlying", np.float32, domain=("sα.w", "sβ.w"), function=lambda wsα, wsβ: np.add(wsα, wsβ) / 2)
    ε = constant("ε", "fees", position="fees")

    def execute(self, feeds, *args, fees, **kwargs):
        feeds = {str(security): dataset for security, dataset in feeds.items()}
        yield self.τ(**feeds, fees=fees)
        yield self.x(**feeds, fees=fees)
        yield self.wo(**feeds, fees=fees)
        yield self.wτ(**feeds, fees=fees)
        yield self.sμ(**feeds)

class StrangleCalculation(StrategyCalculation): pass
class VerticalCalculation(StrategyCalculation): pass

class VerticalPutCalculation(VerticalCalculation):
    τ = equation("τ", "tau", np.int16, domain=("pα.τ", "pβ.τ"), function=lambda τpα, τpβ: τpα)
    x = equation("x", "size", np.float32, domain=("pα.x", "pβ.x"), function=lambda xpα, xpβ: np.minimum(xpα, xpβ))
    wo = equation("wo", "spot", np.float32, domain=("pα.w", "pβ.w", "ε"), function=lambda wpα, wpβ, ε: (wpβ - wpα) * 100 - ε)
    wτ = equation("wτ", "future", np.float32, domain=("pα.k", "pβ.k", "ε"), function=lambda kpα, kpβ, ε: np.minimum(kpα - kpβ, 0) * 100 - ε)

class VerticalCallCalculation(VerticalCalculation):
    τ = equation("τ", "tau", np.int16, domain=("cα.τ", "cβ.τ"), function=lambda τcα, τcβ: τcα)
    x = equation("x", "size", np.float32, domain=("cα.x", "cβ.x"), function=lambda xcα, xcβ: np.minimum(xcα, xcβ))
    wo = equation("wo", "spot", np.float32, domain=("cα.w", "cβ.w", "ε"), function=lambda wcα, wcβ, ε: (wcβ - wcα) * 100 - ε)
    wτ = equation("wτ", "future", np.float32, domain=("cα.k", "cβ.k", "ε"), function=lambda kcα, kcβ, ε: + np.minimum(kcβ - kcα, 0) * 100 - ε)


class CalculationsMeta(type):
    def __iter__(cls):
        contents = {Strategies.Vertical.Put: VerticalPutCalculation, Strategies.Vertical.Call: VerticalCallCalculation}
        return ((key, value) for key, value in contents.items())

    class Strangle:
        pass
    class Collar:
        pass
    class Vertical:
        Put = VerticalPutCalculation
        Call = VerticalCallCalculation

class Calculations(object, metaclass=CalculationsMeta):
    pass


class StrategyQuery(ntuple("Query", "current ticker expire strategies")): pass
class StrategyCalculator(Calculator, calculations=ODict(list(Calculations))):
    def execute(self, query, *args, **kwargs):
        securities = {security: dataset for security, dataset in query.securities.items()}
        if not bool(securities):
            return
        parser = lambda security, dataset: self.parser(dataset, *args, security=security, **kwargs)
        function = lambda strategy: all([security in securities.keys() for security in strategy.securities])
        calculations = {strategy: calculation for strategy, calculation in self.calculations.items() if function(strategy)}
        securities = {security: parser(security, dataset) for security, dataset in securities.items()}
        strategies = {strategy: calculation(securities, *args, **kwargs) for strategy, calculation in calculations.items()}
        if not bool(strategies):
            return
        yield StrategyQuery(query.current, query.ticker, query.expire, strategies)

    @kwargsdispatcher("security")
    def parser(self, dataset, *args, security, **kwargs): raise ValueError(str(security))

    @parser.register.value(*list(Securities.stocks))
    def stock(self, dataset, *args, **kwargs): return dataset

    @parser.register.value(*list(Securities.options))
    def option(self, dataset, *args, security, **kwargs):
        dataset = dataset.rename({"strike": str(security)})
        dataset["strike"] = dataset[str(security)].expand_dims(["ticker", "date", "expire"])
        return dataset



