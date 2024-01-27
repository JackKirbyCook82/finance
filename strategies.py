# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Strategy Objects
@author: Jack Kirby Cook

"""

import numpy as np
from enum import IntEnum
from collections import namedtuple as ntuple

from support.calculations import Calculation, equation, source, constant
from support.dispatchers import typedispatcher
from support.pipelines import Processor

from finance.securities import Positions, Instruments, Securities

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Strategy", "Strategies", "Calculations", "StrategyCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


Spreads = IntEnum("Strategy", ["STRANGLE", "COLLAR", "VERTICAL"], start=1)
class Strategy(ntuple("Strategy", "spread instrument position")):
    def __new__(cls, spread, instrument, position, *args, **kwargs): return super().__new__(cls, spread, instrument, position)
    def __init__(self, *args, **kwargs): self.__securities = kwargs["securities"]
    def __str__(self): return "|".join([str(value.name).lower() for value in self if bool(value)])
    def __int__(self): return int(self.spread) * 100 + int(self.instrument) * 10 + int(self.position) * 1

    @property
    def title(self): return "|".join([str(string).title() for string in str(self).split("|")])
    @property
    def securities(self): return self.__securities

StrangleLong = Strategy(Spreads.STRANGLE, 0, Positions.LONG, securities=[Securities.Option.Put.Long, Securities.Option.Call.Long])
CollarLong = Strategy(Spreads.COLLAR, 0, Positions.LONG, securities=[Securities.Option.Put.Long, Securities.Option.Call.Short, Securities.Stock.Long])
CollarShort = Strategy(Spreads.COLLAR, 0, Positions.SHORT, securities=[Securities.Option.Call.Long, Securities.Option.Put.Short, Securities.Stock.Short])
VerticalPut = Strategy(Spreads.VERTICAL, Instruments.PUT, 0, securities=[Securities.Option.Put.Long, Securities.Option.Put.Short])
VerticalCall = Strategy(Spreads.VERTICAL, Instruments.CALL, 0, securities=[Securities.Option.Call.Long, Securities.Option.Call.Short])


class StrategiesMeta(type):
    def __iter__(cls): return iter([StrangleLong, CollarLong, CollarShort, VerticalPut, VerticalCall])
    def __getitem__(cls, indexkey): return cls.retrieve(indexkey)

    @typedispatcher
    def retrieve(cls, indexkey): raise TypeError(type(indexkey).__name__)
    @retrieve.register(int)
    def integer(cls, index): return {int(strategy): strategy for strategy in iter(cls)}[index]
    @retrieve.register(str)
    def string(cls, string): return {str(strategy): strategy for strategy in iter(cls)}[str(string).lower()]
    @retrieve.register(tuple)
    def value(cls, value): return {str(strategy): strategy for strategy in iter(cls)}[value]
    @retrieve.register(tuple)
    def value(cls, value): return {(strategy.spread, strategy.instrument, strategy.postion): strategy for strategy in iter(cls)}[value]

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
    sμ = equation("sμ", "underlying", np.float32, domain=("sα.w", "sβ.w"), function=lambda wsα, wsβ: np.add(wsα, wsβ) / 2)
    ε = constant("ε", "fees", position="fees")

    pα = source("pα", str(Securities.Option.Put.Long), position=str(Securities.Option.Put.Long), variables={"τ": "tau", "w": "price", "k": "strike", "x": "size"}, destination=True)
    pβ = source("pβ", str(Securities.Option.Put.Short), position=str(Securities.Option.Put.Short), variables={"τ": "tau", "w": "price", "k": "strike", "x": "size"}, destination=True)
    cα = source("cα", str(Securities.Option.Call.Long), position=str(Securities.Option.Call.Long), variables={"τ": "tau", "w": "price", "k": "strike", "x": "size"}, destination=True)
    cβ = source("cβ", str(Securities.Option.Call.Short), position=str(Securities.Option.Call.Short), variables={"τ": "tau", "w": "price", "k": "strike", "x": "size"}, destination=True)
    sα = source("sα", str(Securities.Stock.Long), position=str(Securities.Stock.Long), variables={"w": "price", "x": "size"}, destination=True)
    sβ = source("sβ", str(Securities.Stock.Short), position=str(Securities.Stock.Short), variables={"w": "price", "x": "size"}, destination=True)

    def execute(self, feeds, *args, fees, **kwargs):
        feeds = {str(security): dataset for security, dataset in feeds.items()}
        yield self.τ(**feeds, fees=fees)
        yield self.x(**feeds, fees=fees)
        yield self.wo(**feeds, fees=fees)
        yield self.wτn(**feeds, fees=fees)
        yield self.wτx(**feeds, fees=fees)
        yield self.wτo(**feeds, fees=fees)
        yield self.sμ(**feeds)

class StrangleCalculation(StrategyCalculation): pass
class VerticalCalculation(StrategyCalculation): pass
class CollarCalculation(StrategyCalculation): pass

class StrangleLongCalculation(StrangleCalculation):
    τ = equation("τ", "tau", np.int16, domain=("pα.τ", "cα.τ"), function=lambda τpα, τcα: τpα)
    x = equation("x", "size", np.int64, domain=("pα.x", "cα.x"), function=lambda xpα, xcα: np.minimum(xpα, xcα))
    wo = equation("wo", "spot", np.float32, domain=("pα.w", "cα.w", "ε"), function=lambda wpα, wcα, ε: - (wpα + wcα) * 100 - ε)
    wτn = equation("wτn", "minimum", np.float32, domain=("pα.k", "cα.k", "ε"), function=lambda kpα, kcα, ε: np.maximum(kpα - kcα, 0) * 100 - ε)
    wτx = equation("wτx", "maximum", np.float32, domain=("pα.k", "cα.k", "ε"), function=lambda kpα, kcα, ε: np.Inf * 100 - ε)
    wτo = equation("wτo", "current", np.float32, domain=("pα.k", "cα.k", "sμ", "ε"), function=lambda kpα, kcα, sμ, ε: (np.maximum(kpα - sμ, 0) + np.maximum(sμ - kcα, 0)) * 100 - ε)

class VerticalPutCalculation(VerticalCalculation):
    τ = equation("τ", "tau", np.int16, domain=("pα.τ", "pβ.τ"), function=lambda τpα, τpβ: τpα)
    x = equation("x", "size", np.int64, domain=("pα.x", "pβ.x"), function=lambda xpα, xpβ: np.minimum(xpα, xpβ))
    wo = equation("wo", "spot", np.float32, domain=("pα.w", "pβ.w", "ε"), function=lambda wpα, wpβ, ε: (wpβ - wpα) * 100 - ε)
    wτn = equation("wτn", "minimum", np.float32, domain=("pα.k", "pβ.k", "ε"), function=lambda kpα, kpβ, ε: np.minimum(kpα - kpβ, 0) * 100 - ε)
    wτx = equation("wτx", "maximum", np.float32, domain=("pα.k", "pβ.k", "ε"), function=lambda kpα, kpβ, ε: np.maximum(kpα - kpβ, 0) * 100 - ε)
    wτo = equation("wτo", "current", np.float32, domain=("pα.k", "pβ.k", "sμ", "ε"), function=lambda kpα, kpβ, sμ, ε: (np.maximum(kpα - sμ, 0) - np.maximum(kpβ - sμ, 0)) * 100 - ε)

class VerticalCallCalculation(VerticalCalculation):
    τ = equation("τ", "tau", np.int16, domain=("cα.τ", "cβ.τ"), function=lambda τcα, τcβ: τcα)
    x = equation("x", "size", np.int64, domain=("cα.x", "cβ.x"), function=lambda xcα, xcβ: np.minimum(xcα, xcβ))
    wo = equation("wo", "spot", np.float32, domain=("cα.w", "cβ.w", "ε"), function=lambda wcα, wcβ, ε: (wcβ - wcα) * 100 - ε)
    wτn = equation("wτn", "minimum", np.float32, domain=("cα.k", "cβ.k", "ε"), function=lambda kcα, kcβ, ε: np.minimum(kcβ - kcα, 0) * 100 - ε)
    wτx = equation("wτx", "maximum", np.float32, domain=("cα.k", "cβ.k", "ε"), function=lambda kcα, kcβ, ε: np.maximum(kcβ - kcα, 0) * 100 - ε)
    wτo = equation("wτo", "current", np.float32, domain=("cα.k", "cβ.k", "sμ", "ε"), function=lambda kcα, kcβ, sμ, ε: (np.maximum(sμ - kcα, 0) - np.maximum(sμ - kcβ, 0)) * 100 - ε)

class CollarLongCalculation(CollarCalculation):
    τ = equation("τ", "tau", np.int16, domain=("pα.τ", "cβ.τ"), function=lambda τpα, τcβ: τpα)
    x = equation("x", "size", np.int64, domain=("pα.x", "cβ.x"), function=lambda xpα, xcβ: np.minimum(xpα, xcβ))
    wo = equation("wo", "spot", np.float32, domain=("pα.w", "cβ.w", "sα.w", "ε"), function=lambda wpα, wcβ, wsα, ε: (wcβ - wpα - wsα) * 100 - ε)
    wτn = equation("wτn", "minimum", np.float32, domain=("pα.k", "cβ.k", "ε"), function=lambda kpα, kcβ, ε: np.minimum(kpα, kcβ) * 100 - ε)
    wτx = equation("wτx", "maximum", np.float32, domain=("pα.k", "cβ.k", "ε"), function=lambda kpα, kcβ, ε: np.maximum(kpα, kcβ) * 100 - ε)
    wτo = equation("wτo", "current", np.float32, domain=("pα.k", "cβ.k", "sμ", "ε"), function=lambda kpα, kcβ, sμ, ε: (np.maximum(kpα - sμ, 0) - np.maximum(sμ - kcβ, 0) + sμ) * 100 - ε)

class CollarShortCalculation(CollarCalculation):
    τ = equation("τ", "tau", np.int16, domain=("pβ.τ", "cα.τ"), function=lambda τpβ, τcα: τpβ)
    x = equation("x", "size", np.int64, domain=("pβ.x", "cα.x"), function=lambda xpβ, xcα: np.minimum(xpβ, xcα))
    wo = equation("wo", "spot", np.float32, domain=("pβ.w", "cα.w", "sβ.w", "ε"), function=lambda wpβ, wcα, wsβ, ε: (wpβ - wcα + wsβ) * 100 - ε)
    wτn = equation("wτn", "minimum", np.float32, domain=("pβ.k", "cα.k", "ε"), function=lambda kpβ, kcα, ε: np.minimum(-kpβ, -kcα) * 100 - ε)
    wτx = equation("wτx", "maximum", np.float32, domain=("pβ.k", "cα.k", "ε"), function=lambda kpβ, kcα, ε: np.maximum(-kpβ, -kcα) * 100 - ε)
    wτo = equation("wτo", "current", np.float32, domain=("pβ.k", "cα.k", "sμ", "ε"), function=lambda kpβ, kcα, sμ, ε: (np.maximum(sμ - kcα, 0) - np.maximum(kpβ - sμ, 0) - sμ) * 100 - ε)


class CalculationsMeta(type):
    def __iter__(cls):
        contents = {VerticalPut: VerticalPutCalculation, VerticalCall: VerticalCallCalculation}
        contents.update({CollarLong: CollarLongCalculation, CollarShort: CollarShortCalculation})
        contents.update({StrangleLong: StrangleLongCalculation})
        return ((key, value) for key, value in contents.items())

    class Strangle:
        Long = StrangleLongCalculation
    class Collar:
        Long = CollarLongCalculation
        Short = CollarShortCalculation
    class Vertical:
        Put = VerticalPutCalculation
        Call = VerticalCallCalculation

    @property
    def Strangles(cls): return iter({StrangleLong: StrangleLongCalculation}.items())
    @property
    def Collars(cls): return iter({CollarLong: CollarLongCalculation, CollarShort: CollarShortCalculation}.items())
    @property
    def Verticals(cls): return iter({VerticalPut: VerticalPutCalculation, VerticalCall: VerticalCallCalculation}.items())

class Calculations(object, metaclass=CalculationsMeta):
    pass


class StrategyQuery(ntuple("Query", "current ticker expire strategies")):
    def __str__(self): return f"{self.ticker}|{self.expire.strftime('%Y-%m-%d')}"


class StrategyCalculator(Processor):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        calculations = {strategy: calculation(*args, **kwargs) for (strategy, calculation) in iter(Calculations)}
        self.calculations = calculations

    def execute(self, query, *args, **kwargs):
        stocks = {security: dataset for security, dataset in query.stocks.items() if dataset["price"].size > 0}
        options = {security: dataset for security, dataset in query.options.items() if dataset["price"].size > 0}
        if not bool(stocks) or not bool(options) or len(stocks) != 2:
            return
        stocks = {security: self.stocks(dataset, *args, security=security, **kwargs) for security, dataset in stocks.items()}
        options = {security: self.options(dataset, *args, security=security, **kwargs) for security, dataset in options.items()}
        function = lambda strategy: all([security in list(stocks.keys()) + list(options.keys()) for security in strategy.securities])
        calculations = {strategy: calculation for strategy, calculation in self.calculations.items() if function(strategy)}
        strategies = {strategy: calculation(stocks | options, *args, **kwargs) for strategy, calculation in calculations.items()}
        if not bool(strategies):
            return
        yield StrategyQuery(query.current, query.ticker, query.expire, strategies)

    @staticmethod
    def stocks(dataset, *args, **kwargs): return dataset
    @staticmethod
    def options(dataset, *args, security, **kwargs):
        dataset = dataset.rename({"strike": str(security)})
        dataset["strike"] = dataset[str(security)].expand_dims(["ticker", "date", "expire"])
        return dataset




