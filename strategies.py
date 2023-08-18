# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Strategy Objects
@author: Jack Kirby Cook

"""

import numpy as np
import xarray as xr
from enum import IntEnum
from itertools import product
from collections import namedtuple as ntuple

from support.pipelines import Calculator
from support.calculations import Calculation, feed, equation
from support.dispatchers import kwargsdispatcher
from finance.securities import Security, Securities, Positions

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Strategy", "Strategies", "StrategyCalculator", "ValuationCalculator"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


Strategies = IntEnum("Strategy", ["STRANGLE", "COLLAR", "VERTICAL", "CONDOR"], start=1)
class Strategy(ntuple("Strategy", "strategy option position")):
    def __int__(self): return sum([self.strategy * 100, self.security * 10, self.position * 1])
    def __str__(self): return "|".join([str(value.name).lower() for value in self if bool(value)])


class StrategyCalculation(Calculation):
    wpα = feed("put|long", np.float32, axes="(i)", key=Security(Securities.PUT, Positions.LONG), variable="price")
    wpβ = feed("put|short", np.float32, axes="(j)", key=Security(Securities.PUT, Positions.SHORT), variable="price")
    wcα = feed("call|long", np.float32, axes="(k)", key=Security(Securities.CALL, Positions.LONG), variable="price")
    wcβ = feed("call|short", np.float32, axes="(l)", key=Security(Securities.CALL, Positions.SHORT), variable="price")
    wsα = feed("stock|long", np.float32, axes="()", key=Security(Securities.STOCK, Positions.LONG), variable="price")
    wsβ = feed("stock|short", np.float32, axes="()", key=Security(Securities.STOCK, Positions.SHORT), variable="price")

    kpα = feed("put|long", np.float32, axes="(i)", key=Security(Securities.PUT, Positions.LONG), variable="strike")
    kpβ = feed("put|short", np.float32, axes="(j)", key=Security(Securities.PUT, Positions.SHORT), variable="strike")
    kcα = feed("call|long", np.float32, axes="(k)", key=Security(Securities.CALL, Positions.LONG), variable="strike")
    kcβ = feed("call|short", np.float32, axes="(l)", key=Security(Securities.CALL, Positions.SHORT), variable="strike")

    def __init_subclass__(cls, *args, strategy, securities, **kwargs):
        cls.__strategy__ = strategy
        cls.__securities__ = securities

    def __call__(self, securities, *args, **kwargs):
        assert isinstance(securities, dict)
        if not all([security in securities.keys() for security in self.securities]):
            return
        strategies = self.vo(securities).to_dataset(name="spot")
        strategies["value"] = self.vω(securities)
        return strategies

    @property
    def strategy(self): return self.__class__.__strategy__
    @property
    def securities(self): return self.__class__.__securities__


strangle_strategy = Strategy(Strategies.STRANGLE, 0, Positions.LONG)
strangle_securities = [Security(Securities.PUT, Positions.LONG), Security(Securities.CALL, Positions.SHORT)]
class StrangleLongCalculation(StrategyCalculation, strategy=strangle_strategy, securities=strangle_securities):
    vo = equation("spot", np.float32, axes="(i,k)", function=lambda wpα, wcα: - np.add.outer(wpα, wcα))
    vω = equation("val-", np.float32, axes="(i,k)", function=lambda kpα, kcα: + np.maximum(np.add.outer(kpα, -kcα), 0))
    vγ = equation("val+", np.float32, axes="(i,k)", function=lambda kpα, kcα: + np.ones((kpα.shape, kcα.shape)) * np.inf)


collarlong_strategy = Strategy(Strategies.COLLAR, 0, Positions.LONG)
collarlong_securities = [Security(Securities.PUT, Positions.LONG), Security(Securities.CALL, Positions.SHORT)]
class CollarLongCalculation(StrategyCalculation, strategy=collarlong_strategy, securities=collarlong_securities):
    vo = equation("spot", np.float32, axes="(i,l)", function=lambda wpα, wcβ, wsα: - np.add.outer(wpα, -wcβ) - wsα)
    vω = equation("val-", np.float32, axes="(i,l)", function=lambda kpα, kcβ: + np.minimum.outer(kpα, kcβ))
    vγ = equation("val+", np.float32, axes="(i,l)", function=lambda kpα, kcβ: + np.maximum.outer(kpα, kcβ))


collarshort_strategy = Strategy(Strategies.COLLAR, 0, Positions.SHORT)
collarshort_securities = [Security(Securities.PUT, Positions.SHORT), Security(Securities.CALL, Positions.LONG), Security(Securities.STOCK, Positions.LONG)]
class CollarShortCalculation(StrategyCalculation, strategy=collarshort_strategy, securities=collarshort_securities):
    vo = equation("spot", np.float32, axes="(j,k)", function=lambda wpβ, wcα, wsβ: - np.add.outer(-wpβ, wcα) + wsβ)
    vω = equation("val-", np.float32, axes="(j,k)", function=lambda kpβ, kcα: + np.minimum.outer(-kpβ, -kcα))
    vγ = equation("val+", np.float32, axes="(j,k)", function=lambda kpβ, kcα: + np.maximum.outer(-kpβ, -kcα))


verticalput_strategy = Strategy(Strategies.VERTICAL, Securities.PUT, 0)
verticalput_securities = [Security(Securities.PUT, Positions.LONG), Security(Securities.PUT, Positions.SHORT), Security(Securities.STOCK, Positions.SHORT)]
class VerticalPutCalculation(StrategyCalculation, strategy=verticalput_strategy, securities=verticalput_securities):
    vo = equation("spot", np.float32, axes="(i,j)", function=lambda wpα, wpβ: - np.add.outer(wpα, -wpβ))
    vω = equation("val-", np.float32, axes="(i,j)", function=lambda kpα, kpβ: + np.minimum(np.add.outer(kpα, -kpβ), 0))
    vγ = equation("val+", np.float32, axes="(i,j)", function=lambda kpα, kpβ: + np.maximum(np.add.outer(kpα, -kpβ), 0))


verticalcall_strategy = Strategy(Strategies.VERTICAL, Securities.CALL, 0)
verticalcall_securities = [Security(Securities.CALL, Positions.LONG), Security(Securities.CALL, Positions.SHORT)]
class VerticalCallCalculation(StrategyCalculation, strategy=verticalcall_strategy, securities=verticalcall_securities):
    vo = equation("spot", np.float32, axes="(k,l)", function=lambda wcα, wcβ: - np.add.outer(wcα, -wcβ))
    vω = equation("val-", np.float32, axes="(k,l)", function=lambda kcα, kcβ: + np.minimum(np.add.outer(-kcα, kcβ), 0))
    vγ = equation("val+", np.float32, axes="(k,l)", function=lambda kcα, kcβ: + np.maximum(np.add.outer(-kcα, kcβ), 0))


condor_strategy = Strategy(Strategies.CONDOR, 0, 0)
condor_securities = [Security(Securities.PUT, Positions.LONG), Security(Securities.PUT, Positions.SHORT), Security(Securities.CALL, Positions.LONG), Security(Securities.CALL, Positions.SHORT)]
class CondorCalculation(StrategyCalculation, strategy=condor_strategy, securities=condor_securities):
    vo = equation("spot", np.float32, axes="(i,j,k,l)", function=lambda vp, vc: + np.add(vp[:, :, np.newaxis, np.newaxis], vc[np.newaxis, np.newaxis, :, :]))
    vω = equation("val-", np.float32, axes="(i,j,k,l)", function=lambda oω, fo: + np.maximum(oω, fo))
    vγ = equation("val+", np.float32, axes="(i,j,k,l)", function=lambda oγ, fo: + np.minimum(oγ, fo))

    vp = equation("vp", np.float32, axes="(i,j)", function=lambda wpα, wpβ: - np.add.outer(wpα, -wpβ))
    vc = equation("vc", np.float32, axes="(k,l)", function=lambda wcα, wcβ: - np.add.outer(wcα, -wcβ))

    pω = equation("pω", np.float32, axes="(i,j)", function=lambda kpα, kpβ: + np.add.outer(kpα, -kpβ))
    cω = equation("cω", np.float32, axes="(k,l)", function=lambda kcα, kcβ: + np.add.outer(-kcα, kcβ))
    oω = equation("oω", np.float32, axes="(i,j,k,l)", function=lambda pω, cω: + np.minimum(np.minimum(pω[:, :, np.newaxis, np.newaxis], cω[np.newaxis, np.newaxis, :, :]), 0))

    pγ = equation("pγ", np.float32, axes="(i,j)", function=lambda kpα, kpβ: + np.add.outer(kpα, -kpβ))
    cγ = equation("cγ", np.float32, axes="(k,l)", function=lambda kcα, kcβ: + np.add.outer(-kcα, kcβ))
    oγ = equation("oγ", np.float32, axes="(i,j,k,l)", function=lambda pγ, cγ: + np.minimum(np.minimum(pγ[:, :, np.newaxis, np.newaxis], cγ[np.newaxis, np.newaxis, :, :]), 0))

    fα = equation("fα", np.float32, axes="(i,k)", function=lambda kpα, kcα: np.maximum(np.add.outer(kpα, -kcα), 0))
    fβ = equation("fβ", np.float32, axes="(j,l)", function=lambda kpβ, kcβ: np.maximum(np.add.outer(kpβ, -kcβ), 0))
    fo = equation("fo", np.float32, axes="(i,j,k,l)", function=lambda fα, fβ: np.add(fα[:, np.newaxis, :, np.newaxis], -fβ[np.newaxis, :, np.newaxis, :]))


class StrategyCalculator(Calculator, calculations=list(StrategyCalculation.__subclasses__())):
    def execute(self, contents, *args, partition, **kwargs):
        ticker, expire, dataset = contents
        assert isinstance(dataset, xr.Dataset)
        key = lambda security, position: Security(security, position)
        value = lambda security, position: self.parser(dataset, security=security, position=position, partition=partition)
        securities = {key(security, position): value(security, position) for security, position in product(Securities, Positions)}
        for calculation in iter(self.calculations):
            strategies = calculation(securities, *args, **kwargs)
            if strategies is None:
                continue
            yield ticker, expire, calculation.strategy, calculation.securities, strategies

    @kwargsdispatcher("security")
    def parser(self, dataset, *args, security, position, **kwargs): pass

    @parser.register(Securities.STOCK)
    def stock(self, dataset, *args, security, position, **kwargs):
        dataset = dataset.sel({"security": int(security), "position": int(position)})
        dataset = dataset.drop_vars(["security", "position"])
        return dataset

    @parser.register(Securities.PUT, Security.CALL)
    def option(self, dataset, *args, security, position, partition, **kwargs):
        name = str(Security(security, position))
        dataset = dataset.sel({"security": int(security), "position": int(position)})
        dataset = dataset.rename({"strike": name})
        dataset["strike"] = dataset[name]
        dataset = dataset.drop_vars(["security", "position"])
        dataset = dataset.chunk({name: partition}) if bool(partition) else dataset
        return dataset


class ValuationCalculation(Calculation):
    ρ = feed("discount", np.float16, variable="discount")
    to = feed("date", np.datetime64, variable="date")
    tτ = feed("expire", np.datetime64, variable="expire")
    vo = feed("spot", np.float16, variable="spot")
    vτ = feed("value", np.float16, variable="value")

    τau = equation("τau", np.int16, function=lambda tτ, to: np.timedelta64(np.datetime64(tτ, "ns") - np.datetime64(to, "ns"), "D") / np.timedelta64(1, "D"))
    inc = equation("income", np.float32, function=lambda vo, vτ: + np.maximum(vo, 0) + np.maximum(vτ, 0))
    cost = equation("cost", np.float32, function=lambda vo, vτ: - np.minimum(vo, 0) - np.minimum(vτ, 0))
    apy = equation("apy", np.float32, function=lambda r, τau: np.power(r + 1, np.power(τau / 365, -1)) - 1)
    npv = equation("npv", np.float32, function=lambda π, τau, ρ: π * np.power(ρ / 365 + 1, τau))
    π = equation("profit", np.float32, function=lambda inc, cost: inc - cost)
    r = equation("return", np.float32, function=lambda π, cost: π / cost)

    def __call__(self, strategies, *args, discount, **kwargs):
        assert isinstance(strategies, xr.Dataset)
        strategies["tau"] = self.τau(strategies)
        strategies["cost"] = self.cost(strategies)
        strategies["apy"] = self.apy(strategies)
        return strategies


class ValuationCalculator(Calculator, calculations=[ValuationCalculation]):
    def execute(self, contents, *args, **kwargs):
        ticker, expire, strategy, securities, dataset = contents
        assert isinstance(dataset, xr.Dataset)
        for calculation in iter(self.calculations):
            valuations = calculation(dataset, *args, **kwargs)
            yield ticker, expire, strategy, securities, valuations


