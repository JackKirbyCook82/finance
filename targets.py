# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Target Objects
@author: Jack Kirby Cook

"""

import os.path
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import dask.dataframe as dk
from abc import ABC, abstractmethod
from functools import total_ordering
from datetime import datetime as Datetime
from collections import namedtuple as ntuple

from support.meta import SubclassMeta
from support.pipelines import Processor, Saver, Loader
from support.visualize import Figure, Axes, Coordinate, Plot

from finance.strategies import Strategies

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TargetSaver", "TargetLoader", "TargetCalculator", "TargetAnalysis"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


@total_ordering
class TargetValuation(ntuple("Valuation", "spot future tau phi")):
    def __new__(cls, *args, spot, future, tau, datetime, lifetime=None, **kwargs):
        phi = (datetime + lifetime) if bool(lifetime) else None
        return super().__new__(cls, spot, future, tau, phi)

    def __bool__(self): return (self.phi <= self.chi) if self.phi is not None else True
    def __eq__(self, other): return self.apy == other.apy
    def __lt__(self, other): return self.apy < other.apy

    def __str__(self):
        contents = dict(apy=self.apy, tau=self.tau, income=self.income, cost=self.cost)
        string = "[{apy:.2f}%, ${income:.2f}|${cost:.2f}, ☀{tau:.0f}]".format(**contents)
        return string

    @property
    def apy(self): return np.power(self.gains + 1, np.power(self.tau / 365, -1)) - 1
    @property
    def income(self): return + np.maximum(self.spot, 0) + np.maximum(self.future, 0)
    @property
    def cost(self): return - np.minimum(self.spot, 0) - np.minimum(self.future, 0)
    @property
    def gains(self): return self.profit / self.cost
    @property
    def profit(self): return self.income - self.cost
    @property
    def chi(self): return Datetime.now()


class TargetSecurity(ntuple("Security", "instrument position"), ABC, metaclass=SubclassMeta):
    def __new__(cls, security, *args, **kwargs):
        assert cls.__formatting__ is not None
        if cls is TargetSecurity:
            subcls = cls[str(security.instrument)]
            return subcls(security, *args, **kwargs)
        return super().__new__(security.instrument, security.position)

    def __init__(self, security, *args, **kwargs):
        self.__function = security.payoff

    @abstractmethod
    def payoff(self, domain): pass
    @property
    def function(self): return self.__function


class TargetStock(TargetSecurity, key="stock"):
    def __init__(self, *args, ticker, **kwargs):
        super().__init__(*args, **kwargs)
        self.__ticker = ticker

    def __str__(self):
        security = "|".join([str(content.name).lower() for content in self]).title()
        contents = dict(ticker=self.ticker)
        string = "{}[{ticker}]".format(security, **contents)
        return string

    def payoff(self, domain): return self.function(domain)
    def contents(self): return dict(ticker=self.ticker)

    @property
    def ticker(self): return self.__ticker


class TargetOption(TargetSecurity, keys=["put", "call"]):
    def __init__(self, security, *args, ticker, expire, **kwargs):
        super().__init__(*args, **kwargs)
        self.__strike = kwargs[str(security)]
        self.__ticker = ticker
        self.__expire = expire

    def __str__(self):
        security = "|".join([str(content.name).lower() for content in self]).title()
        contents = dict(ticker=self.ticker, strike=self.strike, expire=self.expire)
        string = "{}[{ticker}, ${strike:.2f}, {expire}]".format(security, **contents)
        return string

    def payoff(self, domain): return self.function(domain, self.strike)
    def contents(self): return dict(ticker=self.ticker, strike=self.strike, expire=self.expire)

    @property
    def ticker(self): return self.__ticker
    @property
    def strike(self): return self.__strike
    @property
    def expire(self): return self.__expire


@total_ordering
class TargetStrategy(ntuple("Strategy", "spread security instrument")):
    def __new__(cls, strategy, *args, **kwargs): return super().__new__(cls, *strategy)
    def __init__(self, *args, securities, valuation, **kwargs):
        self.__securities = securities
        self.__valuation = valuation

    def __bool__(self): return bool(self.valuation)
    def __eq__(self, other): return self.valuation.apy == other.valuation.apy
    def __lt__(self, other): return self.valuation.apy < other.valuation.apy

    def __str__(self):
        strategy = "|".join([str(content.name).lower() for content in self]).title()
        securities = [str(security) for security in self.securities]
        valuation = str(self.valuation)
        return "\n".join([strategy + valuation, *securities])

    def ceiling(self, domain): return np.max(self.payoff(domain))
    def floor(self, domain): return np.min(self.payoff(domain))
    def payoff(self, domain):
        payoffs = np.array([security.payoff(domain) for security in self.securities])
        return np.sum(payoffs, axis=0)

    @property
    def securities(self): return self.__securities
    @property
    def valuation(self): return self.__valuation


class TargetVariable(ntuple("Variable", "variable name size scale string")):
    def __call__(self, values):
        for lower, upper in zip(values[:-1], values[1:]):
            yield self.string.format(lower * self.scale, upper * self.scale)


class TargetSaver(Saver):
    def execute(self, contents, *args, **kwargs):
        ticker, expire, strategy, dataframe = contents
        assert isinstance(dataframe, (pd.DataFrame, dk.DataFrame))
        folder = os.path.join(self.repository, str(ticker), str(expire.strftime("%Y%m%d")))
        if not os.path.isdir(folder):
            os.mkdir(folder)
        file = str(strategy) + ".csv"
        self.write(dataframe, file=file, mode="a")


class TargetLoader(Loader):
    def execute(self, ticker, *args, **kwargs):
        folder = os.path.join(self.repository, str(ticker))
        for foldername in os.listdir(folder):
            expire = Datetime.strptime(os.path.splitext(foldername)[0], "%Y%m%d").date()
            strategy, valuations = self.strategies(ticker, expire)
            yield ticker, expire, strategy, valuations

    def strategies(self, ticker, expire):
        datatypes = {}
        folder = os.path.join(self.repository, str(ticker), str(expire.strftime("%Y%m%d")))
        for filename in os.listdir(folder):
            strategy = Strategies[str(filename).split(".")[0]]
            file = os.path.join(folder, filename)
            dataframe = self.refer(file=file, datatypes=datatypes, datetypes=[])
            return strategy, dataframe


class TargetProcessor(Processor, ABC):
    @staticmethod
    def parser(dataframe, *args, apy=None, funds=None, size=None, interest=None, volume=None, **kwargs):
        dataframe = dataframe.where(dataframe["apy"] >= float(apy)) if apy is not None else dataframe
        dataframe = dataframe.where(dataframe["cost"] <= float(funds)) if funds is not None else dataframe
        dataframe = dataframe.where(dataframe["size"] >= size) if bool(size) else dataframe
        dataframe = dataframe.where(dataframe["interest"] >= interest) if bool(interest) else dataframe
        dataframe = dataframe.where(dataframe["volume"] >= volume) if bool(volume) else dataframe
        dataframe = dataframe.dropna(how="all")
        return dataframe

    @staticmethod
    def partitions(dataframe):
        if not hasattr(dataframe, "npartitions"):
            yield dataframe
            return
        for index in dataframe.npartitions:
            partition = dataframe.get_partition(index).compute()
            yield partition


class TargetCalculator(TargetProcessor):
    def execute(self, contents, *args, **kwargs):
        pctdiff = lambda value, other: (value - other) / other
        ticker, expire, strategy, dataframe = contents
        assert isinstance(dataframe, xr.Dataset)
        dataframe = dataframe.to_dask_dataframe() if bool(dataframe.chunks) else dataframe.to_dataframe()
        dataframe = self.parser(dataframe, *args, **kwargs)
        for partition in self.partitions(dataframe):
            partition = partition.sort_values("apy", axis=1, ascending=False, ignore_index=True, inplace=False)
            for index, record in enumerate(partition.to_dict("records")):
                valuation = TargetValuation(*args, **record, **kwargs)
                securities = [TargetSecurity(security, *args, **record, **kwargs) for security in strategy.securities]
                strategy = TargetStrategy(strategy, *args, valuation=valuation, securities=securities, **record, **kwargs)
                assert pctdiff(record["cost"], valuation.cost) <= 0.01
                assert pctdiff(record["apy"], valuation.apy) <= 0.01
                if not bool(strategy.valuation):
                    continue
                yield strategy


class TargetAnalysis(TargetProcessor):
    def __init__(self, *args, size=25, **kwargs):
        super().__init__(*args, **kwargs)
        self.__table = None

    def execute(self, contents, *args, **kwargs):
        ticker, expire, strategy, dataframe = contents
        assert isinstance(dataframe, xr.Dataset)
        dataframe = dataframe.to_dask_dataframe() if bool(dataframe.chunks) else dataframe.to_dataframe()
        dataframe = self.parser(dataframe, *args, **kwargs)
        self.table = dk.concat(list(filter(None, [self.table, dataframe])))

    def visualize(self, *args, variables, size=25, formats={}, scales={}, **kwargs):
        assert isinstance(variables, dict) and isinstance(size, int)
        variables = [TargetVariable(variable, name, size, formats.get(variable, "{:.0f}|{:.0f}"), scales.get(variable, 1)) for variable, name in variables.items()]
        data = self.table[[column.name for column in variables]].to_dask_array()
        return self.figure(variables, data)

    @staticmethod
    def figure(variables, data):
        histogram, edges = da.histogramdd(data, bins=[column.size for column in variables.values()])
        histogram = histogram.compute() if hasattr(histogram, "partitions") else histogram
        edges = edges.compute() if hasattr(edges, "partitions") else edges
        grid = np.meshgrid(*[np.arange(column.size) for column in variables.values()])
        figure = Figure(size=(8, 8), layout=(1, 1), name=None)
        axes = Axes.Axes3D(name=None)
        plot = Plot.Scatter3D(name=None)
        for (variable, column), values in zip(variables.items(), edges):
            ticks = np.arange(column.size)
            labels = list(column(values))
            coordinate = Coordinate(variable, column.name, ticks, labels, 45)
            setattr(axes, column.variable, coordinate)
#        plot.sizes = histogram.ravel()
#        for column, values in zip(variables, grid):
#            values = values.ravel()
#            setattr(plot, column.variable, values.ravel())
        axes[plot.name] = plot
        figure[1, 1] = axes
        return figure

    @property
    def table(self): return self.__table
    @table.setter
    def table(self, table): self.__table = table



