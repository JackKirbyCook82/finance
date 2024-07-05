# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 31 2024
@name:   Holdings Objects
@author: Jack Kirby Cook

"""

import logging
import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC
from itertools import product
from collections import OrderedDict as ODict

from finance.variables import Variables, Contract
from support.calculations import Variable, Equation, Calculation
from support.pipelines import Producer, Consumer
from support.tables import Tables, Options
from support.files import File

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["HoldingFiles", "HoldingTable", "HoldingReader", "HoldingWriter"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


holdings_dates = {"expire": "%Y%m%d"}
holdings_parsers = {"instrument": Variables.Instruments, "option": Variables.Options, "position": Variables.Positions}
holdings_formatters = {"instrument": int, "option": int, "position": int}
holdings_types = {"ticker": str, "strike": np.float32, "quantity": np.int32}
holdings_filename = lambda query: "_".join([str(query.ticker).upper(), str(query.expire.strftime("%Y%m%d"))])
holdings_parameters = dict(datatype=pd.DataFrame, filename=holdings_filename, dates=holdings_dates, parsers=holdings_parsers, formatters=holdings_formatters, types=holdings_types)
holdings_header = ["ticker", "expire", "strike", "instrument", "option", "position", "quantity"]

holdings_formats = {(lead, lag): lambda column: f"{column:.02f}" for lead, lag in product(["npv", "cost"], list(map(lambda scenario: str(scenario), Variables.Scenarios)))}
holdings_formats.update({(lead, lag): lambda column: f"{column * 100:.02f}%" for lead, lag in product(["apy"], list(map(lambda scenario: str(scenario), Variables.Scenarios)))})
holdings_formats.update({("priority", ""): lambda priority: f"{priority:.02f}"})
holdings_formats.update({("status", ""): lambda status: str(status)})
holdings_options = Options.Dataframe(rows=20, columns=25, width=1000, formats=holdings_formats, numbers=lambda column: f"{column:.02f}")


class HoldingFile(File, variable=Variables.Datasets.HOLDINGS, header=holdings_header, **holdings_parameters):
    pass


class HoldingTable(Tables.Dataframe, datatype=pd.DataFrame, options=holdings_options):
    def write(self, locator, content, *args, **kwargs):
        locator = self.locate(locator)
        super().write(locator, content, *args, **kwargs)

    def read(self, locator, *args, **kwargs):
        locator = self.locate(locator)
        return super().read(locator, *args, **kwargs)

    def locate(self, locator):
        index, column = locator
        assert isinstance(index, (int, slice))
        if isinstance(index, slice):
            assert index.step is None
            length = len(list(range(index.start, index.stop)))
            index = slice(length) if length > len(self.table.index) else index
        if isinstance(column, str):
            column = (column, "")
            column = list(self.table.columns).index(column)
        elif isinstance(column, tuple):
            assert len(column) == 2
            column = list(self.table.columns).index(column)
        return index, column


class HoldingEquation(Equation): pass
class XXXEquation(HoldingEquation):
    yi = Variable("yi", "price", np.float32, function=lambda x, k, Θ, Φ: Φ * np.max(Θ * (x - k), 0))
    Θ = Variable("Θ", "theta", np.int32, function=lambda m: int(Variables.Theta[str(m)]))
    Φ = Variable("Φ", "phi", np.int32, function=lambda n: int(Variables.Phi[str(n)]))

    m = Variable("m", "option", Variables.Options, position=0, locator="option")
    n = Variable("n", "position", Variables.Positions, position=0, locator="position")
    xi = Variable("xi", "underlying", np.float32, position=0, locator="underlying")
    k = Variable("k", "strike", np.float32, position=0, locator="strike")

    def execute(self, exposure, *args, **kwargs):
        equation = self.equation(*args, **kwargs)
        yield equation.yi(exposure)


class HoldingCalculation(Calculation, ABC, fields=["xxx"]):
class XXXCalculation(HoldingCalculation, xxx=, equation=XXXEquation):
    def execute(self, exposure, *args, **kwargs):
        equation = self.equation(*args, **kwargs)
        yield equation.y(exposure)


class HoldingReader(Producer, ABC):
    def __init__(self, *args, source, **kwargs):
        super().__init__(*args, **kwargs)
        self.__source = source

    def execute(self, *args, **kwargs):
        valuations = self.read(*args, **kwargs)
        assert isinstance(valuations, pd.DataFrame)
        if bool(valuations.empty):
            return
        valuations["quantity"] = 1
        contract = self.contract(valuations, *args, **kwargs)
        options = self.options(valuations, *args, **kwargs)
        stocks = self.stocks(valuations, *args, **kwargs)
        stocks = stocks.dropna(how="all", axis=1)
        securities = pd.concat([contract, options, stocks, valuations["quantity"]], axis=1, ignore_index=False)
        holdings = self.holdings(securities, *args, **kwargs)
        index = [column for column in holdings_header if column != "quantity"]
        holdings = holdings.groupby(index, as_index=False, dropna=False, sort=False)["quantity"].sum()
        for (ticker, expire), dataframe in iter(holdings.groupby(["ticker", "expire"])):
            contract = Contract(ticker, expire)
            holdings = {Variables.Querys.CONTRACT: contract, Variables.Datasets.HOLDINGS: dataframe}
            yield holdings

    def read(self, *args, **kwargs):
        if not bool(self.source):
            return pd.DataFrame()
        mask = self.source.table["status"] == Variables.Status.PURCHASED
        dataframe = self.source.table.where(mask).dropna(how="all", inplace=False)
        self.source.remove(dataframe, *args, **kwargs)
        return dataframe

    @staticmethod
    def contract(dataframe, *args, **kwargs):
        dataframe = dataframe.droplevel("scenario", axis=1)
        return dataframe[["ticker", "expire"]]

    @staticmethod
    def options(dataframe, *args, **kwargs):
        securities = list(map(str, Variables.Securities.Options))
        securities = [column for column in securities if column in dataframe.columns]
        dataframe = dataframe.droplevel("scenario", axis=1)
        return dataframe[securities]

    @staticmethod
    def stocks(dataframe, *args, **kwargs):
        securities = list(map(str, Variables.Securities.Stocks))
        strategies = lambda cols: list(map(str, cols["strategy"].stocks))
        underlying = lambda cols: np.round(cols["underlying"], decimals=2)
        function = lambda cols: [underlying(cols) if column in strategies(cols) else np.NaN for column in securities]
        dataframe = dataframe.droplevel("scenario", axis=1)
        dataframe = dataframe.apply(function, axis=1, result_type="expand")
        dataframe.columns = securities
        return dataframe

    @staticmethod
    def holdings(dataframe, *args, **kwargs):
        securities = [security for security in list(map(str, Variables.Securities)) if security in dataframe.columns]
        contracts = [column for column in dataframe.columns if column not in securities]
        dataframe = dataframe.melt(id_vars=contracts, value_vars=securities, var_name="security", value_name="strike")
        dataframe = dataframe.where(dataframe["strike"].notna()).dropna(how="all", inplace=False)
        dataframe["security"] = dataframe["security"].apply(Variables.Securities)
        dataframe["instrument"] = dataframe["security"].apply(lambda security: security.instrument)
        dataframe["option"] = dataframe["security"].apply(lambda security: security.option)
        dataframe["position"] = dataframe["security"].apply(lambda security: security.position)
        return dataframe

    @property
    def source(self): return self.__source


class HoldingWriter(Consumer, ABC):
    def __init__(self, *args, destination, liquidity, priority, valuation, capacity=None, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        calculations = {variables["pricing"]: calculation for variables, calculation in ODict(list(HoldingCalculation)).items()}
        self.__calculation = calculations[](*args, **kwargs)
        self.__destination = destination
        self.__valuation = valuation
        self.__liquidity = liquidity
        self.__priority = priority
        self.__capacity = capacity

    def execute(self, contents, *args, **kwargs):
        valuations = contents[self.valuation]
        exposure = contents.get(Variables.Datasets.EXPOSURE, None)
        assert isinstance(valuations, pd.DataFrame)
        if bool(valuations.empty):
            return
        valuations = self.market(valuations, *args, **kwargs)
        valuations = self.prioritize(valuations, *args, **kwargs)
        valuations = self.portfolio(valuations, *args, exposure=exposure, **kwargs)
        if bool(valuations.empty):
            return
        valuations = valuations.reset_index(drop=True, inplace=False)
        self.write(valuations, *args, **kwargs)

    def market(self, dataframe, *args, **kwargs):
        index = set(dataframe.columns) - {"scenario", "apy", "npv", "cost"}
        dataframe = dataframe.pivot(index=index, columns="scenario")
        dataframe = dataframe.reset_index(drop=False, inplace=False)
        dataframe["liquidity"] = dataframe.apply(self.liquidity, axis=1)
        return dataframe

    def prioritize(self, dataframe, *args, **kwargs):
        dataframe["priority"] = dataframe.apply(self.priority, axis=1)
        dataframe = dataframe.sort_values("priority", axis=0, ascending=False, inplace=False, ignore_index=False)
        dataframe = dataframe.where(dataframe["priority"] > 0).dropna(axis=0, how="all")
        dataframe = dataframe.reset_index(drop=True, inplace=False)
        return dataframe

    def portfolio(self, dataframe, *args, exposure, **kwargs):
        if exposure is None:
            return dataframe

        pd.set_option("display.max_columns", 100)
        pd.set_option("display.width", 300)
        print(dataframe)
        print(exposure)

        exposure = exposure.set_index(["ticker", "expire", "strike", "instrument", "option", "position"], drop=True, inplace=False)
        exposure = xr.Dataset.from_dataframe(exposure).fillna(0)
        exposure = exposure.squeeze("ticker").squeeze("expire").squeeze("instrument")
        exposure["underlying"] = np.unique(exposure["strike"].values)
        exposure = exposure.stack({"holdings": ["strike", "option", "position"]})
        results = self.calculation(exposure, *args, **kwargs)

        print(exposure)
        print(results)

        raise Exception()

    def write(self, dataframe, *args, **kwargs):
        dataframe["status"] = Variables.Status.PROSPECT
        if not bool(self.destination):
            self.destination.table = dataframe
        else:
            index = np.max(self.destination.table.index.values) + 1
            dataframe = dataframe.set_index(dataframe.index + index, drop=True, inplace=False)
            self.destination.concat(dataframe, *args, **kwargs)
        self.destination.sort("priority", reverse=True)
        if bool(self.capacity):
            self.destination.truncate(self.capacity)

    @property
    def destination(self): return self.__destination
    @property
    def calculation(self): return self.__calculation
    @property
    def valuation(self): return self.__valuation
    @property
    def liquidity(self): return self.__liquidity
    @property
    def priority(self): return self.__priority
    @property
    def capacity(self): return self.__capacity


class HoldingFiles(object):
    Holding = HoldingFile




