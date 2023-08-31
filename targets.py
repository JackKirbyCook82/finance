# -*- coding: utf-8 -*-
"""
Created on Weds Jul 19 2023
@name:   Target Objects
@author: Jack Kirby Cook

"""

import xarray as xr

from support.pipelines import Processor

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["TargetProcessor"]
__copyright__ = "Copyright 2023, Jack Kirby Cook"
__license__ = ""


class TargetProcessor(Processor):
    def execute(self, contents, *args, funds, apy, **kwargs):
        ticker, expire, strategy, valuations = contents
        assert isinstance(valuations, xr.Dataset)
        dataframe = valuations.to_dask_dataframe() if bool(valuations.chunks) else valuations.to_dataframe()
        dataframe = dataframe.where(dataframe["apy"] >= apy)
        dataframe = dataframe.where(dataframe["cost"] <= funds)
        dataframe = dataframe.dropna(how="all")
        for partition in self.partitions(dataframe):
            partition = partition.sort_values("apy", axis=1, ascending=False, ignore_index=True, inplace=False)
            for record in partition.to_dict("records"):
                pass

    @staticmethod
    def partitions(dataframe):
        if not hasattr(dataframe, "npartitions"):
            yield dataframe
            return
        for index in dataframe.npartitions:
            partition = dataframe.get_partition(index).compute()
            yield partition



