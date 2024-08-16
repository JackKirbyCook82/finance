# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 2024
@name:   Finance Operation Objects
@author: Jack Kirby Cook

"""

import logging
from abc import ABC
from collections import OrderedDict as ODict

from finance.variables import Variables
from support.pipelines import Producer, Processor, Consumer
from support.filtering import Filter

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Operations"]
__copyright__ = "Copyright 2023,SE Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class FinanceProducer(Producer, ABC):
    def report(self, *args, produced, elapsed, **kwargs):
        contract = produced[Variables.Querys.CONTRACT]
        string = f"{str(self.title)}: {repr(self)}|{str(contract)}[{elapsed:.02f}s]"
        __logger__.info(string)


class FinanceProcessor(Processor, ABC):
    def report(self, *args, produced, consumed, elapsed, **kwargs):
        assert produced[Variables.Querys.CONTRACT] == consumed[Variables.Querys.CONTRACT]
        contract = produced[Variables.Querys.CONTRACT]
        string = f"{str(self.title)}: {repr(self)}|{str(contract)}[{elapsed:.02f}s]"
        __logger__.info(string)


class FinanceConsumer(Consumer, ABC):
    def report(self, *args, consumed, elapsed, **kwargs):
        contract = consumed[Variables.Querys.CONTRACT]
        string = f"{str(self.title)}: {repr(self)}|{str(contract)}[{elapsed:.02f}s]"
        __logger__.info(string)


class FinanceFilter(Filter, Processor, title="Filtered"):
    def processor(self, contents, *args, **kwargs):
        contract = contents[Variables.Querys.CONTRACT]
        update = ODict(list(self.calculate(contents, *args, contract=contract, **kwargs)))
        if not bool(update):
            return
        yield contents | dict(update)

    def inform(self, *args, variable, contract, prior, post, **kwargs):
        variable = str(variable).lower().title()
        string = f"{str(self.title)}: {repr(self)}|{str(contract)}|{str(variable)}[{prior:.0f}|{post:.0f}]"
        __logger__.info(string)

    def report(self, *args, produced, consumed, elapsed, **kwargs):
        assert produced[Variables.Querys.CONTRACT] == consumed[Variables.Querys.CONTRACT]
        contract = produced[Variables.Querys.CONTRACT]
        string = f"{str(self.title)}: {repr(self)}|{str(contract)}[{elapsed:.02f}s]"
        __logger__.info(string)


class Operations:
    Producer = FinanceProducer
    Processor = FinanceProcessor
    Consumer = FinanceConsumer
    Filter = FinanceFilter



