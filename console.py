# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 12 2026
@name:   Finance Logging Objects
@author: Jack Kirby Cook
@file:   finance/logging.py

"""

import logging
from abc import ABC
from enum import Enum, auto

from support.decorators import Decorator

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["LoggingDecorators", "LoggingSizing"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class LoggingSizing(Enum):
    INLET = auto()
    OUTLET = auto()
    BOTH = auto()


class LoggingDecorator(Decorator, ABC):
    def __init__(self, *collections, title, sizing=None, percentage=False, **kwargs):
        super().__init__(**kwargs)
        self.__collections = collections
        self.__percentage = percentage
        self.__sizing = sizing
        self.__title = title
        self.__logger = __logger__

    def identity(self, dataset):
        for collection in self.collections:
            try: yield str(collection(dataset))
            except (KeyError, AttributeError): pass

    def string(self, inlet):
        strings = list()
        for collection in self.collections:
            try: strings.append(str(collection(inlet)))
            except (KeyError, AttributeError): pass
        try: name = str(self.instance.name)
        except AttributeError: name = str(self.instance.__class__.__name__)
        return f"{str(self.title)}: {str(name)}: {', '.join(strings)}"

    @staticmethod
    def size(inlet, outlet, /, sizing):
        size = lambda content: len(content) if content is not None else 0
        if sizing == LoggingSizing.BOTH: string = f"{size(inlet):.0f}->{size(outlet):.0f}"
        elif sizing == LoggingSizing.INLET: string = f"{size(inlet):.0f}"
        elif sizing == LoggingSizing.OUTLET: string = f"{size(outlet):.0f}"
        else: raise ValueError(sizing)
        return string

    @staticmethod
    def percent(inlet, outlet):
        size = lambda content: len(content) if content is not None else 0
        return f"{(size(outlet) / size(inlet)) * 100:.0f}%"

    @property
    def collections(self): return self.__collections
    @property
    def percentage(self): return self.__percentage
    @property
    def logger(self): return self.__logger
    @property
    def sizing(self): return self.__sizing
    @property
    def title(self): return self.__title


class LoggingDecoratorFunction(LoggingDecorator):
    def decorator(self, instance, inlet, *args, **kwargs):
        outlet = self.invoke(instance, inlet, *args, **kwargs)
        if inlet is not None and len(inlet) > 0:
            string = self.string(inlet)
            if bool(self.sizing): string += (" " + self.size(inlet, outlet, sizing=self.sizing))
            if bool(self.percentage): string += (" " + self.percent(inlet, outlet))
            self.logger.info(string)
        return outlet


class LoggingDecoratorGenerator(LoggingDecorator):
    def decorator(self, instance, inlet, *args, **kwargs):
        generator = self.invoke(instance, inlet, *args, **kwargs)
        for outlet in generator:
            if inlet is not None and len(inlet) > 0:
                string = self.string(inlet)
                if bool(self.sizing): string += (" " + self.size(inlet, outlet, sizing=self.sizing))
                if bool(self.percentage): string += (" " + self.percent(inlet, outlet))
                self.logger.info(string)
                yield outlet
            else: yield outlet


class LoggingDecorators:
    Function = LoggingDecoratorFunction
    Generator = LoggingDecoratorGenerator

