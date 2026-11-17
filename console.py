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
from support.mixins import Mixin

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["LoggingMixin", "LoggingDecorators", "LoggingSizing"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"
__logger__ = logging.getLogger(__name__)


class LoggingSizing(Enum):
    INLET = auto()
    OUTLET = auto()
    BOTH = auto()


class LoggingMixin(Mixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        name = self.__class__.__name__
        self.__name = kwargs.get("name", name)
        self.__logger = __logger__

    def console(self, *strings, title, **kwargs):
        string = ", ".join(strings)
        string = f"{str(title)}: {str(self.name)}: {str(string)}"
        self.logger.info(string)

    @property
    def logger(self): return self.__logger
    @property
    def name(self): return self.__name


class LoggingDecorator(Decorator, ABC):
    def __init__(self, *collections, title, sizing=None, percentage=False, **kwargs):
        super().__init__(**kwargs)
        self.__collections = collections
        self.__percentage = percentage
        self.__sizing = sizing
        self.__title = title
        self.__logger = __logger__

    def heading(self, inlet):
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
        outlet = self.function(instance, inlet, *args, **kwargs)
        if bool(inlet) and inlet is not None:
            string = self.heading(inlet)
            if bool(self.sizing): string += (" " + self.size(inlet, outlet, sizing=self.sizing))
            if bool(self.percentage): string += (" " + self.percent(inlet, outlet))
            self.logger.info(string)
        return outlet


class LoggingDecoratorGenerator(LoggingDecorator):
    def decorator(self, instance, inlet, *args, **kwargs):
        generator = self.function(instance, inlet, *args, **kwargs)
        if bool(inlet) and inlet is not None:
            heading = self.heading(inlet)
            for outlet in generator:
                string = str(heading)
                if bool(self.sizing): string += (" " + self.size(inlet, outlet, sizing=self.sizing))
                if bool(self.percentage): string += (" " + self.percent(inlet, outlet))
                self.logger.info(string)
                yield outlet
        else: yield from generator


class LoggingDecorators:
    Function = LoggingDecoratorFunction
    Generator = LoggingDecoratorGenerator

