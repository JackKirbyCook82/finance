# -*- coding: utf-8 -*-
"""
Created on Sat July 4 2026
@name:   Finance Broker Objects
@author: Jack Kirby Cook

"""

from enum import Enum
from dataclasses import dataclass
from types import SimpleNamespace
from attr.converters import to_bool

__version__ = "1.0.0"
__author__ = "Jack Kirby Cook"
__all__ = ["Brokerage", "Authenticator", "Account"]
__copyright__ = "Copyright 2026, Jack Kirby Cook"
__license__ = "MIT License"


class Loading(object):
    @classmethod
    def load(cls, file, /, delimiter=" ", **kwargs):
        assert isinstance(delimiter, str)
        lines = file.read_text().splitlines()
        header = str(lines[0]).split(delimiter)
        body = [str(line).split(" ") for line in lines[1:]]
        records = [dict(zip(header, line)) for line in body]
        return [SimpleNamespace(**record) for record in records]


@dataclass(frozen=True)
class Brokerage: website: Enum; live: bool


@dataclass(frozen=True)
class Authenticator(Loading):
    identity: str; code: str

    @classmethod
    def load(cls, file, /, **kwargs):
        records = super().load(file, **kwargs)
        instances = {Brokerage(record.website, to_bool(record.live)): cls(record.identity, record.code) for record in records}
        return instances


@dataclass(frozen=True)
class Account(Loading):
    identity: str; username: str; password: str

    @classmethod
    def load(cls, file, /, **kwargs):
        records = super().load(file, **kwargs)
        instances = {Brokerage(record.website, to_bool(record.live)): cls(record.identity, record.username, record.password) for record in records}
        return instances


