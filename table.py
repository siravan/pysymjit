import ctypes
from ctypes.util import find_library
import numpy as np


class VirtualTable:
    def __init__(self):
        path = find_library("m")
        self.lib = ctypes.CDLL(path)

        self.addr = []
        self.dict = {}

        self.exp = self.unary(self.lib.exp, "exp")
        self.log = self.unary(self.lib.log, "log")
        self.log10 = self.unary(self.lib.log10, "log10")
        self.pow = self.binary(self.lib.pow, "pow")

        self.sin = self.unary(self.lib.sin, "sin")
        self.cos = self.unary(self.lib.cos, "cos")
        self.tan = self.unary(self.lib.tan, "tan")

        self.sinh = self.unary(self.lib.sinh, "sinh")
        self.cosh = self.unary(self.lib.cosh, "cosh")
        self.tanh = self.unary(self.lib.tanh, "tanh")

        self.asin = self.unary(self.lib.asin, "asin")
        self.acos = self.unary(self.lib.acos, "acos")
        self.atan = self.unary(self.lib.atan, "atan")

        self.asinh = self.unary(self.lib.asinh, "asinh")
        self.acosh = self.unary(self.lib.acosh, "acosh")
        self.atanh = self.unary(self.lib.atanh, "atanh")

    def unary(self, func, name):
        self.addr.append(ctypes.c_long.from_buffer(func).value)
        self.dict[name] = len(self.addr) - 1
        func.restype = ctypes.c_double
        func.argtypes = [ctypes.c_double]
        return func

    def binary(self, func, name):
        self.addr.append(ctypes.c_long.from_buffer(func).value)
        self.dict[name] = len(self.addr) - 1
        func.restype = ctypes.c_double
        func.argtypes = [ctypes.c_double, ctypes.c_double]
        return func

    def vt(self):
        return np.asarray(self.addr, dtype=np.uint64)

    def find(self, op):
        return self.dict[op]
