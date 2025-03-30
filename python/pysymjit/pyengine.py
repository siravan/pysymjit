import sys
import ctypes
import mmap
from ctypes.util import find_library
import numpy as np
import platform

from . import amd
from . import vectorizer

def can_compile():
    if sys.platform == "linux" and platform.machine() == "x86_64":
        return True
    elif sys.platform == "darwin" and platform.machine() == "x86_64":
        return True
    else:
        return False


class Memory:
    def __init__(self, model):
        self.names = ["0.0", "1.0", "-1.0", "-0.0"]
        self.vals = [0.0, 1.0, -1.0, -0.0]

        self.names.append(model.iv.name)
        self.vals.append(0.0)

        self.first_state = len(self.names)
        self.count_states = len(model.states)

        for var in model.states:
            self.names.append(var.name)
            self.vals.append(0.0)

        self.first_param = len(self.names)
        self.count_params = len(model.params)

        for var in model.params:
            self.names.append(var.name)
            self.vals.append(0.0)

        self.first_obs = len(self.names)
        self.count_obs = len(model.obs)

        for var in model.obs:
            self.names.append(var.name)
            self.vals.append(0.0)

        self.first_diff = len(self.names)
        self.count_diffs = len(model.odes)

        for i in range(len(model.odes)):
            self.names.append(f"δ{model.states[i].name}")
            self.vals.append(0.0)

        self.stack_ptr = 0
        self.stack_size = 0

    def index(self, name):
        return self.names.index(name)

    def index_diff(self, name):
        return self.names.index(f"δ{name}")

    def mem(self):
        return np.asarray(self.vals)

    def constant(self, val):
        val = float(val)
        try:
            return self.names.index(str(val))
        except:
            self.names.append(str(val))
            self.vals.append(val)
            return len(self.vals) - 1

    def push(self):
        self.stack_ptr += 1
        self.stack_size = max(self.stack_size, self.stack_ptr)
        return self.stack_ptr - 1

    def pop(self):
        assert self.stack_ptr > 0
        self.stack_ptr -= 1
        return self.stack_ptr


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


class PyCompiler:        
    def __init__(self, model):
        mem = Memory(model)
        vt = VirtualTable()

        prog = amd.Amd()
        model.compile(0, prog, mem, vt)

        self.code = mmap.mmap(
            -1,
            len(prog.buf),
            prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
            flags=mmap.MAP_ANON | mmap.MAP_PRIVATE,
        )
        self.code.write(prog.buf)
        self.code.flush()

        fac = ctypes.CFUNCTYPE(
            ctypes.c_size_t,    # restype 
            ctypes.c_size_t,    # address of the heap
            ctypes.c_size_t,    # address of the virtual table
        )

        self.addr = ctypes.addressof(ctypes.c_long.from_buffer(self.code))
        self.func = fac(self.addr)
        self.heap = mem.mem()
        self.table = vt.vt()
        self.heap_ptr = ctypes.addressof(ctypes.c_long.from_buffer(self.heap))
        self.table_ptr = ctypes.addressof(ctypes.c_long.from_buffer(self.table))
        self.mem = mem
        self.prog = prog

        vprog = amd.Amd()
        vec = vectorizer.vectorize_func(vprog, mem)

        self.vcode = mmap.mmap(
            -1,
            len(vprog.buf),
            prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
            flags=mmap.MAP_ANON | mmap.MAP_PRIVATE,
        )
        self.vcode.write(vprog.buf)
        self.vcode.flush()

        fac = ctypes.CFUNCTYPE(
            ctypes.c_size_t,
            ctypes.c_size_t,    # address of the heap
            ctypes.c_size_t,    # address of the virtual table
            ctypes.c_size_t,    # address of the 2D buffer (buf)
            ctypes.c_size_t,    # number of vectorization repeats (# columns of buf)
            ctypes.c_size_t,    # func
        )

        self.vfunc = fac(ctypes.addressof(ctypes.c_long.from_buffer(self.vcode)))
        self.vprog = vprog         
        self.populate()           

    def populate(self):
        first_state = self.mem.first_state
        first_param = self.mem.first_param
        first_obs = self.mem.first_obs
        first_diff = self.mem.first_diff
        
        self.count_states = self.mem.count_states
        self.count_params = self.mem.count_params
        self.count_obs = self.mem.count_obs
        self.count_diffs = self.mem.count_diffs

        self.states = self.heap[first_state:first_state+self.count_states]
        self.params = self.heap[first_param:first_param+self.count_params]
        self.obs = self.heap[first_obs:first_obs+self.count_obs]
        self.diffs = self.heap[first_diff:first_diff+self.count_diffs]        
        
    def dump(self, name, what="scalar"):
        if what == "scalar":
            buf = self.prog.buf
        elif what == "vectorized":
            buf = self.vprog.buf
        else:
            raise ValueError("undefined `what`")
            
        with open(name, "wb") as fd:
            fd.write(buf)
            
    def execute(self, t = 0.0):
        self.func(self.heap_ptr, self.table_ptr)
        
    def execute_vectorized(self, buf):
        self.vfunc(
            self.heap_ptr,
            self.table_ptr,
            ctypes.addressof(ctypes.c_long.from_buffer(buf)),
            ctypes.c_size_t(buf.shape[1]),
            self.addr,
        ) 

