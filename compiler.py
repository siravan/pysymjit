from sympy import *
import numpy as np
import ctypes
import mmap
import numbers

import amd
import table
import tree


class Func:
    def __init__(self, model, prog, mem, vt):
        self.code = mmap.mmap(-1, len(prog.buf), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC, flags=mmap.MAP_ANON | mmap.MAP_PRIVATE)
        self.code.write(prog.buf)
        self.code.flush()
        
        fac = ctypes.CFUNCTYPE(
            ctypes.c_size_t, 
            ctypes.POINTER(ctypes.c_double), 
            ctypes.POINTER(ctypes.c_size_t)
        )
        self.func = fac(ctypes.addressof(ctypes.c_long.from_buffer(self.code)))      
        self.heap = mem.mem()
        self.table = vt.vt()
        self.mem = mem
        self.prog = prog
        
    def __del__(self):
        self.code.close()

    def __call__(self, *args):
        if len(args) > self.mem.count_states:
            p = np.array(args[self.mem.count_states:], dtype="double")
            self.heap[self.mem.first_param:self.mem.first_param+self.mem.count_params] = p
    
        if isinstance(args[0], numbers.Number):
            u = np.array(args[:self.mem.count_states], dtype="double")
            self.heap[self.mem.first_state:self.mem.first_state+self.mem.count_states] = u

            self.res = self.func(
                np.ctypeslib.as_ctypes(self.heap), 
                np.ctypeslib.as_ctypes(self.table)
            )

            return self.heap[self.mem.first_obs:self.mem.first_obs+self.mem.count_obs].copy()
        else:
            raise ValueError("vectorized operations are not supported yet")


def compile_func(states, eqs, params=None, obs=None):
    f = tree.model(states, eqs, params, obs)
    mem = tree.Memory(f)
    prog = amd.Amd()
    vt = table.VirtualTable()
    
    f.compile(prog, mem, vt)
    
    return Func(f, prog, mem, vt)

################################################################

def test():
    x, y = symbols('x y')

    eqs = [
        [x + y],
        [x - y],
        [x * y],
        [x / y],
        [1 / x],
        [sqrt(x)],
        [x ** 2],
        [x ** 3],
        [x ** 4],
        [x ** (-1)],
        [x ** (-2)],
        [x ** (-3)],
        [x ** (-4)],
        [x ** Rational(1, 2)],
        [x ** Rational(3, 2)],
        [x ** Rational(2, 3)],
        [x ** Rational(4, 3)],
        [x ** y],
        [-x],
        [abs(x - y)],    
        [exp(x)],
        [log(x)],
        [sin(x)],
        [cos(x)],
        [tan(x)],
        #[csc(x)],
        #[sec(x)],
        #[cot(x)],
        [sinh(x)],
        [cosh(x)],
        [tanh(x)],
        #[csch(x)],
        #[sech(x)],
        #[coth(x)],
        [asin(x)],
        [acos(x)],
        [atan(x)],
        [asinh(x)],
        [acosh(1 + x)],
        [atanh(x)],
        [Piecewise((1, x > y), (0, True))],
        [Piecewise((1, x < y), (0, True))],
        [Piecewise((1, x >= y), (0, True))],
        [Piecewise((1, x <= y), (0, True))],
        [Piecewise((1, ((x < y) | (x > sqrt(y)))), (0, True))],
        [Piecewise((1, ((x < y) & (x > sqrt(y)))), (0, True))],
        [Piecewise((1, ((x < y) ^ (x > sqrt(y)))), (0, True))],
    ]

    for eq in eqs:
        print('testing ', eq)
        f = compile_func([x, y], eq)
        g = lambdify([x, y], eq)
        X = np.random.rand() * 0.9 + 0.1
        Y = np.random.rand() * 0.9 + 0.1
        np.testing.assert_array_almost_equal(f(X, Y), g(X, Y))
        
    print('ok!')


