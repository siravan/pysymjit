import ctypes
import mmap
import numbers
import numpy as np
from sympy import *
import matplotlib.pyplot as plt

import amd
import table
import tree
import vectorizer


class Func:
    def __init__(self, model):
        mem = tree.Memory(model)
        vt = table.VirtualTable()

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

    def __del__(self):
        self.code.close()
        self.vcode.close()

    def __call__(self, *args):
        first_state = self.mem.first_state
        count_states = self.mem.count_states
        last_state = first_state + count_states
        first_param = self.mem.first_param
        last_param = first_param + self.mem.count_params
        first_obs = self.mem.first_obs
        last_obs = first_obs + self.mem.count_obs

        if len(args) > count_states:
            p = np.array(args[count_states:], dtype="double")
            self.heap[first_param:last_param] = p

        if isinstance(args[0], numbers.Number):
            u = np.array(args[:count_states], dtype="double")
            self.heap[first_state:last_state] = u

            self.func(
                self.heap_ptr,
                self.table_ptr,
            )

            return self.heap[first_obs:last_obs].copy()
        else:
            return self.call_vectorized(*args)

    def call_vectorized(self, *args):
        first_state = self.mem.first_state
        count_states = self.mem.count_states
        last_state = first_state + count_states
        first_obs = self.mem.first_obs
        count_obs = self.mem.count_obs
        last_obs = first_obs + count_obs

        assert len(args) >= count_states
        shape = args[0].shape
        n = args[0].size
        h = max(count_states, count_obs)
        buf = np.zeros((h, n), dtype="double")

        for i in range(count_states):
            assert args[i].shape == shape
            buf[i, :] = args[i].ravel()

        self.vfunc(
            self.heap_ptr,
            self.table_ptr,
            ctypes.addressof(ctypes.c_long.from_buffer(buf)),
            ctypes.c_size_t(n),
            self.addr,
        )

        res = []
        for i in range(count_obs):
            y = buf[i, :].reshape(shape)
            res.append(y)

        return res


def compile_func(states, eqs, params=None, obs=None):
    f = tree.model(states, eqs, params, obs)
    return Func(f)


################################################################


def test():
    x, y = symbols("x y")

    eqs = [
        [x + y],
        [x - y],
        [x * y],
        [x / y],
        [1 / x],
        [sqrt(x)],
        [x**2],
        [x**3],
        [x**4],
        [x ** (-1)],
        [x ** (-2)],
        [x ** (-3)],
        [x ** (-4)],
        [x ** Rational(1, 2)],
        [x ** Rational(3, 2)],
        [x ** Rational(2, 3)],
        [x ** Rational(4, 3)],
        [x**y],
        [-x],
        [abs(x - y)],
        [exp(x)],
        [log(x)],
        [sin(x)],
        [cos(x)],
        [tan(x)],
        # [csc(x)],
        # [sec(x)],
        # [cot(x)],
        [sinh(x)],
        [cosh(x)],
        [tanh(x)],
        # [csch(x)],
        # [sech(x)],
        # [coth(x)],
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

    X = np.random.rand(15) * 0.9 + 0.1
    Y = np.random.rand(15) * 0.9 + 0.1

    for eq in eqs:
        print("testing ", eq)
        f = compile_func([x, y], eq)
        g = lambdify([x, y], eq)
        np.testing.assert_array_almost_equal(f(X[0], Y[0]), g(X[0], Y[0]))
        np.testing.assert_array_almost_equal(f(X, Y), g(X, Y))

    print("ok!")


def manderbrot():
    x, y, a, b = symbols("x y a b")

    A, B = np.meshgrid(np.arange(-2, 1, 0.002), np.arange(-1.5, 1.5, 0.002))
    X = np.zeros_like(A)
    Y = np.zeros_like(A)

    f = compile_func([a, b, x, y], [x**2 - y**2 + a, 2 * x * y + b])

    for i in range(20):
        X, Y = f(A, B, X, Y)

    Z = np.hypot(X, Y)

    plt.imshow(Z < 2)
    plt.show()
