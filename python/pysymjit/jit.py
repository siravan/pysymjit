import mmap
import ctypes
from ctypes.util import find_library
import os

path = find_library("m")
lib = ctypes.CDLL(path)

s = lib.sin
s.restype = ctypes.c_double
s.argtypes = [ctypes.c_double]

PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
CODE_SIZE = PAGE_SIZE

code = bytes(
    [0xF2, 0x0F, 0x58, 0xC0, 0xFF, 0xD7, 0xC3]
)  # addsd xmm0, xmm0; call rdi; ret;
mem = mmap.mmap(
    -1,
    CODE_SIZE,
    prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
    flags=mmap.MAP_ANON | mmap.MAP_PRIVATE,
)
mem.write(code)
mem.flush()

p = ctypes.addressof(ctypes.c_long.from_buffer(mem))
f = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_long, ctypes.c_double)(p)

print(f(ctypes.c_long.from_buffer(s), 0.5))
mem.close()
