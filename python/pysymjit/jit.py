import ctypes
import os
import numpy as np

libc = ctypes.cdll.msvcrt
win = ctypes.windll

s = libc.sin
s.restype = ctypes.c_double
s.argtypes = [ctypes.c_double]

print(s)

CODE_SIZE = 4096

alloc = win.kernel32.VirtualAlloc
alloc.restype = ctypes.c_void_p
alloc.argtypes = [
    ctypes.c_void_p,    # lpAddress
    ctypes.c_size_t,    # dwSize
    ctypes.c_uint32,    # flAllocationType
    ctypes.c_uint32     # flProtect
]

protect = win.kernel32.VirtualProtect
protect.restype = ctypes.c_bool
protect.argtypes = [
    ctypes.c_void_p,    # lpAddress
    ctypes.c_size_t,    # dwSize    
    ctypes.c_uint32,    # flNewProtect
    ctypes.POINTER(ctypes.c_uint32)    # lpflOldProtect
]

MEM_COMMIT = 0x00001000
MEM_RESERVE = 0x00002000
PAGE_EXECUTE_READ = 0x00000020
PAGE_EXECUTE_READWRITE = 0x00000040
PAGE_READWRITE = 0x00000004

code = bytes(
    [ 
        0x50,                       # push rax 
        0x48, 0x83, 0xec, 0x20,     # sub rsp, 32
        0xF2, 0x0F, 0x10, 0xC1,     # movsd xmm0, xmm1
        0xF2, 0x0F, 0x58, 0xC0,     # addsd xmm0, xmm0
        0x48, 0x89, 0xC8,           # mov rax, rcx  
        0xFF, 0xD0,                 # call rax
        0x48, 0x83, 0xc4, 0x20,     # add rsp, 32
        0x58,                       # pop rax
        0xC3                        # ret
    ]
)  

mem = alloc(
    None,
    CODE_SIZE,
    MEM_RESERVE | MEM_COMMIT,
    PAGE_EXECUTE_READWRITE
)

ctypes.memmove(mem, code, len(code))

old = ctypes.c_uint32(0)


#protect(
#    mem,
#    CODE_SIZE,
#    PAGE_EXECUTE_READ,
#    ctypes.pointer(old)    
#)    

# p = ctypes.addressof(mem)
# print(hex(p))
f = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_uint64, ctypes.c_double)(mem)
q = ctypes.c_ulonglong.from_buffer(s)

print(q)

#for i in range(10):
#    print(hex(mem[i]))

for i in range(20):
    res = f(q, 0.1*i)
    print(0.1*i, res)