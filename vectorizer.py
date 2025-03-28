import ctypes
import amd

def vectorize_func(prog, mem):
    first_state = mem.first_state
    last_state = first_state + mem.count_states
    first_obs = mem.first_obs
    last_obs = first_obs + mem.count_obs

    # prologue
    prog.push("rbp")  # heap
    prog.push("rbx")  # table
    prog.push("r12")  #
    prog.push("r13")  #
    prog.push("r14")  # func
    prog.push("r15")  #
    prog.push("rcx")  #
    
    prog.mov("rbp", "rdi")  # heap
    prog.mov("rbx", "rsi")  # table
    prog.mov("r12", "rdx")  # buf (third argument)
    prog.mov("r13", "rcx")  # n (fourth argument)
    prog.mov("r15", "rcx")
    prog.mov("r14", "r8")   # func

    # r13 := 8 * n
    prog.add("r13", "r13")
    prog.add("r13", "r13")
    prog.add("r13", "r13")   

    prog.set_label("loop")
    
    prog.mov("rdi", "r12")

    for i in range(first_state, last_state):
        prog.movsd_xmm_mem(0, "rdi", 0)
        prog.movsd_mem_xmm("rbp", 8 * i, 0)
        prog.add("rdi", "r13")

    prog.mov("rdi", "rbp")
    prog.mov("rsi", "rbx")
    prog.call("r14")

    prog.mov("rdi", "r12")

    for i in range(first_obs, last_obs):
        prog.movsd_xmm_mem(0, "rbp", 8 * i)
        prog.movsd_mem_xmm("rdi", 0, 0)
        prog.add("rdi", "r13")

    prog.add_imm("r12", 8)
    prog.dec("r15")
    prog.jnz("loop")

    prog.mov("rax", "r13")

    # epilogue
    prog.pop("rcx")
    prog.pop("r15")
    prog.pop("r14")
    prog.pop("r13")
    prog.pop("r12")
    prog.pop("rbx")
    prog.pop("rbp")
    prog.ret()

    prog.apply_jumps()
        
        
