from sympy import *
import numbers
import numpy as np


class Unary:
    def __init__(self, op, arg):
        self.op = op
        self.arg = arg
        self.is_pure = arg.is_pure and is_pure(op)

    def __repr__(self):
        return f"Unary[{self.is_pure}]('{self.op}', {self.arg})"

    def compile(self, dst, prog, mem, vt):
        self.arg.compile(0, prog, mem, vt)

        if self.op == "neg":
            prog.movsd_xmm_mem(1, "rbp", 8 * mem.constant(-0.0))
            prog.vxorpd(dst, 0, 1)
        elif self.op == "abs":
            prog.movsd_xmm_mem(1, "rbp", 8 * mem.constant(-0.0))
            prog.vandnpd(dst, 1, 0)
        elif self.op == "root":
            prog.vsqrtsd(dst, 0)
        elif self.op == "square":
            prog.vmulsd(dst, 0, 0)
        elif self.op == "cube":
            prog.vmulsd(1, 0, 0)
            prog.vmulsd(dst, 0, 1)
        elif self.op == "recip":
            prog.movsd_xmm_mem(1, "rbp", 8 * mem.constant(1.0))
            prog.vdivsd(dst, 1, 0)
        elif self.op == "exp":
            self.call(dst, prog, vt.find("exp"))
        elif self.op == "ln":
            self.call(dst, prog, vt.find("log"))
        elif self.op == "sin":
            self.call(dst, prog, vt.find("sin"))
        elif self.op == "cos":
            self.call(dst, prog, vt.find("cos"))
        elif self.op == "tan":
            self.call(dst, prog, vt.find("tan"))
        elif self.op == "sinh":
            self.call(dst, prog, vt.find("sinh"))
        elif self.op == "cosh":
            self.call(dst, prog, vt.find("cosh"))
        elif self.op == "tanh":
            self.call(dst, prog, vt.find("tanh"))
        elif self.op == "arcsin":
            self.call(dst, prog, vt.find("asin"))
        elif self.op == "arccos":
            self.call(dst, prog, vt.find("acos"))
        elif self.op == "arctan":
            self.call(dst, prog, vt.find("atan"))
        elif self.op == "arcsinh":
            self.call(dst, prog, vt.find("asinh"))
        elif self.op == "arccosh":
            self.call(dst, prog, vt.find("acosh"))
        elif self.op == "arctanh":
            self.call(dst, prog, vt.find("atanh"))
        else:
            raise ValueError(f"unary op {self.op} not defined")

    def call(self, dst, prog, idx):
        prog.mov_reg_mem("rax", "rbx", 8 * idx)
        prog.call("rax")
        if dst != 0:
            prog.movapd(dst, 0)


class Binary:
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right
        self.is_pure = left.is_pure and right.is_pure and is_pure(op)

    def __repr__(self):
        return f"Binary[{self.is_pure}]('{self.op}', {self.left}, {self.right})"

    def compile(self, dst, prog, mem, vt):
        sp = mem.push()
        use_reg = sp < 3 and self.left.is_pure

        if use_reg:
            self.right.compile(3 + sp, prog, mem, vt)
        else:
            self.right.compile(0, prog, mem, vt)
            prog.movsd_mem_xmm("rsp", 8 * sp, 0)

        self.left.compile(0, prog, mem, vt)
        sp = mem.pop()

        if use_reg:
            r = 3 + sp
        else:
            prog.movsd_xmm_mem(1, "rsp", 8 * sp)
            r = 1

        if self.op == "plus":
            prog.vaddsd(dst, 0, r)
        elif self.op == "minus":
            prog.vsubsd(dst, 0, r)
        elif self.op == "times":
            prog.vmulsd(dst, 0, r)
        elif self.op == "divide":
            prog.vdivsd(dst, 0, r)
        elif self.op == "power":
            prog.movapd(1, r)
            self.call(dst, prog, vt.find("pow"))
        elif self.op == "gt":
            prog.vcmpnlesd(dst, 0, r)
        elif self.op == "geq":
            prog.vcmpnltsd(dst, 0, r)
        elif self.op == "lt":
            prog.vcmpltsd(dst, 0, r)
        elif self.op == "leq":
            prog.vcmplesd(dst, 0, r)
        elif self.op == "eq":
            prog.vcmpeqsd(dst, 0, r)
        elif self.op == "neq":
            prog.vcmpneqsd(dst, 0, r)
        elif self.op == "and":
            prog.vandpd(dst, 0, r)
        elif self.op == "or":
            prog.vorpd(dst, 0, r)
        elif self.op == "xor":
            prog.vxorpd(dst, 0, r)
        else:
            raise ValueError(f"binary op {self.op} not defined")

    def call(self, dst, prog, idx):
        prog.mov_reg_mem("rax", "rbx", 8 * idx)
        prog.call("rax")
        if dst != 0:
            prog.movapd(dst, 0)


class IfElse:
    def __init__(self, cond, true_val, false_val):
        self.cond = cond
        self.true_val = true_val
        self.false_val = false_val
        self.is_pure = cond.is_pure and true_val.is_pure and false_val.is_pure

    def __repr__(self):
        return f"IfElse[{self.is_pure}]({self.cond}, {self.true_val}, {self.false_val})"

    def compile(self, dst, prog, mem, vt):
        sp = mem.push()
        use_reg_cond = sp < 3 and self.true_val.is_pure and self.false_val.is_pure

        if use_reg_cond:
            self.cond.compile(3 + sp, prog, mem, vt)
        else:
            self.cond.compile(0, prog, mem, vt)
            prog.movsd_mem_xmm("rsp", 8 * sp, 0)

        sp = mem.push()
        use_reg_false = sp < 3 and self.true_val.is_pure

        if use_reg_false:
            self.false_val.compile(3 + sp, prog, mem, vt)
        else:
            self.false_val.compile(0, prog, mem, vt)
            prog.movsd_mem_xmm("rsp", 8 * sp, 0)

        self.true_val.compile(0, prog, mem, vt)

        sp = mem.pop()

        if use_reg_false:
            f = 3 + sp
        else:
            prog.movsd_xmm_mem(1, "rsp", 8 * sp)
            f = 1

        sp = mem.pop()

        if use_reg_cond:
            c = 3 + sp
        else:
            prog.movsd_xmm_mem(2, "rsp", 8 * sp)
            c = 2

        prog.vandpd(0, 0, c)
        prog.vandnpd(c, c, f)
        prog.vorpd(dst, 0, c)


class Const:
    def __init__(self, val):
        self.val = float(val)
        self.is_pure = True

    def __repr__(self):
        return f"Const({self.val})"

    def compile(self, dst, prog, mem, vt):
        prog.movsd_xmm_mem(dst, "rbp", 8 * mem.constant(self.val))


class Var:
    def __init__(self, name):
        self.name = str(name)
        self.is_pure = True

    def __repr__(self):
        return f"Var('{self.name}')"

    def compile(self, dst, prog, mem, vt):
        prog.movsd_xmm_mem(dst, "rbp", 8 * mem.index(self.name))


class Eq:
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return f"{self.lhs} = {self.rhs}"

    def compile(self, dst, prog, mem, vt):
        self.rhs.compile(dst, prog, mem, vt)
        prog.movsd_mem_xmm("rbp", 8 * mem.index(self.lhs.name), dst)


class EqDiff:
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return f"δ{self.lhs} = {self.rhs}"

    def compile(self, dst, prog, mem, vt):
        self.rhs.compile(dst, prog, mem, vt)
        prog.movsd_mem_xmm("rbp", 8 * mem.index_diff(self.lhs.name), dst)


class Model:
    def __init__(self, iv, states, params, obs, eqs, odes):
        self.iv = iv
        self.states = states
        self.params = params
        self.obs = obs
        self.odes = odes
        self.eqs = eqs

    def __repr__(self):
        return f"""Model(
            iv: {self.iv}
            states: {self.states}
            params: {self.params}
            obs: {self.obs}
            eqs: {self.eqs}
            odes: {self.odes})
        )"""

    def compile(self, dst, prog, mem, vt):
        for eq in self.eqs:
            eq.compile(dst, prog, mem, vt)

        for eq in self.odes:
            eq.compile(dst, prog, mem, vt)

        # note that we generate the prologue after the main body
        # because we need to know the stack size
        prog.begin_prepend()
        self.prologue(prog, mem.stack_size * 8)
        prog.end_prepend()
        self.epilogue(prog, mem.stack_size * 8)

    def prologue(self, prog, n):
        prog.push("rbp")
        prog.push("rbx")
        prog.push("rax")
        prog.mov("rbp", "rdi")
        prog.mov("rbx", "rsi")  # different than the rust code
        if n > 0:
            prog.sub_rsp(n)

    def epilogue(self, prog, n):
        prog.vzeroupper()
        if n > 0:
            prog.add_rsp(n)
        prog.pop("rax")
        prog.pop("rbx")
        prog.pop("rbp")
        prog.ret()


def is_pure(op):
    ops = [
        "plus",
        "minus",
        "times",
        "division",
        "square",
        "cube",
        "root",
        "recip",
        "abs",
        "neg",
        "lt",
        "leq",
        "gt",
        "geq",
        "eq",
        "neq",
        "and",
        "or",
        "xor",
    ]
    return op in ops


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
        assert len(model.obs) == len(model.eqs)
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


# ******************** Lowering ***********************


def lower(y):
    if y.is_Number or isinstance(y, numbers.Number):
        return Const(y)
    elif y.is_Symbol:
        return Var(y)
    elif y.is_Relational:
        return relational(y)
    elif y.is_Boolean:
        return boolean(y)
    elif y.is_Piecewise:
        return piecewise(y)
    else:
        return lower_tree(y)


def lower_tree(y):
    if y.is_Add:
        return lower_add(y)
    elif y.is_Mul:
        return lower_mul(y)
    elif y.is_Pow:
        return lower_pow(y)
    elif y.is_Function:
        if len(y.args) == 1:
            return Unary(operation(y.func), lower(y.args[0]))
        elif len(y.args) == 2:
            return Binary(operation(y.func), lower(y.args[0]), lower(y.args[1]))
    raise ValueError(f"unrecognized expression: {y}")


def lower_add(y):
    assert y.is_Add
    if len(y.args) == 1:
        return lower(y.args[0])
    if len(y.args) == 2:
        return Binary("plus", lower(y.args[0]), lower(y.args[1]))
    else:
        return Binary("plus", lower(y.args[0]), lower(y.func(*y.args[1:])))


def lower_mul(y):
    assert y.is_Mul
    if len(y.args) == 2 and y.args[1].is_Pow and y.args[1].args[1] == -1:
        return Binary("divide", lower(y.args[0]), lower(y.args[1].args[0]))
    if len(y.args) == 2:
        return Binary("times", lower(y.args[0]), lower(y.args[1]))
    else:
        return Binary("times", lower(y.args[0]), lower(y.func(*y.args[1:])))


def lower_pow(y):
    assert y.is_Pow
    power = y.args[1]
    arg = y.args[0]

    if power == 2:
        return Unary("square", lower(arg))
    elif power == 3:
        return Unary("cube", lower(arg))
    elif power == -1:
        return Unary("recip", lower(arg))
    elif power == -2:
        return Unary("recip", lower(arg**2))
    elif power == -3:
        return Unary("recip", lower(arg**3))
    elif power == Rational(1, 2):
        return Unary("root", lower(arg))
    elif power == Rational(3, 2):
        return Unary("root", lower(arg**3))
    else:
        return Binary("power", lower(y.args[0]), lower(y.args[1]))


def operation(func):
    op = str(func)
    if func == sqrt:
        op = "root"
    elif func == log:
        op = "ln"  # this is confusing but sympy uses `log` for natural logarithm
    elif func == Abs:
        op = "abs"
    elif func == asin:
        op = "arcsin"
    elif func == acos:
        op = "arccos"
    elif func == atan:
        op = "arctan"
    elif func == asinh:
        op = "arcsinh"
    elif func == acosh:
        op = "arccosh"
    elif func == atanh:
        op = "arctanh"

    return op


def relational(y):
    f = y.func
    op = ""

    if f == LessThan:
        op = "lt"
    elif f == StrictLessThan:
        op = "leq"
    elif f == GreaterThan:
        op = "gt"
    elif f == StrictGreaterThan:
        op = "geq"
    elif f == Equality:
        op = "eq"
    elif f == Unequality:
        op = "neq"
    else:
        raise ValueError("unrecognized relational operator")

    return Binary(op, lower(y.args[0]), lower(y.args[1]))


def boolean(y):
    f = y.func
    op = ""

    if f == And:
        op = "and"
    elif f == Or:
        op = "or"
    elif f == Xor:
        op = "xor"
    else:
        raise ValueError("unrecognized boolean operator")

    return Binary(op, lower(y.args[0]), lower(y.args[1]))


def piecewise(y):
    cond = y.args[0][1]
    x1 = y.args[0][0]

    if len(y.args) == 1:
        return expr(x1)
    if len(y.args) == 2:
        x2 = y.args[1][0]
    else:
        x2 = piecewise(*y.args[1:])

    return IfElse(lower(cond), lower(x1), lower(x2))


def model(states, eqs, params=None, obs=None):
    if not isinstance(states, list):
        states = [states]

    if not isinstance(eqs, list):
        eqs = [eqs]

    if params is None:
        params = []

    if obs is None:
        obs = [f"${i}" for i in range(len(eqs))]

    model = Model(
        Var("$_"),  # iv
        [Var(x) for x in states],  # states
        [Var(x) for x in params],  # params
        [Var(x) for x in obs],  # obs
        [Eq(Var(lhs), lower(rhs)) for (lhs, rhs) in zip(obs, eqs)],  # eqs
        [],  # odes
    )

    return model
