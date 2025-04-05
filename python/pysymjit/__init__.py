import numpy as np
import numbers

import pyengine
   
############################################################################
     
class Func:
    def __init__(self, compiler):
        self.compiler = compiler
        self.count_states = self.compiler.count_states
        self.count_params = self.compiler.count_params
        self.count_obs = self.compiler.count_obs        

    def __call__(self, *args):
        if len(args) > self.count_states:
            p = np.array(args[self.count_states:], dtype="double")
            self.compiler.params[:] = p
    
        if isinstance(args[0], numbers.Number):
            u = np.array(args[:self.count_states], dtype="double")
            self.compiler.states[:] = u
            self.compiler.execute()
            return self.compiler.obs.copy()
        else:
            return self.call_vectorized(*args)
            
    def call_vectorized(self, *args):
        assert(len(args) >= self.count_states)
        shape = args[0].shape
        n = args[0].size
        h = max(self.count_states, self.count_obs)
        buf = np.zeros((h, n), dtype="double")

        for i in range(self.count_states):
            assert(args[i].shape == shape)
            buf[i,:] = args[i].ravel()            
        
        self.compiler.execute_vectorized(buf)
        
        res = []
        for i in range(self.count_obs):            
            y = buf[i,:].reshape(shape)
            res.append(y)
            
        return res

    def dump(self, name, what="scalar"):
        self.compiler.dump(name, what=what)


class OdeFunc:
    def __init__(self, compiler):
        self.compiler = compiler

    def __call__(self, t, y, *args):
        y = np.array(y, dtype="double")
        self.compiler.states[:] = y

        if len(args) > 0:
            p = np.array(args, dtype="double")
            self.compiler.params[:] = p

        self.compiler.execute(t)
        return self.compiler.diffs.copy()
        
    def get_u0(self):
        return self.compiler.get_u0()

    def get_p(self):
        return self.compiler.get_p()
        
    def dump(self, name, what="scalar"):
        self.compiler.dump(name, what=what)        
        

class JacFunc:
    def __init__(self, compiler):
        self.compiler = compiler
        self.count_states = self.compiler.count_states

    def __call__(self, t, y, *args):
        y = np.array(y, dtype="double")
        self.compiler.states[:] = y

        if len(args) > 0:
            p = np.array(args, dtype="double")
            self.compiler.params[:] = p

        self.compiler.execute()
        jac = self.compiler.obs.copy()        
        return jac.reshape((self.count_states, self.count_states))
        
    def dump(self, name, what="scalar"):
        self.compiler.dump(name, what=what)        


def compile_func(states, eqs, params=None, obs=None, ty="arm", use_simd=True):
    if pyengine.can_compile():
        model = pyengine.tree.model(states, eqs, params, obs)
        compiler = pyengine.PyCompiler(model, ty=ty)
        return Func(compiler)
    else:
        raise ValueError("unsupported platform")


def compile_ode(iv, states, odes, params=None, ty="native", use_simd=False):
    if pyengine.can_compile():
        model = pyengine.tree.model_ode(iv, states, odes, params)
        compiler = pyengine.PyCompiler(model)
        return OdeFunc(compiler)
    else:
        raise ValueError("unsupported platform")
    
def compile_jac(iv, states, odes, params=None, ty="native", use_simd=False):
    if pyengine.can_compile():
        model = pyengine.tree.model_jac(iv, states, odes, params)
        compiler = pyengine.PyCompiler(model)
        return JacFunc(compiler)
    else:
        raise ValueError("unsupported platform")        


