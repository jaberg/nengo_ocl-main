
import os
import pyopencl as cl

import sim_npy
from clraggedarray import CLRaggedArray

from raggedarray import RaggedArray
from clra_gemv import plan_ragged_gather_gemv
from clra_nonlinearities import plan_lif, plan_lif_rate

class Simulator(sim_npy.Simulator):

    def RaggedArray(self, *args, **kwargs):
        val = RaggedArray(*args, **kwargs)
        if len(val.buf) == 0:
            return None
        else:
            return CLRaggedArray(self.queue, val)

    def __init__(self, context, model, n_prealloc_probes=1000,
                profiling=None):
        if profiling is None:
            profiling = bool(int(os.getenv("NENGO_OCL_PROFILING", 0)))
        self.context = context
        self.profiling = profiling
        if profiling:
            self.queue = cl.CommandQueue(
                context,
                properties=cl.command_queue_properties.PROFILING_ENABLE)
        else:
            self.queue = cl.CommandQueue(context)
        sim_npy.Simulator.__init__(self,
                                   model,
                                   )

    def _prep_all_data(self):
        # -- replace the numpy-allocated RaggedArray with OpenCL one
        self.all_data = CLRaggedArray(self.queue, self.all_data)

    def plan_ragged_gather_gemv(self, *args, **kwargs):
        return plan_ragged_gather_gemv(
            self.queue, *args, **kwargs)

    def plan_SimLIF(self, ops):
        J = self.all_data[[self.sidx[op.J] for op in ops]]
        V = self.all_data[[self.sidx[op.voltage] for op in ops]]
        W = self.all_data[[self.sidx[op.refractory_time] for op in ops]]
        S = self.all_data[[self.sidx[op.output] for op in ops]]
        ref = self.RaggedArray([op.nl.tau_ref for op in ops])
        tau = self.RaggedArray([op.nl.tau_rc for op in ops])
        dt = self.model.dt
        return [plan_lif(self.queue, J, V, W, V, W, S, ref, tau, dt,
                        tag="lif", upsample=1)]

    def plan_SimLIFRate(self, nls):
        raise NotImplementedError()
        #J = self.all_data[[self.sidx[nl.input_signal] for nl in nls]]
        #R = self.all_data[[self.sidx[nl.output_signal] for nl in nls]]
        #ref = self.RaggedArray([nl.tau_ref for nl in nls])
        #tau = self.RaggedArray([nl.tau_rc for nl in nls])
        #return plan_lif_rate(self.queue, J, R, ref, tau,
                             #tag="lif_rate", n_elements=10)

