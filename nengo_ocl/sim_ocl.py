import os
import time
import numpy as np
import pyopencl as cl

import sim_npy
from .raggedarray import RaggedArray
from .clraggedarray import CLRaggedArray
from .clra_gemv import plan_ragged_gather_gemv
from .clra_nonlinearities import plan_lif, plan_lif_rate, plan_probes
from .plan import Plan, Prog, PythonPlan

from raggedarray import RaggedArray
from clra_gemv import plan_ragged_gather_gemv
from clra_nonlinearities import plan_lif, plan_lif_rate
from plan import Plan, Prog

class Simulator(sim_npy.Simulator):

    def RaggedArray(self, *args, **kwargs):
        val = RaggedArray(*args, **kwargs)
        if len(val.buf) == 0:
            return None
        else:
            return CLRaggedArray(self.queue, val)

    def __init__(self, model, context, n_prealloc_probes=1000,
                 profiling=None):
        if profiling is None:
            profiling = int(os.getenv("NENGO_OCL_PROFILING", 0))
        self.context = context
        self.profiling = profiling
        if profiling:
            self.queue = cl.CommandQueue(
                context,
                properties=cl.command_queue_properties.PROFILING_ENABLE)
        else:
            self.queue = cl.CommandQueue(context)
        self.n_prealloc_probes = n_prealloc_probes
        sim_npy.Simulator.__init__(self,
                                   model,
                                   )
        if all(isinstance(p, Plan) for p in self._plan):
            self._prog = Prog(self._plan)
        else:
            self._prog = None

    def _prep_all_data(self):
        # -- replace the numpy-allocated RaggedArray with OpenCL one
        self.all_data = CLRaggedArray(self.queue, self.all_data)

    def plan_ragged_gather_gemv(self, *args, **kwargs):
        return plan_ragged_gather_gemv(self.queue, *args, **kwargs)

    def plan_direct(self, nls):
        ### TODO: this is sub-optimal, since it involves copying everything
        ### off the device, running the nonlinearity, then copying back on
        sidx = self.sidx
        def direct():
            for nl in nls:
                J = self.all_data[sidx[nl.input_signal]]
                output = nl.fn(J)
                self.all_data[sidx[nl.output_signal]] = output
        return PythonPlan(direct, name="direct", tag="direct")

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

    def plan_SimLIFRate(self, ops):
        J = self.all_data[[self.sidx[op.J] for op in ops]]
        R = self.all_data[[self.sidx[op.output] for op in ops]]
        ref = self.RaggedArray([op.nl.tau_ref for op in ops])
        tau = self.RaggedArray([op.nl.tau_rc for op in ops])
        dt = self.model.dt
        return [plan_lif_rate(self.queue, J, R, ref, tau, dt,
                              tag="lif_rate", n_elements=10)]

    def plan_probes(self):
        if len(self.model.probes) > 0:
            n_prealloc = self.n_prealloc_probes

            probes = self.model.probes
            periods = [int(np.round(float(p.dt) / self.model.dt))
                       for p in probes]

            sim_step = self.all_data[[self.sidx[self.model.steps]]]
            X = self.all_data[[self.sidx[p.sig] for p in probes]]
            Y = self.RaggedArray(
                [np.zeros((n_prealloc, p.sig.shape[0])) for p in probes])

            cl_plan = plan_probes(self.queue, sim_step, periods, X, Y,
                                  tag="probes")
            self._max_steps_between_probes = n_prealloc * min(periods)
            cl_plan.Y = Y
            self._cl_probe_plan = cl_plan
            return [cl_plan]
        else:
            return []

    def drain_probe_buffers(self):
        self.queue.finish()
        with sim_npy.Timer('drain_probes', enabled=True):
            plan = self._cl_probe_plan
            bufpositions = plan.cl_bufpositions.get()
            for i, probe in enumerate(self.model.probes):
                n_buffered = bufpositions[i]
                if n_buffered:
                    self.probe_outputs[probe].extend(plan.Y[i][:n_buffered])
            plan.cl_bufpositions.fill(0)
            self.queue.finish()


    def print_profiling(self):
        print '-' * 80
        print '%s\t%s\t%s\t%s' % (
            'n_calls', 'runtime', 'q-time', 'subtime')
        time_running_kernels = 0.0
        for p in self._plan:
            if isinstance(p, Plan):
                print '%i\t%2.3f\t%2.3f\t%2.3f\t<%s, tag=%s>' % (
                    p.n_calls, sum(p.ctimes), sum(p.btimes), sum(p.atimes),
                    p.name, p.tag)
                time_running_kernels += sum(p.ctimes)
            else:
                print p, getattr(p, 'cumtime', '<unknown>')
        print '-' * 80
        print 'totals:\t%2.3f\t%2.3f\t%2.3f' % (
            time_running_kernels, 0.0, 0.0)

    def step(self):
        return self.run_steps(1)

    def run_steps(self, N, verbose=False):
        for fn in self._plan:
            fn()
        self.drain_probe_buffers()
        self.queue.finish()
        with sim_npy.Timer('run_steps', enabled=True):
            profiling = self.profiling
            # -- precondition: the probe buffers have been drained
            bufpositions = self._cl_probe_plan.cl_bufpositions.get()
            assert np.all(bufpositions == 0)
            # -- we will go through N steps of the simulator
            #    in groups of up to B at a time, draining
            #    the probe buffers after each group of B
            while N:
                B = min(N, self._max_steps_between_probes)
                if self._prog is None:
                    for bb in xrange(B):
                        for fn in self._plan:
                            fn(profiling)
                        self.sim_step += 1
                else:
                    self._prog.call_n_times(B, self.profiling)
                self.drain_probe_buffers()
                N -= B
        if self.profiling > 1:
            self.print_profiling()


    def probe_data(self, probe):
        return np.vstack(self.probe_outputs[probe])
