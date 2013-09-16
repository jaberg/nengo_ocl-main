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

    def __init__(self, context, model, n_prealloc_probes=1000,
                 profiling=False):
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

    def plan_SimLIFRate(self, nls):
        raise NotImplementedError()
        J = self.all_data[[self.sidx[nl.input_signal] for nl in nls]]
        R = self.all_data[[self.sidx[nl.output_signal] for nl in nls]]
        ref = self.RaggedArray([nl.tau_ref for nl in nls])
        tau = self.RaggedArray([nl.tau_rc for nl in nls])
        return plan_lif_rate(self.queue, J, R, ref, tau,
                             tag="lif_rate", n_elements=10)

    def plan_probes(self):
        if len(self.model.probes) > 0:
            buf_len = 1000

            probes = self.model.probes
            periods = [int(np.round(float(p.dt) / self.model.dt))
                       for p in probes]

            sim_step = self.all_data[[self.sidx[self.model.steps]]]
            P = self.RaggedArray(periods)
            X = self.all_data[[self.sidx[p.sig] for p in probes]]
            Y = self.RaggedArray(
                [np.zeros((p.sig.shape[0], buf_len)) for p in probes])

            cl_plan = plan_probes(self.queue, sim_step, P, X, Y, tag="probes")

            lengths = [period * buf_len for period in periods]
            def probe_copy_fn(profiling=False):
                if profiling: t0 = time.time()
                for i, length in enumerate(lengths):
                    ### use (sim_step + 1), since device sim_step is updated
                    ### at start of time step, and self.sim_step at end
                    if (self.sim_step + 1) % length == length - 1:
                        self.probe_outputs[probes[i]].append(Y[i].T)
                if profiling:
                    t1 = time.time()
                    probe_copy_fn.cumtime += t1 - t0
            probe_copy_fn.cumtime = 0.0

            self._probe_periods = periods
            self._probe_buffers = Y
            return [cl_plan, probe_copy_fn]
        else:
            return []

    def post_run(self):
        """Perform cleanup tasks after a run"""

        ### Copy any remaining probe data off device
        for i, probe in enumerate(self.model.probes):
            period = self._probe_periods[i]
            buffer = self._probe_buffers[i]
            pos = ((self.sim_step + 1) / period) % buffer.shape[1]
            if pos > 0:
                self.probe_outputs[probe].append(buffer[:,:pos].T)

        ### concatenate probe buffers
        for probe in self.model.probes:
            self.probe_outputs[probe] = np.vstack(self.probe_outputs[probe])

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

        import matplotlib.pyplot as plt
        for p in self._plan:
            if isinstance(p, Plan):
                plt.plot(p.btimes)
        plt.show()

    def run_steps(self, N, verbose=False):
        if self._prog is None:
            for i in xrange(N):
                self.step()
        else:
            self._prog.call_n_times(N, self.profiling)
        self.post_run()
