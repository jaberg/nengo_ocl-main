import os
import time
import numpy as np
import pyopencl as cl

import sim_npy
from .raggedarray import RaggedArray
from .clraggedarray import CLRaggedArray
from .clra_gemv import plan_ragged_gather_gemv
from .clra_nonlinearities import plan_lif, plan_lif_rate, plan_probes
from .plan import BasePlan, PythonPlan, Plan, Prog

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
            #print 'n_prealloc', n_prealloc

            probes = self.model.probes
            periods = [int(np.round(float(p.dt) / self.model.dt))
                       for p in probes]
            #print 'model dt', self.model.dt
            #print [p.dt for p in probes]
            #print 'periods', periods
            for p in probes:
                if p.sig.size != p.sig.shape[0]:
                    raise NotImplementedError('probing non-vector', p)


            X = self.all_data[[self.sidx[p.sig] for p in probes]]
            Y = self.RaggedArray(
                [np.zeros((n_prealloc, p.sig.shape[0])) for p in probes])

            cl_plan = plan_probes(self.queue, periods, X, Y, tag="probes")
            self._max_steps_between_probes = n_prealloc * min(periods)
            #print 'max inter steps', self._max_steps_between_probes
            cl_plan.Y = Y
            self._cl_probe_plan = cl_plan
            return [cl_plan]
        else:
            return []

    def drain_probe_buffers(self):
        self.queue.finish()
        plan = self._cl_probe_plan
        bufpositions = plan.cl_bufpositions.get()
        for i, probe in enumerate(self.model.probes):
            n_buffered = bufpositions[i]
            if n_buffered:
                # XXX: this syntax retrieves *ALL* of Y from the device
                #      because the :n_buffered only works on the ndarray
                #      *after* it has been transferred.
                self.probe_outputs[probe].extend(plan.Y[i][:n_buffered])
        plan.cl_bufpositions.fill(0)
        self.queue.finish()


    def print_profiling(self, sort=None):
        """
        Parameters
        ----------
        sort : indicates the column to sort by (negative number sorts ascending)
            (0 = n_calls, 1 = runtime, 2 = q-time, 3 = subtime)
        """
        ### make and sort table
        table = []
        unknowns = []
        for p in self._plan:
            if isinstance(p, BasePlan):
                table.append(
                    (p.n_calls, sum(p.ctimes), sum(p.btimes), sum(p.atimes),
                    p.name, p.tag))
            else:
                unknowns.append((str(p), getattr(p, 'cumtime', '<unknown>')))

        if sort is not None:
            reverse = sort >= 0
            table.sort(key=lambda x: x[abs(sort)], reverse=reverse)

        ### printing
        print '-' * 80
        print '%s\t%s\t%s\t%s' % ('n_calls', 'runtime', 'q-time', 'subtime')

        for r in table:
            print '%i\t%2.3f\t%2.3f\t%2.3f\t<%s, tag=%s>' % r

        print '-' * 80
        col_sum = lambda c: sum(map(lambda x: x[c], table))
        print 'totals:\t%2.3f\t%2.3f\t%2.3f' % (
            col_sum(1), col_sum(2), col_sum(3))

        if len(unknowns) > 0:
            print
            for r in unknowns:
                print "%s %s" % r

    def step(self):
        return self.run_steps(1)

    def run_steps(self, N, verbose=False):
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
