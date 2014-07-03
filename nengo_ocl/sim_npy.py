"""
numpy Simulator in the style of the OpenCL one, to get design right.
"""
from collections import defaultdict
import numpy as np

from nengo.objects import LIF, LIFRate

from raggedarray import shape0
from raggedarray import shape1
from raggedarray import RaggedArray
from raggedarray import ragged_gather_gemv

def isview(obj):
    return obj.base is not obj


def lif_step(dt, J, voltage, refractory_time, spiked, tau_ref, tau_rc,
             upsample):
    """
    Replicates nengo.nonlinear.LIF's step_math0 function, except for
    a) not requiring a LIF instance and
    b) not adding the bias
    """
    if upsample != 1:
        raise NotImplementedError()

    # Euler's method
    dV = dt / tau_rc * (J - voltage)

    # increase the voltage, ignore values below 0
    v = np.maximum(voltage + dV, 0)

    # handle refractory period
    post_ref = 1.0 - (refractory_time - dt) / dt

    # set any post_ref elements < 0 = 0, and > 1 = 1
    v *= np.clip(post_ref, 0, 1)

    # determine which neurons spike
    # if v > 1 set spiked = 1, else 0
    spiked[...] = (v > 1) * 1.0

    old = np.seterr(all='ignore')
    try:

        # linearly approximate time since neuron crossed spike threshold
        overshoot = (v - 1) / dV
        spiketime = dt * (1.0 - overshoot)

        # adjust refractory time (neurons that spike get a new
        # refractory time set, all others get it reduced by dt)
        new_refractory_time = spiked * (spiketime + tau_ref) \
                              + (1 - spiked) * (refractory_time - dt)
    finally:
        np.seterr(**old)

    # return an ordered dictionary of internal variables to update
    # (including setting a neuron that spikes to a voltage of 0)

    voltage[:] = v * (1 - spiked)
    refractory_time[:] = new_refractory_time


def idxs(seq, offset):
    rval = dict((s, i + offset) for (i, s) in enumerate(seq))
    return rval, offset + len(rval)


def islif(obj):
    return isinstance(obj, LIF)


def islifrate(obj):
    return isinstance(obj, LIFRate)


class Simulator(object):
    def __init__(self, model, n_prealloc_probes=1000):
        self.model = model
        self.nonlinearities = sorted(self.model.nonlinearities)

        self.bias_signals = [nl.bias_signal for nl in self.nonlinearities]
        self.input_signals = [nl.input_signal for nl in self.nonlinearities]
        self.output_signals = [nl.output_signal for nl in self.nonlinearities]

        bias_signals_set = set(self.bias_signals)

        self.const_signals = [sig for sig in model.signals
                if hasattr(sig, 'value') and not sig in bias_signals_set]

        self.all_signals = self.bias_signals + self.const_signals
        self.all_signals.extend(self.input_signals)
        self.all_signals.extend(self.output_signals)
        all_signals_set = set(self.all_signals)
        self.vector_signals = [sig for sig in model.signals
                if sig not in all_signals_set]
        self.all_signals.extend(self.vector_signals)

        self.all_signals_set = set(self.all_signals)

        assert len(self.all_signals_set) == len(self.all_signals)

        self.all_signal_idxs, offset = idxs(self.all_signals, 0)
        assert offset == len(model.signals)

        # -- append duplicate vector_signals
        #    which are necessary for transforms
        self.tmp_vector_idxs, offset = idxs(self.vector_signals, offset)
        self.all_signals.extend(self.vector_signals)

        self.n_prealloc_probes = n_prealloc_probes
        self.sim_step = 0

    def RaggedArray(self, *args, **kwargs):
        return RaggedArray(*args, **kwargs)

    def alloc_signals(self):
        zeros = np.zeros
        self.all_data_A = self.RaggedArray(
            [zeros(s.shape) + getattr(s, 'value', zeros(s.shape))
                for s in self.all_signals],
            names=[getattr(s, 'name', '') for s in self.all_signals])
        self.all_data_B = self.RaggedArray(
            [zeros(s.shape) + getattr(s, 'value', zeros(s.shape))
                for s in self.all_signals],
            names=[getattr(s, 'name', '') for s in self.all_signals])

        # -- by default all of the signals are
        #    column vector "views" of themselves
        starts = []
        shape0s = []
        shape1s = []
        ldas = []
        names = []
        orig_len = len(self.all_signals)
        base_starts = self.all_data_A.starts

        # -- add signalviews that arise in filters, transforms,
        #    encoders, and decoders.
        def _add_view(obj, sidx):
            if not isview(obj):
                return
            idx = sidx[obj.base]
            starts.append(base_starts[idx] + obj.offset)
            shape0s.append(shape0(obj))
            shape1s.append(shape1(obj))
            if obj.ndim == 0:
                ldas.append(0)
            elif obj.ndim == 1:
                ldas.append(obj.shape[0])
            elif obj.ndim == 2:
                # -- N.B. the original indexing was
                #    based on ROW-MAJOR storage, and
                #    this simulator uses COL-MAJOR storage
                ldas.append(obj.elemstrides[0])
            else:
                raise NotImplementedError()
            names.append(getattr(obj, 'name', ''))

            rval = len(names) - 1 + orig_len
            sidx[obj] = rval
            return rval

        def add_view(obj):
            _add_view(obj, self.all_signal_idxs)
            if obj.base in self.tmp_vector_idxs:
                _add_view(obj, self.tmp_vector_idxs)

        for filt in self.model.filters:
            add_view(filt.alpha_signal)
            add_view(filt.oldsig)
            add_view(filt.newsig)

        for tf in self.model.transforms:
            add_view(tf.alpha_signal)
            add_view(tf.insig)
            add_view(tf.outsig)

        for enc in self.model.encoders:
            #add_view(tf.weights) # TODO for learning
            add_view(enc.sig)
            add_view(enc.pop.input_signal)

        for dec in self.model.decoders:
            #add_view(tf.weights) # TODO for learning
            add_view(dec.sig)
            add_view(dec.pop.input_signal)

        self.all_data_A.add_views(starts, shape0s, shape1s, ldas, names)
        self.all_data_B.add_views(starts, shape0s, shape1s, ldas, names)

    def alloc_probes(self):
        probes = self.model.probes
        dts = list(sorted(set([probe.dt for probe in probes])))
        self.probes_by_period = {}
        self.probe_output = {}
        for dt in dts:
            sp_dt = [probe for probe in probes if probe.dt == dt]
            period = int(dt // self.model.dt)
            self.probes_by_period[period] = sp_dt
        for probe in probes:
            self.probe_output[probe] = []

    def alloc_populations(self):
        nls = self.nonlinearities
        sidx = self.all_signal_idxs
        pop_J_idxs = [sidx[nl.input_signal] for nl in nls]
        pop_bias_idxs = [sidx[nl.bias_signal] for nl in nls]
        pop_output_idxs = [sidx[nl.output_signal] for nl in nls]

        self.pop_bias = self.all_data_A[pop_bias_idxs]
        self.pop_J = self.all_data_B[pop_J_idxs]
        self.pop_output = self.all_data_B[pop_output_idxs]
        self.pidx, _ = idxs(self.nonlinearities, 0)

        lif_idxs = []
        direct_idxs = []
        for i, p in enumerate(nls):
            if islif(p):
                lif_idxs.append(i)
            else:
                direct_idxs.append(i)

        self.pop_lif_idxs = lif_idxs
        self.pop_direct_idxs = direct_idxs

        if lif_idxs:
            # sorting in the constructor was supposed to ensure this:
            assert lif_idxs == range(lif_idxs[0], lif_idxs[0] + len(lif_idxs))

            tau_rcs = set([nls[li].tau_rc for li in lif_idxs])
            tau_refs = set([nls[li].tau_ref for li in lif_idxs])
            if len(tau_rcs) > 1 or len(tau_refs) > 1:
                raise NotImplementedError('Non-homogeneous lif population')

            self.pop_lif_rep = nls[lif_idxs[0]]
            #self.pop_lif_J = self.pop_J.view1d(lif_idxs)
            #self.pop_lif_output = self.pop_output.view1d(lif_idxs)

            self.pop_lif_voltage = self.RaggedArray(
                [np.zeros(nls[idx].n_neurons) for idx in lif_idxs])
            self.pop_lif_reftime = self.RaggedArray(
                [np.zeros(nls[idx].n_neurons) for idx in lif_idxs])

        if direct_idxs:
            self.pop_direct_ins = self.pop_J[direct_idxs]
            self.pop_direct_outs = self.pop_output[direct_idxs]
            dtype = self.pop_direct_ins.buf.dtype
            # N.B. This uses numpy-based RaggedArray specifically.
            self.pop_direct_ins_npy = RaggedArray(
                [np.zeros(nls[di].n_in, dtype=dtype) for di in direct_idxs])
            self.pop_direct_outs_npy = RaggedArray(
                [np.zeros(nls[di].n_out, dtype=dtype) for di in direct_idxs])

    def alloc_transforms(self):
        sidx = self.all_signal_idxs
        tmpidx = self.tmp_vector_idxs
        transforms = self.model.transforms
        tidx, _ = idxs(transforms, 0)
        tf_by_outsig = defaultdict(list)
        for tf in transforms:
            tf_by_outsig[tf.outsig].append(tf)

        A_js = []
        X_js = []
        for sig in self.vector_signals:
            A_js_i = []
            X_js_i = []
            for tf in tf_by_outsig[sig]:
                if tf.alpha.size == 1:
                    A_js_i.append(tmpidx[tf.insig])
                    X_js_i.append(sidx[tf.alpha_signal])
                else:
                    A_js_i.append(sidx[tf.alpha_signal])
                    X_js_i.append(tmpidx[tf.insig])
            A_js.append(A_js_i)
            X_js.append(X_js_i)

        self.tf_A_js = self.RaggedArray(A_js)
        self.tf_X_js = self.RaggedArray(X_js)

        dst_signal_idxs = [sidx[v] for v in self.vector_signals]
        self.tf_Y = self.all_data_B[dst_signal_idxs]

    def alloc_filters(self):
        sidx = self.all_signal_idxs
        filters = self.model.filters
        f_by_newsig = defaultdict(list)
        for f in filters:
            f_by_newsig[f.newsig].append(f)

        A_js = []
        X_js = []
        for sig in self.vector_signals:
            A_js_i = []
            X_js_i = []
            for f in f_by_newsig[sig]:
                if f.alpha.size == 1:
                    A_js_i.append(sidx[f.oldsig])
                    X_js_i.append(sidx[f.alpha_signal])
                else:
                    A_js_i.append(sidx[f.alpha_signal])
                    X_js_i.append(sidx[f.oldsig])
            A_js.append(A_js_i)
            X_js.append(X_js_i)

        self.f_A_js = self.RaggedArray(A_js)
        self.f_X_js = self.RaggedArray(X_js)

        dst_signal_idxs = [sidx[v] for v in self.vector_signals]
        self.f_Y = self.all_data_B[dst_signal_idxs]

    def alloc_encoders(self):
        encoders = self.model.encoders
        sidx = self.all_signal_idxs

        # -- which encoder(s) does each population use
        self.enc_A_js = self.RaggedArray([
            [sidx[enc.weights_signal] for enc in encoders if enc.pop == pop]
            for pop in self.model.nonlinearities])

        # -- and which corresponding signal does it encode
        self.enc_X_js = self.RaggedArray([
            [sidx[enc.sig] for enc in encoders if enc.pop == pop]
            for pop in self.model.nonlinearities])

        Yidxs = [sidx[pop.input_signal] for pop in self.model.nonlinearities]
        self.enc_Y = self.all_data_B[Yidxs]

    def alloc_decoders(self):
        decoders = self.model.decoders
        sidx = self.all_signal_idxs

        # -- which decoder(s) does each signal use
        self.dec_A_js = self.RaggedArray([
            [sidx[dec.weights_signal] for dec in decoders if dec.sig == sig]
            for sig in self.vector_signals])

        # -- and which corresponding population does it decode
        self.dec_X_js = self.RaggedArray([
            [sidx[dec.pop.output_signal]
                for dec in decoders if dec.sig == sig]
            for sig in self.vector_signals])

        dst_idxs = [self.tmp_vector_idxs[sig]
            for sig in self.vector_signals]
        self.dec_Y = self.all_data_B[dst_idxs]

    def alloc_all(self):
        self.alloc_signals()
        self.alloc_probes()
        self.alloc_populations()
        self.alloc_transforms()
        self.alloc_filters()
        self.alloc_encoders()
        self.alloc_decoders()

    def do_transforms(self):
        """
        Combine the elements of input accumulator buffer (sigs_ic)
        into *add* them into sigs
        """
        # ADD linear combinations of signals to some signals
        ragged_gather_gemv(
            Ms=self.all_data_A.shape0s,
            Ns=self.all_data_A.shape1s,
            alpha=1.0,
            A=self.all_data_B,
            A_js=self.tf_A_js,
            X=self.all_data_B,
            X_js=self.tf_X_js,
            beta=1.0,
            Y=self.tf_Y,
            )

    def do_filters(self):
        """
        Recombine the elements of previous signal buffer (sigs)
        and write them back to `sigs`
        """
        # ADD linear combinations of signals to some signals
        ragged_gather_gemv(
            Ms=self.all_data_A.shape0s,
            Ns=self.all_data_A.shape1s,
            alpha=1.0,
            A=self.all_data_A,
            A_js=self.f_A_js,
            X=self.all_data_A,
            X_js=self.f_X_js,
            beta=0.0,
            Y=self.f_Y,
            )

    def do_encoders(self):
        ragged_gather_gemv(
            Ms=self.all_data_A.shape0s,
            Ns=self.all_data_A.shape1s,
            alpha=1.0,
            A=self.all_data_A, A_js=self.enc_A_js,
            X=self.all_data_A, X_js=self.enc_X_js,
            beta=1.0, Y_in=self.pop_bias,
            Y=self.enc_Y,
            )

    def do_populations(self):
        dt = self.model.dt

        if self.pop_lif_idxs:
            J = self.pop_J.view1d(self.pop_lif_idxs)
            out = self.pop_output.view1d(self.pop_lif_idxs)
            lif_step(
                dt=dt,
                J=J.T,
                voltage=self.pop_lif_voltage.buf,
                refractory_time=self.pop_lif_reftime.buf,
                spiked=out.T,
                tau_ref=self.pop_lif_rep.tau_ref,
                tau_rc=self.pop_lif_rep.tau_rc,
                upsample=self.pop_lif_rep.upsample
                )

        for ii in self.pop_direct_idxs:
            nl = self.nonlinearities[ii]
            popidx = self.pidx[nl]
            self.pop_output[popidx][...] = nl.fn(self.pop_J[popidx])

    def do_decoders(self):
        ragged_gather_gemv(
            Ms=self.all_data_A.shape0s,
            Ns=self.all_data_A.shape1s,
            alpha=1.0,
            A=self.all_data_B,
            A_js=self.dec_A_js,
            X=self.all_data_B,
            X_js=self.dec_X_js,
            beta=0.0,
            Y=self.dec_Y,
            )

    def do_probes(self):
        for period in self.probes_by_period:
            if 0 == self.sim_step % period:
                for probe in self.probes_by_period[period]:
                    val = self.all_signal_idxs[probe.sig]
                    self.probe_output[probe].append(
                        self.all_data_B[val].copy())

    def do_all(self):
        #print '-' * 10 + 'A' * 10 + '-' * 10
        #print self.all_data_A
        #print '-' * 20
        self.do_encoders()          # encode sigs_t into pops
        #print self.pop_J
        self.do_populations()       # pop dynamics
        #print self.pop_J
        self.do_decoders()          # decode into sigs_ic
        #print self.pop_J
        self.do_filters()           # make sigs_t's contribution to t + 1
        #print self.pop_J
        self.do_transforms()        # add sigs_ic's contribution to t + 1
        #print self.pop_J
        self.do_probes()
        #print self.pop_J
        self.all_data_A.buf[:] = self.all_data_B.buf

        #print '-' * 10 + 'B' * 10 + '-' * 10
        #print self.all_data_A
        #print '-' * 20

    def step(self):
        self.do_all()
        self.sim_step += 1

    def run_steps(self, N, verbose=False):
        for i in xrange(N):
            self.step()

    def signal(self, sig):
        probes = [sp for sp in self.model.probes if sp.sig == sig]
        if len(probes) == 0:
            raise KeyError()
        elif len(probes) > 1:
            raise KeyError()
        else:
            return self.signal_probe_output(probes[0])

    def probe_data(self, probe):
        return np.asarray(self.probe_output[probe])

