"""
numpy Simulator in the style of the OpenCL one, to get design right.
"""
from collections import defaultdict
import math
import numpy as np

from simulator import lif_step


class RaggedArray(object):
    # a linear buffer that is partitioned into
    # sections of various lengths.
    # 
    def __init__(self, listoflists):

        starts = []
        lens = []
        buf = []

        for l in listoflists:
            starts.append(len(buf))
            lens.append(len(l))
            buf.extend(l)

        self.starts = starts
        self.lens = lens
        self.buf = np.asarray(buf)

    def shallow_copy(self):
        rval = self.__class__.__new__(self.__class__)
        rval.starts = self.starts
        rval.lens = self.lens
        rval.buf = self.buf
        return rval

    def add_view(self, start, length):
        assert start >= 0
        assert start + length <= len(self.buf)
        # -- creates copies, same semantics
        #    as OCL version
        self.starts = self.starts + [start]
        self.lens = self.lens + [length]
        return len(self.starts) - 1

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, item):
        o = self.starts[item]
        n = self.lens[item]
        rval = self.buf[o: o + n]
        if len(rval) != n:
            raise ValueError('buf not long enough')
        return rval

    def __setitem__(self, item, val):
        o = self.starts[item]
        n = self.lens[item]
        if len(val) != n:
            raise ValueError('wrong len')
        rval = self.buf[o: o + n]
        if len(rval) != n:
            raise ValueError('buf not long enough')
        self.buf[o: o + n] = val


def raw_ragged_gather_gemv(BB,
        Ns, alphas,
        A_starts, A_data,
        A_js_starts,
        A_js_lens,
        A_js_data,
        X_starts,
        X_data,
        X_js_starts,
        X_js_data,
        betas,
        Y_in_starts,
        Y_in_data,
        Y_starts,
        Y_lens,
        Y_data):
    for bb in xrange(BB):
        alpha = alphas[bb]
        beta = betas[bb]
        n_dot_products = A_js_lens[bb]
        y_offset = Y_starts[bb]
        y_in_offset = Y_in_starts[bb]
        M = Y_lens[bb]
        for mm in xrange(M):
            Y_data[y_offset + mm] = beta * Y_in_data[y_in_offset + mm]

        for ii in xrange(n_dot_products):
            x_i = X_js_data[X_js_starts[bb] + ii]
            a_i = A_js_data[A_js_starts[bb] + ii]
            N_i = Ns[ii]
            x_offset = X_starts[x_i]
            a_offset = A_starts[a_i]
            for mm in xrange(M):
                y_sum = 0.0
                for nn in xrange(N_i):
                    y_sum += X_data[x_offset + nn] * A_data[a_offset + nn * M + mm]
                Y_data[y_offset + mm] += alpha * y_sum


def ragged_gather_gemv(Ms, Ns, alpha, A, A_js, X, X_js,
                       beta, Y, Y_in=None):
    """
    """
    try:
        float(alpha)
        alpha = [alpha] * len(Y)
    except TypeError:
        pass

    try:
        float(beta)
        beta = [beta] * len(Y)
    except TypeError:
        pass

    if Y_in is None:
        Y_in = Y

    use_raw_fn = 1
    if use_raw_fn:
        # This is close to the OpenCL reference impl
        return raw_ragged_gather_gemv(
            len(Y),
            Ns,
            alpha,
            A.starts,
            A.buf,
            A_js.starts,
            A_js.lens,
            A_js.buf,
            X.starts,
            X.buf,
            X_js.starts,
            X_js.buf,
            beta,
            Y_in.starts,
            Y_in.buf,
            Y.starts,
            Y.lens,
            Y.buf)
    else:
        # -- less-close to the OpenCL impl
        #print alpha
        #print A.buf, 'A'
        #print X.buf, 'X'
        #print Y_in.buf, 'in'

        for i in xrange(len(Y)):
            try:
                y_i = beta[i] * Y_in[i]  # -- ragged getitem
            except:
                print i, beta, Y_in
                raise
            alpha_i = alpha[i]

            x_js_i = X_js[i] # -- ragged getitem
            A_js_i = A_js[i] # -- ragged getitem

            for xi, ai in zip(x_js_i, A_js_i):
                x_ij = X[xi] # -- ragged getitem
                M_i = Ms[ai]
                N_i = Ns[ai]
                assert N_i == len(x_ij)
                A_ij = A[ai].reshape(N_i, M_i) # -- ragged getitem
                #print xi, x_ij, A_ij
                try:
                    y_i += alpha_i * np.dot(x_ij, A_ij.reshape(N_i, M_i))
                except:
                    print i, xi, ai, A_ij, x_ij
                    raise

            Y[i] = y_i


def alloc_transform_helper(signals, transforms, sigs, sidx, RaggedArray,
        outsig_fn, insig_fn,
        ):
    tidx = dict((tf, i) for (i, tf) in enumerate(transforms))
    tf_by_outsig = defaultdict(list)
    for tf in transforms:
        tf_by_outsig[outsig_fn(tf)].append(tf)

    # N.B. the optimization below may not be valid
    # when alpha is not a scalar multiplier
    tf_weights = RaggedArray(
        [[tf.alpha] for tf in transforms])
    tf_Ns = [1] * len(transforms)
    tf_Ms = [1] * len(transforms)

    tf_sigs = sigs.shallow_copy()
    del sigs

    # -- which transform(s) does each signal use
    tf_weights_js = [[tidx[tf] for tf in tf_by_outsig[sig]]
        for sig in signals]

    # -- which corresponding(s) signal is transformed
    tf_signals_js = [[sidx[insig_fn(tf)] for tf in tf_by_outsig[sig]]
        for sig in signals]

    # -- Optimization:
    #    If any output signal is the sum of 
    #    consecutive weights with consecutive signals
    #    then turn that into *one* dot product
    #    of a new longer weight vector with a new
    #    longer signal vector.
    #    TODO: still do something like this for a 
    #    *part* of the wjs/sjs if possible
    #    TODO: sort the sjs or wjs to canonicalize things
    if 1:
      for ii, (wjs, sjs) in enumerate(
            zip(tf_weights_js, tf_signals_js)):
        assert len(wjs) == len(sjs)
        K = len(wjs)
        if len(wjs) <= 1:
            continue
        if wjs != range(wjs[0], wjs[0] + len(wjs)):
            continue
        if sjs != range(sjs[0], sjs[0] + len(sjs)):
            continue
        # -- precondition satisfied
        # XXX length should be the sum of the lenghts
        #     of the tf_weights[i] for i in wjs
        new_w_j = tf_weights.add_view(
            int(tf_weights.starts[wjs[0]]), K)
        tf_Ns.append(K)
        tf_Ms.append(1)
        new_s_j = tf_sigs.add_view(
            int(tf_sigs.starts[sjs[0]]), K)
        tf_weights_js[ii] = [new_w_j]
        tf_signals_js[ii] = [new_s_j]

    if 0:
        for ii, (wjs, sjs) in enumerate(
                zip(tf_weights_js, tf_signals_js)):
            if wjs:
                print wjs, sjs

    return locals()


class Simulator(object):
    def __init__(self, model, n_prealloc_probes=1000):
        self.model = model
        self.sidx = dict((s, i) for (i, s) in enumerate(model.signals))
        self.pidx = dict((p, i) for (i, p) in enumerate(model.populations))
        self.eidx = dict((s, i) for (i, s) in enumerate(model.encoders))
        self.didx = dict((p, i) for (i, p) in enumerate(model.decoders))
        self.n_prealloc_probes = n_prealloc_probes
        self.sim_step = 0

        if not all(s.n == 1 for s in self.model.signals):
            raise NotImplementedError()

    def RaggedArray(self, *args, **kwargs):
        return RaggedArray(*args, **kwargs)

    def alloc_signals(self):
        self.sigs_ic = self.RaggedArray([[0.0] for s in self.model.signals])

        self.sigs = self.RaggedArray([[getattr(s, 'value', 0.0)]
            for s in self.model.signals])

        # -- not necessary in ocl if signals fit into shared memory
        #    shared memory can be used as the copy in that case.
        self._sigs_copy = self.RaggedArray([[getattr(s, 'value', 0.0)]
            for s in self.model.signals])

    def alloc_signal_probes(self):
        signal_probes = self.model.signal_probes
        dts = list(sorted(set([sp.dt for sp in signal_probes])))
        self.sig_probes_output = {}
        self.sig_probes_buflen = {}
        self.sig_probes_Ms = self.RaggedArray([[1]])
        self.sig_probes_Ns = self.RaggedArray([[1]])
        self.sig_probes_A = self.RaggedArray([[1.0]])
        self.sig_probes_A_js = {}
        self.sig_probes_X_js = {}
        for dt in dts:
            sp_dt = [sp for sp in signal_probes if sp.dt == dt]
            period = int(dt // self.model.dt)

            # -- allocate storage for the probe output
            sig_probes = self.RaggedArray([[0.0] for sp in sp_dt])
            buflen = len(sig_probes.buf)
            sig_probes.buf = np.zeros(
                    buflen * self.n_prealloc_probes,
                    dtype=sig_probes.buf.dtype)
            self.sig_probes_output[period] = sig_probes
            self.sig_probes_buflen[period] = buflen
            self.sig_probes_A_js[period] = self.RaggedArray([[0] for sp in sp_dt])
            self.sig_probes_X_js[period] = self.RaggedArray(
                [[self.sidx[sp.sig]] for sp in sp_dt])

    def alloc_populations(self):
        def zeros():
            return self.RaggedArray(
                [[0.0] * p.n for p in self.model.populations])
        # -- lif-specific stuff
        self.pop_ic = zeros()
        self.pop_voltage = zeros()
        self.pop_rt = zeros()
        self.pop_output = zeros()
        self.pop_jbias = self.RaggedArray(
                [p.bias for p in self.model.populations])

    def alloc_transforms(self):
        stuff = alloc_transform_helper(
                self.model.signals,
                self.model.transforms,
                self.sigs_ic,
                self.sidx,
                self.RaggedArray,
                (lambda f: f.outsig),
                (lambda f: f.insig),
                )

        self.tf_weights_js = self.RaggedArray(stuff['tf_weights_js'])
        self.tf_signals_js = self.RaggedArray(stuff['tf_signals_js'])
        self.tf_weights = stuff['tf_weights']
        self.tf_signals = stuff['tf_sigs']
        self.tf_Ns = stuff['tf_Ns']
        self.tf_Ms = stuff['tf_Ms']

    def alloc_filters(self):
        stuff = alloc_transform_helper(
                self.model.signals,
                self.model.filters,
                self._sigs_copy,
                self.sidx,
                self.RaggedArray,
                (lambda f: f.newsig),
                (lambda f: f.oldsig),
                )

        self.f_signals = stuff['tf_sigs']
        self.f_weights = stuff['tf_weights']
        self.f_weights_js = self.RaggedArray(stuff['tf_weights_js'])
        self.f_signals_js = self.RaggedArray(stuff['tf_signals_js'])
        self.f_Ns = stuff['tf_Ns']
        self.f_Ms = stuff['tf_Ms']

    def alloc_encoders(self):
        encoders = self.model.encoders
        self.enc_weights = self.RaggedArray([enc.weights.flatten() for enc in encoders])

        self.enc_Ms = [enc.weights.shape[0] for enc in encoders]
        self.enc_Ns = [enc.weights.shape[1] for enc in encoders]

        # -- which encoder(s) does each population use
        self.enc_weights_js = self.RaggedArray([
            [self.eidx[enc] for enc in encoders if enc.pop == pop]
            for pop in self.model.populations])

        # -- and which corresponding signal does it encode
        self.enc_signals_js = self.RaggedArray([
            [self.sidx[enc.sig]
                for enc in encoders if enc.pop == pop]
            for pop in self.model.populations])

    def alloc_decoders(self):
        decoders = self.model.decoders
        self.dec_weights = self.RaggedArray([dec.weights.flatten() for dec in decoders])
        self.dec_Ms = [dec.weights.shape[0] for dec in decoders]
        self.dec_Ns = [dec.weights.shape[1] for dec in decoders]

        # -- which decoder(s) does each signal use
        self.dec_weights_js = self.RaggedArray([
            [self.didx[dec] for dec in decoders if dec.sig == sig]
            for sig in self.model.signals])

        # -- and which corresponding population does it decode
        self.dec_pops_js = self.RaggedArray([
            [self.pidx[dec.pop]
                for dec in decoders if dec.sig == sig]
            for sig in self.model.signals])

    def alloc_all(self):
        self.alloc_signals()
        self.alloc_signal_probes()
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
            Ms=self.tf_Ms,
            Ns=self.tf_Ns,
            alpha=1.0,
            A=self.tf_weights,
            A_js=self.tf_weights_js,
            X=self.tf_signals,
            X_js=self.tf_signals_js,
            beta=1.0,
            Y=self.sigs,
            )

    def do_filters(self):
        """
        Recombine the elements of previous signal buffer (sigs)
        and write them back to `sigs`
        """
        # ADD linear combinations of signals to some signals
        self._sigs_copy.buf[:] = self.sigs.buf
        ragged_gather_gemv(
            Ms=self.f_Ms,
            Ns=self.f_Ns,
            alpha=1.0,
            A=self.f_weights,
            A_js=self.f_weights_js,
            X=self.f_signals,
            X_js=self.f_signals_js,
            beta=0.0,
            Y=self.sigs,
            )

    def do_custom_transforms(self):
        for ct in self.model.custom_transforms:
            ipos = self.sidx[ct.insig]
            opos = self.sidx[ct.outsig]
            self.sigs[opos] = ct.func(self.sigs[ipos])

    def do_encoders(self):
        ragged_gather_gemv(
            Ms=self.enc_Ms,
            Ns=self.enc_Ns,
            alpha=1.0,
            A=self.enc_weights,
            A_js=self.enc_weights_js,
            X=self.sigs,
            X_js=self.enc_signals_js,
            beta=1.0,
            Y_in=self.pop_jbias,
            Y=self.pop_ic,
            )

    def do_populations(self):
        lif_step(
            J=self.pop_ic.buf,
            voltage=self.pop_voltage.buf,
            refractory_time=self.pop_rt.buf,
            spiked=self.pop_output.buf,
            dt=self.model.dt,
            tau_rc=0.02,
            tau_ref=0.002,
            upsample=1)

    def do_decoders(self):
        ragged_gather_gemv(
            Ms=self.dec_Ms,
            Ns=self.dec_Ns,
            alpha=1.0,
            A=self.dec_weights,
            A_js=self.dec_weights_js,
            X=self.pop_output,
            X_js=self.dec_pops_js,
            beta=0.0,
            Y=self.sigs_ic,
            )

    def do_probes(self):
        for period in self.sig_probes_output:
            if 0 == self.sim_step % period:
                probe_out = self.sig_probes_output[period]
                buflen = self.sig_probes_buflen[period]
                A_js = self.sig_probes_A_js[period]
                X_js = self.sig_probes_X_js[period]

                bufidx = int(self.sim_step // period)
                orig_buffer = probe_out.buf
                this_buffer = probe_out.buf[bufidx * buflen:]
                try:
                    probe_out.buf = this_buffer
                    ragged_gather_gemv(
                        Ms=self.sig_probes_Ms,
                        Ns=self.sig_probes_Ns,
                        alpha=1.0,
                        A=self.sig_probes_A,
                        A_js=A_js,
                        X=self.sigs,
                        X_js=X_js,
                        beta=0.0,
                        Y=probe_out,
                        )
                finally:
                    probe_out.buf = orig_buffer

    def do_all(self):
        # -- step 1: sigs store all signal values of step t (sigs_t)
        self.do_encoders()          # encode sigs_t into pops
        self.do_populations()       # pop dynamics
        self.do_decoders()          # decode into sigs_ic
        self.do_filters()           # make sigs_t's contribution to t + 1
        self.do_transforms()        # add sigs_ic's contribution to t + 1
        self.do_custom_transforms() # complete calculation of sigs of t + 1
        self.do_probes()

    def step(self):
        self.do_all()
        self.sim_step += 1

    def run_steps(self, N):
        for i in xrange(N):
            self.step()

    def signal(self, sig):
        probes = [sp for sp in self.model.signal_probes if sp.sig == sig]
        if len(probes) == 0:
            raise KeyError()
        elif len(probes) > 1:
            raise KeyError()
        else:
            return self.signal_probe_output(probes[0])

    def signal_probe_output(self, probe):
        period = int(probe.dt // self.model.dt)
        last_elem = int(math.ceil(self.sim_step / float(period)))

        # -- figure out which signal it is among the ones with the same dt
        sps_dt = [sp for sp in self.model.signal_probes if sp.dt == probe.dt]
        probe_idx = sps_dt.index(probe)
        all_rows = self.sig_probes_output[period].buf.reshape(
                (-1, self.sig_probes_buflen[period]))
        start = self.sig_probes_output[period].starts[probe_idx]
        olen = self.sig_probes_output[period].lens[probe_idx]
        return all_rows[:last_elem, start:start + olen]


