import numpy as np
import pyopencl as cl
import time
from base import Model
from sim_ocl import Simulator

ctx = cl.create_some_context()

def test_mini(D=4, n_nets=1, neurons_per_product=3):
    # D is dimensionality of semantic pointers

    m = Model(.001)

    A = [m.signal() for ii in range(D)]
    B = [m.signal() for ii in range(D)]

    for ii in range(D):
        m.filter(1.0, A[ii], A[ii])
        m.filter(1.0, B[ii], B[ii])


    class CircularConvolution(object):
        # XXX actually create a Fourier matrix here
        fourier_matrix = np.random.randn(D, D)

        def __init__(self, model, A, B,
                neurons_per_product=128,
                neuron_type=None):
            D = len(A)
            # -- the output signals
            self.C = [m.signal() for ii in range(D)]

            # -- Fourier transforms of A, B, C
            self.AF = [m.signal() for ii in range(D)]
            self.BF = [m.signal() for ii in range(D)]
            self.CF = [m.signal() for ii in range(D)]

            # -- compute the Fourier transform of A and B
            #    as Filters, so populations are delayed by
            #    one time step.
            for jj in range(D):
                for ii in range(D):
                    alpha = self.fourier_matrix[ii, jj]
                    m.filter(alpha, A[ii], self.AF[jj])
            # TODO: make the optimization more robust
            #       that recognizes dot products of
            #       strided things
            for jj in range(D):
                for ii in range(D):
                    alpha = self.fourier_matrix[ii, jj]
                    m.filter(alpha, B[ii], self.BF[jj])

            # -- compute the complex elementwise product of A, B
            #    in Fourier domain
            for jj in range(4):
                for ii in range(D):
                    pop = m.population(neurons_per_product)
                    m.encoder(self.AF[ii], pop)
                    m.encoder(self.BF[ii], pop)
                    m.decoder(pop, self.CF[ii])

            # -- compute inverse Fourier transform of decoded
            #    products into output signal C
            for ii in range(D):
                for jj in range(D):
                    alpha = self.fourier_matrix[ii, jj]
                    m.transform(alpha, self.CF[jj], self.C[ii])

    for ii in range(n_nets):
        CircularConvolution(m, A, B,
            neurons_per_product=neurons_per_product)

    sim = Simulator(ctx, m)
    sim.alloc_all()
    sim.plan_all()

    t0 = time.time()
    sim.run_steps(100)
    t1 = time.time()

    print 'time', ((t1 - t0) * 10)

def test_large():
    return test_mini(
        D=500,
        n_nets=5,
        neurons_per_product=128)


