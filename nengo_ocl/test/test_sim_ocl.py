"""
Black-box testing of the sim_npy Simulator.

TestCase classes are added automatically from
nengo.tests.helpers.simulator_test_cases, but
you can still run individual test files like this:

    nosetests -sv test/test_sim_npy.py:TestSimulator.test_simple_direct_mode

"""

from nengo_ocl.tricky_imports import unittest

import nengo.tests.helpers

from nengo_ocl import sim_ocl

import pyopencl as cl
from nengo.tests.helpers import load_nengo_tests, simulator_test_suite

from nengo_ocl import sim_ocl


ctx = cl.create_some_context(interactive=False)

from nengo_ocl import sim_ocl

def simulator_allocator(*args, **kwargs):
    rval = sim_ocl.Simulator(ctx, *args, **kwargs)
    rval.alloc_all()
    rval.plan_all()
    return rval

load_tests = load_nengo_tests(simulator_allocator)

try:
    import nose  # If we have nose, maybe we're doing nosetests.

    # Stop some functions from running automatically
    load_nengo_tests.__test__ = False
    load_tests.__test__ = False
    simulator_test_suite.__test__ = False

    # unittest won't try to run this, but nose will
    def test_nengo():
        nengo_suite = simulator_test_suite(simulator_allocator)

        # Unfortunately, this makes us run nose twice,
        # which gives two sets of results.
        # I don't know a way around this.
        assert nose.run(suite=nengo_suite, exit=False)

except ImportError:
    pass


ctx = cl.create_some_context()

def simulator_allocator(*args, **kwargs):
    rval = sim_ocl.Simulator(ctx, *args, **kwargs)
    rval.plan_all()
    return rval

load_tests = nengo.tests.helpers.load_nengo_tests(simulator_allocator)

for foo in load_tests(None, None, None):
    class MyCLS(foo.__class__):
        def Simulator(self, model):
            return simulator_allocator(model)
    globals()[foo.__class__.__name__] = MyCLS
    MyCLS.__name__ = foo.__class__.__name__
    del MyCLS
    del foo
del load_tests


if __name__ == "__main__":
    unittest.main()
