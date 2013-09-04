import numpy as np
import pyopencl as cl
from plan import Plan
from mako.template import Template
from clarray import to_device

def plan_parallel_ragged_gather_gemv2(queue,
    alpha, A, A_js, X, X_js,
    beta, Y, Y_in=None, tag=None,
    group_size=32,
    ):
    """
    """

    # TODO: if alpha or beta is a float
    #       then render it into the kernel text.
    try:
        alpha = float(alpha)
        cl_alpha = None
    except TypeError:
        cl_alpha = to_device(queue,
            np.asarray([alpha] * len(Y), Y.buf.dtype))

    try:
        beta = float(beta)
        cl_beta = None
    except TypeError:
        cl_beta = to_device(queue,
            np.asarray([beta] * len(Y), Y.buf.dtype))

    if cl_alpha or cl_beta:
        raise NotImplementedError('alpha or beta non-homogeneous')

    if Y_in is None:
        Y_in = Y

    # Make the global size the closest multiple of the group size (ceiling)
    gsize = (group_size, len(Y))
    lsize = (group_size, 1)

    #print gsize
    #print lsize

    if not np.all(shp0 == 1 for shp0 in Y.shape0s):
        raise NotImplementedError(Y.shape0s)

    # XXX check that all the ints are ints not longs
    textconf = {
        'alpha' : str(alpha),
        'beta' : str(beta),
        'type_alpha': None, #cl_alpha.ocldtype,
        'type_beta': None, #cl_beta.ocldtype,
        'type_A': A.cl_buf.ocldtype,
        'type_X': X.cl_buf.ocldtype,
        'type_Y': Y.cl_buf.ocldtype,
        'y_len': len(Y),
        'lsize': group_size,
    }

    text = """
        __kernel void fn(
            //const __global ${type_alpha} * alphas,
            const __global int *A_starts,
            const __global int *A_shape1s,
            const __global int *A_ldas,
            const __global ${type_A} *A_data,
            const __global int *A_js_starts,
            const __global int *A_js_lens,
            const __global int *A_js_data,
            const __global int *X_starts,
            const __global ${type_X} *X_data,
            const __global int *X_js_starts,
            const __global int *X_js_data,
            //const __global ${type_beta} * betas,
            const __global int *Y_in_starts,
            const __global ${type_Y} *Y_in_data,
            const __global int *Y_starts,
            const __global int *Y_lens,
            __global ${type_Y} *Y_data)
    {
        __local ${type_Y} partialDotProduct[${lsize}]; //Scratch space for the dot products
        for (int bb = get_global_id(1);
                 bb < ${y_len};
                 bb += get_global_size(1))
        {
            const int mm = 0; // TODO: SUPPORT M > 1

            const int y_offset = Y_starts[bb];

            % if beta != 0 :
                Y_data[y_offset] = ${beta} * Y_in_data[Y_in_starts[bb]];
            % else :
                Y_data[y_offset] = 0;
            % endif

            ${type_Y} y_sum = 0;
            const int n_dot_products = A_js_lens[bb];
            const __global int* X_js_row = X_js_data + X_js_starts[bb];
            const __global int* A_js_row = A_js_data + A_js_starts[bb];

            for(int ii = 0; ii < n_dot_products; ii++) {
                const int x_ji = X_js_row[ii];
                const int a_ji = A_js_row[ii];
                const int N_i = A_shape1s[a_ji];
                const int lda_i = A_ldas[a_ji];

                const __global ${type_A}* A_row = A_data + A_starts[a_ji];
                const __global ${type_X}* X_row = X_data + X_starts[x_ji];

                for (int nn = get_local_id(0); nn < N_i; nn += get_local_size(0)) {
                    y_sum += A_row[nn * lda_i + mm] * X_row[nn];
                }
            }

            // -- Parallel reduction within local group sum registers
            barrier(CLK_LOCAL_MEM_FENCE);
            partialDotProduct[get_local_id(0)] = y_sum;

            for (uint stride = 1; stride < get_local_size(0); stride *= 2) {
                // XXX not necessary IFF local_size(0) < warp size
                barrier(CLK_LOCAL_MEM_FENCE);

                uint index = 2 * stride * get_local_id(0);
                if (index + stride < get_local_size(0)) {
                    partialDotProduct[index] += partialDotProduct[index + stride];
                }
            }
            if (get_local_id(0) == 0) {
                Y_data[y_offset] += ${alpha} * partialDotProduct[0];
            }
        }
    }
        """

    text = Template(text, output_encoding='ascii').render(**textconf)

    _fn = cl.Program(queue.context, text).build().fn

    full_args = (
                 #cl_alpha,
                 A.cl_starts,
                 A.cl_shape1s,
                 A.cl_ldas,
                 A.cl_buf,
                 A_js.cl_starts,
                 A_js.cl_shape0s,
                 A_js.cl_buf,
                 X.cl_starts,
                 X.cl_buf,
                 X_js.cl_starts,
                 X_js.cl_buf,
                 #cl_beta,
                 Y_in.cl_starts,
                 Y_in.cl_buf,
                 Y.cl_starts,
                 Y.cl_shape0s,
                 Y.cl_buf,
                )

    _fn.set_args(*[arr.data for arr in full_args])

    rval = Plan(queue, _fn, gsize, lsize,
                name='ref_parallel_ragged_gather_gemv',
                tag=tag,
               )
    # prevent garbage-collection
    rval.full_args = full_args
    return rval


def plan_ragged_gather_gemv(queue, alpha, A, A_js, X, X_js,
                            beta, Y, Y_in=None, tag=None):
    """
    """

    # TODO: if alpha or beta is a float
    #       then render it into the kernel text.
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

    cl_alpha = to_device(queue, np.asarray(alpha, Y.buf.dtype))
    cl_beta = to_device(queue, np.asarray(beta, Y.buf.dtype))

    if Y_in is None:
        Y_in = Y

    # XXX check for e.g. all Ns being the same thing
    #     especially all Ns == 1
    # cl_Ns = to_device(queue, np.asarray(Ns, 'int32'))

    # XXX check that all the ints are ints not longs
    textconf = {
        'type_alpha': cl_alpha.ocldtype,
        'type_beta': cl_beta.ocldtype,
        'type_A': A.cl_buf.ocldtype,
        'type_X': X.cl_buf.ocldtype,
        'type_Y': Y.cl_buf.ocldtype,
        'tag': str(tag),
        'do_inner_products': (A_js is not None),
    }

    text = """
        __kernel void fn(
            __global ${type_alpha} * alphas,
            __global int *A_starts,
            __global int *A_shape1s,
            __global int *A_ldas,
            __global ${type_A} *A_data,
            __global int *A_js_starts,
            __global int *A_js_shape0s,
            __global int *A_js_data,
            __global int *X_starts,
            __global ${type_X} *X_data,
            __global int *X_js_starts,
            __global int *X_js_data,
            __global ${type_beta} * betas,
            __global int *Y_in_starts,
            __global ${type_Y} *Y_in_data,
            __global int *Y_starts,
            __global int *Y_shape0s,
            __global ${type_Y} *Y_data)
        {
            const int mm = get_global_id(0);
            const int bb = get_global_id(1);
            const int M = Y_shape0s[bb];
            if (mm < M)
            {
                const ${type_alpha} alpha = alphas[bb];
                const ${type_beta} beta = betas[bb];

                const int y_offset = Y_starts[bb];
                const int y_in_offset = Y_in_starts[bb];

                Y_data[y_offset + mm] = beta * Y_in_data[y_in_offset + mm];

    % if do_inner_products :
                const int n_dot_products = A_js_shape0s[bb];
                X_js_data += X_js_starts[bb];
                A_js_data += A_js_starts[bb];

                ${type_Y} y_sum = 0;
                for (int ii = 0; ii < n_dot_products; ++ii)
                {
                    //printf("${tag}: ii=%i / %i\\n", ii, n_dot_products);
                    const int x_ji = X_js_data[ii];
                    const int a_ji = A_js_data[ii];
                    //printf("x_ji=%i a_ji=%i\\n", x_ji, a_ji);
                    const int N_i = A_shape1s[a_ji];
                    const int x_offset = X_starts[x_ji];
                    const int a_offset = A_starts[a_ji];
                    //printf("x_offset=%i a_offset=%i\\n", x_offset, a_offset);
                    const int lda_i = A_ldas[a_ji];

                    for (int nn = 0; nn < N_i; ++nn)
                        y_sum += X_data[x_offset + nn]
                                 * A_data[a_offset + nn * lda_i + mm];
                }
                Y_data[y_offset + mm] += alpha * y_sum;
    % endif
            }
        }
    """

    text = Template(text, output_encoding='ascii').render(**textconf)
    if False: #tag == 'transforms':
        print text
        print A_js

    ### TODO: use the maximum of A.shape0s that is actually used in this op
    gsize = (int(max(A.shape0s)), int(len(Y)),)
    lsize = None
    _fn = cl.Program(queue.context, text).build().fn
    dummy = A.cl_buf
    full_args = (
        cl_alpha,
        A.cl_starts,
        A.cl_shape1s,
        A.cl_ldas,
        A.cl_buf,
        A_js.cl_starts if A_js is not None else dummy,
        A_js.cl_shape0s if A_js is not None else dummy,
        A_js.cl_buf if A_js is not None else dummy,
        X.cl_starts,
        X.cl_buf,
        X_js.cl_starts if X_js is not None else dummy,
        X_js.cl_buf if X_js is not None else dummy,
        cl_beta,
        Y_in.cl_starts,
        Y_in.cl_buf,
        Y.cl_starts,
        Y.cl_shape0s,
        Y.cl_buf)

    #print [str(arr.dtype)[0] for arr in full_args]
    _fn.set_args(*[arr.data for arr in full_args])
    rval = Plan(queue, _fn, gsize, lsize,
                name='ref_ragged_gather_gemv',
                tag=tag,
               )
    # prevent garbage-collection
    rval.full_args = full_args
    return rval

