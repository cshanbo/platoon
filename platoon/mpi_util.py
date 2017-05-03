# -*- coding: utf-8 -*-
"""
:mod:`mpi_util` -- MPI utility functions for Platoon's classes
==============================================================

.. module:: util
   :platform: Unix
   :synopsis: Contains conversion functions and worker spawning through MPI.

"""
from __future__ import absolute_import, print_function
import os
import sys

import numpy
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from .util import shape_args


def launch_mpi_workers(workers_count, experiment_name, worker_args, devices):
    """
    Helper function for spawning dynamically a Platoon subprocess (usually a
    worker) in multi-node MPI environment.
    """
    if MPI is None:
        raise ImportError("No module named 'mpi4py'")
    import socket
    args = shape_args(experiment_name, worker_args, "worker")
    theano = os.environ['THEANO_FLAGS']
    info = MPI.Info.Create()
    info['host'] = socket.gethostname()
    theano_flags = '%s,device=%s' % (theano, devices[0].strip())
    info.Set('env', 'THEANO_FLAGS=%s\n' % theano_flags)
    errcodes = []

    intercomm = MPI.COMM_SELF.Spawn(sys.executable, args, workers_count, info=info, 
                                    root=0, errcodes=errcodes)
    info.Free()
    if numpy.any(numpy.asarray(errcodes) != MPI.SUCCESS):
        raise PlatoonError("MPI spawn multi error codes: {0}\nArgs passed: {1}".format(errcodes, args))

    return intercomm


if MPI:
    GA_TO_MPI_OP = {
        '+': MPI.SUM,
        "sum": MPI.SUM,
        "add": MPI.SUM,
        '*': MPI.PROD,
        "prod": MPI.PROD,
        "product": MPI.PROD,
        "mul": MPI.PROD,
        "max": MPI.MAX,
        "maximum": MPI.MAX,
        "min": MPI.MIN,
        "minimum": MPI.MIN,
        }

    NP_TO_MPI_TYPE = {
        numpy.dtype('bool'): MPI.C_BOOL,
        numpy.dtype('int8'): MPI.INT8_T,
        numpy.dtype('uint8'): MPI.UINT8_T,
        numpy.dtype('int16'): MPI.INT16_T,
        numpy.dtype('uint16'): MPI.UINT16_T,
        numpy.dtype('int32'): MPI.INT32_T,
        numpy.dtype('uint32'): MPI.UINT32_T,
        numpy.dtype('int64'): MPI.INT64_T,
        numpy.dtype('uint64'): MPI.UINT64_T,
        numpy.dtype('float32'): MPI.FLOAT,
        numpy.dtype('float64'): MPI.DOUBLE,
        numpy.dtype('complex64'): MPI.C_FLOAT_COMPLEX,
        numpy.dtype('complex128'): MPI.C_DOUBLE_COMPLEX,
        # TODO How to handle half types in MPI?
        #  numpy.dtype('float16'): MPI.HALF,
        }


def op_to_mpi(op):
    """
    Converts pygpu collective reduce operation types to MPI reduce operation
    types.
    """
    if MPI is None:
        raise ImportError("No module named 'mpi4py'")
    res = GA_TO_MPI_OP.get(op.lower())
    if res is not None:
        return res
    raise ValueError("Invalid reduce operation: {}".format(str(op)))


def dtype_to_mpi(dtype):
    """
    Converts numpy datatypes to MPI datatypes.
    """
    if MPI is None:
        raise ImportError("No module named 'mpi4py'")
    res = NP_TO_MPI_TYPE.get(numpy.dtype(dtype))
    if res is not None:
        return res
    raise TypeError("Conversion from dtype {} is not known".format(dtype))
