# -*- coding: utf-8 -*-
"""
:mod:`mpi_convert` -- Conversion methods for MPI
================================================

.. module:: util
   :platform: Unix
   :synopsis: Contains methods :fun:`op_to_mpi` and :fun:`dtype_to_mpi`,
      if mpi4py is found.

"""
import numpy
try:
    from mpi4py import rc
    rc.initialize = False
    from mpi4py import MPI
except ImportError:
    MPI = None


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
        raise AttributeError("mpi4py is not imported")
    res = GA_TO_MPI_OP.get(op.lower())
    if res is not None:
        return res
    raise ValueError("Invalid reduce operation: {}".format(str(op)))


def dtype_to_mpi(dtype):
    """
    Converts numpy datatypes to MPI datatypes.
    """
    if MPI is None:
        raise AttributeError("mpi4py is not imported")
    res = NP_TO_MPI_TYPE.get(numpy.dtype(dtype))
    if res is not None:
        return res
    raise TypeError("Conversion from dtype {} is not known".format(dtype))
