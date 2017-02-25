# -*- coding: utf-8 -*-
"""
:mod:`util` -- Common utility functions for Platoon's classes
=============================================================

.. module:: util
   :platform: Unix
   :synopsis: Contains PlatoonException classes and various helpers.

"""
from __future__ import print_function
import os
import sys
import subprocess
import cffi


class PlatoonException(Exception):
    """Exception used for abnormal behaviour related to Platoon.

    Useful for logging and managing error.

    """
    def __init__(self, severity, descr, from_exc=None):
        self.severity = severity
        self.descr = descr
        self.from_exc = from_exc

    def __str__(self):
        d = str(self.severity) + "! " + str(self.descr)
        if self.from_exc is not None:
            d += "\nReason: " + str(self.from_exc)
        return d


class PlatoonError(PlatoonException):
    """
    Exception used for errors related to Platoon.
    """
    def __init__(self, descr, from_exc=None):
        super(PlatoonError, self).__init__("ERROR", descr, from_exc)


class PlatoonWarning(PlatoonException):
    """
    Exception used for warnings related to Platoon.
    """
    def __init__(self, descr, from_exc=None):
        super(PlatoonWarning, self).__init__("WARNING", descr, from_exc)


def mmap(length=0, prot=0x3, flags=0x1, fd=0, offset=0):
    """
    Map file descriptor or shared memory buffer to virtual address space of this
    process and create an object with Python buffer interface for that address.
    """
    _ffi = cffi.FFI()
    _ffi.cdef("void *mmap(void *, size_t, int, int, int, size_t);")
    _lib = _ffi.dlopen(None)

    addr = _ffi.NULL

    m = _lib.mmap(addr, length, prot, flags, fd, offset)
    if m == _ffi.cast('void *', -1):
        raise OSError(_ffi.errno, "for mmap")
    return _ffi.buffer(m, length)


def launch_process(logs_folder, experiment_name, args, device,
                   process_type="worker"):
    """
    Helper function for a Platoon subprocess.
    """
    print("## Starting {0} on {1} ...".format(process_type, device), end=' ')

    log_file = os.path.join(logs_folder, "{0}_{1}.{{}}".format(process_type, device))
    with open(log_file.format("out"), 'w') as stdout_file:
        with open(log_file.format("err"), 'w') as stderr_file:
            env = dict(os.environ)
            env['THEANO_FLAGS'] = '{},device={}'.format(env.get('THEANO_FLAGS', ''), device)
            command = [sys.executable] + shape_args(experiment_name, args, process_type)
            process = subprocess.Popen(command, bufsize=0, stdout=stdout_file, stderr=stderr_file, env=env)

    print("Done")
    return process


def shape_args(experiment_name, args, process_type):
    """
    Returns a proper list of arguments that will spawn a process
    """
    if experiment_name == "platoon" and process_type == "controller":
        executable = ["-m", "platoon.channel.controller"]
    else:
        executable = ["{0}_{1}.py".format(experiment_name, process_type)]
    fixed_args = ["-u"] + executable
    if args:
        fixed_args += args
    return fixed_args


class SingletonType(type):
    """
    Metaclass that implements the singleton pattern for a Python class.
    """
    def __init__(cls, name, bases, dict):
        super(SingletonType, cls).__init__(name, bases, dict)
        cls.instance = None

    def __call__(cls, *args, **kwds):
        if cls.instance is None:
            cls.args = args
            cls.kwds = kwds
            cls.instance = super(SingletonType, cls).__call__(*args, **kwds)
        else:
            if args or kwds:
                print(PlatoonWarning("Worker instance has already been initialized."
                                     "\nArgs: {0}, Kwds: {1}".format(args, kwds)),
                      file=sys.stderr)
        return cls.instance
