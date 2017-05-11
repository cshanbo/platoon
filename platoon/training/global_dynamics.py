# -*- coding: utf-8 -*-
"""
:mod:`training.global_dynamics` -- Collection of global SGD strategies
======================================================================

.. module:: global_dynamics
   :platform: Unix
   :synopsis: Contains :class:`GlobalDynamics` base class for synchronous
              global gradient descents and implementation of various techniques
              using Platoon's :class:`channel.worker.Worker`'s
              :meth:`channel.worker.Worker.all_reduce` interface.

Implementations
---------------
* *:class:`SGD`* : Synchronous variant of Stochastic Gradient Descent for many
                   descending particles.
* *:class:`EASGD`* : Elastic Averaging Stochastic Gradient Descent (synchronous)
* *:class:`Downpour`* : A synchronous variant of Downpour

"""
from __future__ import absolute_import, division

from ..channel.worker import Worker
from ..ops import AllReduceSum


class GlobalDynamics(object):
    """Abstract class which declares the methods and properties that need to
    be implemented by a synchronous global dynamics rule.

    Parameters
    ----------
    worker : :class:`channel.Worker`, optional
       A reference to Worker's instance

    .. versionadded:: 0.6.0

    """
    def __init__(self, worker=None):
        self._worker = None
        if worker is not None:
            self.worker = worker
        self._fn = None

    def __call__(self):
        if self._fn is None:
            raise NotImplementedError("Functionality has not been specified.\n"
                                      "Please use {} method to setup GlobalDynamics"
                                      "for a set of Variables\nor supply your own"
                                      "using {} method.".format(
                                          repr(self.make_rule), repr(self.fn)))
        self._fn()

    @property
    def worker(self):
        """Worker class instance used for global operations"""
        if self._worker is None:
            try:
                self._worker = Worker()  # Draw singleton instance
            except TypeError:
                raise AttributeError("Worker instance has not been created yet.")
        return self._worker

    @worker.setter
    def worker(self, inst):
        if not isinstance(inst, Worker):
            raise TypeError("Argument `inst` is not of platoon.Worker type.")
        self._worker = inst

    def register_fn(self, fun):
        """Internal function implementing global dynamics. Does not accept
        parameters. Global optimization must be done through shared variables.

        The responsibility for supplying a valid internal function falls to the
        user. It must be able to be called like this: ``fun()``. Also in order
        to serve its purpose, it needs to have multi-GPU or even multi-node
        functionality. As a result, a :class:`channel.Worker` or other interface
        need to be used.

        :param fun: Implements global dynamics by using information
                    from many workers.
        :type fun: callable

        """
        if not hasattr(fun, '__call__'):
            raise TypeError("Supplied object is not a callable.")
        self._fn = fun

    def make_rule(self, *args):
        """
        Create :class:`GlobalDynamics` optimization function for
        local data in `args`.

        Implementation in a child class must return a callable object which
        expects no arguments. User must be careful to create a function which
        uses shared objects in order to update local model parameters, such as
        Theano Shared Variables.

        Notes
        -----
        For better performance, try to batch together in the same
        :ref:`theano.compile.SharedVariable` as many model parameter arrays as
        possible. This reduces the number of calls and utilizes the most out of
        the underlying algorithms. One way to do this is to create one c
        contiguous array that contains every set (matrix) of model parameters
        along the first dimension. Then in order to use each set separately,
        create as many view arrays as the number of sets of model parameters,
        i.e. the length of the first dimension. Use the whole array as an input
        to the :meth:`make_rule` function!

        """
        raise NotImplementedError(self.make_rule.__doc__)


class _GlobalDynamicsNoSet(GlobalDynamics):
    def register_fn(self, fun):
        raise AttributeError("Cannot set internal function. Use {} method.".format(
            repr(self.make_rule)))


class SGD(_GlobalDynamicsNoSet):
    """Synchronous Stochastic Gradient Descent:

    It sums or averages model parameter updates found separately (and
    concurrently) by workers which are training on (different) random
    mini-batches of a dataset.

    Parameters
    ----------
    average : bool, optional
       If True, it will normalize the summation of model param updates across
       all workers with the number of workers participating in optimization.
    worker : :class:`channel.Worker`
       See :class:`GlobalDynamics`.

    .. versionadded:: 0.6.0

    """
    def __init__(self, average=False, worker=None):
        self.average = average
        super(SGD, self).__init__(worker)

    def make_rule(self, local_updates):
        """Makes global synchronous SGD rule for the parameters in `local_updates`.

        Parameters
        ----------
        local_updates : {:ref:`theano.compile.SharedVariable`,
                         list of :ref:`theano.compile.SharedVariable`}
           These variables represent the updates found
           by local optimization dynamics on the model's parameters.

        .. seealso:: Notes on :meth:`GlobalDynamics.make_rule`

        """
        import theano
        if isinstance(local_updates, theano.compile.SharedVariable):
            local_updates = [local_updates]
        global_updates = []
        for update in local_updates:
            gup = AllReduceSum(update, inplace=True)
            if self.average:
                gup /= self.worker.global_size
            global_updates.append(gup)
        self._fn = theano.function([], [],
                                   updates=list(zip(local_updates, global_updates)),
                                   accept_inplace=True)


def SumSGD(worker=None):
    """Synchronous Stochastic Gradient Descent: summing version

    .. seealso:: Class :class:`SGD`
    .. versionadded:: 0.6.0

    """
    return SGD(average=False, worker=worker)


def AverageSGD(worker=None):
    """Synchronous Stochastic Gradient Descent: averaging version

    .. seealso:: Class :class:`SGD`
    .. versionadded:: 0.6.0

    """
    return SGD(average=True, worker=worker)


class EASGD(_GlobalDynamicsNoSet):
    """Synchronous variant of Elastic Averaging Stochastic Gradient Descent

    This algorithm is described in more details in the following paper:
    http://arxiv.org/abs/1412.6651

    .. seealso:: Class :class:`GlobalDynamics` for parameters
    .. versionadded:: 0.6.0

    """
    def make_rule(self, local_particle, central_particle, alpha):
        """Make EASGD rule.

        According to this rule, every N iterations, a worker synchronizes his
        parameters with the master parameters. This is done by moving each set of
        parameters toward the other by an amount proportional to the difference
        between the individual params (this proportion is parameterized by `alpha`).

        Parameters
        ----------
        local_particle : {:ref:`theano.compile.SharedVariable`,
                          list of :ref:`theano.compile.SharedVariable`}
           A particle's position in parameter space doing local SGD.
        central_particle : {:ref:`theano.compile.SharedVariable`,
                            list of :ref:`theano.compile.SharedVariable`}
           Central particle's position in parameter space interacting with
           local particles.
        alpha: scalar
           "Elastic" force's coefficient

        .. note::
           If `alpha` == 0 is used, there is no synchronization of the
           parameters meaning that each worker is independently training using SGD.

        .. seealso:: Notes on :meth:`GlobalDynamics.make_rule`

        """
        import theano
        if isinstance(local_particle, theano.compile.SharedVariable):
            local_particle = [local_particle]
        if isinstance(central_particle, theano.compile.SharedVariable):
            central_particle = [central_particle]
        self.alpha = alpha

        new_local = []
        new_central = []
        for local_position, central_position in zip(local_particle, central_particle):
            distance = local_position - central_position
            elastic_force = alpha * distance
            # Note: not equivalent to physical force as `elastic_force`:=Δx/Δt
            # and not Δp/Δt
            local_new_position = local_position - elastic_force
            total_elastic_force = AllReduceSum(elastic_force, inplace=True)
            central_new_position = central_position + total_elastic_force

            new_local.append(local_new_position)
            new_central.append(central_new_position)

        updates = list(zip(local_particle, new_local)) + \
            list(zip(central_particle, new_central))
        self._fn = theano.function([], [], updates=updates, accept_inplace=True)


class Downpour(_GlobalDynamicsNoSet):
    """Synchronous variant of Downpour distributed optimization technique

    This algorithm is described in details in the following paper:
    http://research.google.com/archive/large_deep_networks_nips2012.html

    Parameters
    ----------
    average : bool, optional
       If True, it will average the sum of locally accumulated parameter updates
       in every global update.
    worker : :class:`channel.Worker`, optional
       See :class:`GlobalDynamics`.

    .. versionadded:: 0.6.0

    """
    def __init__(self, average=False, worker=None):
        self.average = average
        super(Downpour, self).__init__(worker)

    def make_rule(self, local_particle, local_acc_updates, global_particle):
        """Make Downpour rule.

        All particles along with the global particle start from the same
        position. According to this rule, each local particle executes descent
        normally but their parameter updates are accumulated (e.g. by moving
        average) to a variable. Every N iterations, the local accumulated
        updates are added together and applied to the global particle. Each
        local particle restarts from global particle's position.

        Parameters
        ----------
        local_particle : {:ref:`theano.compile.SharedVariable`,
                          list of :ref:`theano.compile.SharedVariable`}
           A particle's position in parameter space doing local SGD.
        local_acc_updates : {:ref:`theano.compile.SharedVariable`,
                             list of :ref:`theano.compile.SharedVariable`}
           Shared variable accumulating local parameter updates.
        global_particle : {:ref:`theano.compile.SharedVariable`,
                           list of :ref:`theano.compile.SharedVariable`}
           A particle whose position is updated only by the Downpour process and
           resets position of local particles.

        .. seealso:: Notes on :meth:`GlobalDynamics.make_rule`

        """
        import theano
        from theano.tensor import basic
        if isinstance(local_particle, theano.compile.SharedVariable):
            local_particle = [local_particle]
        if isinstance(local_acc_updates, theano.compile.SharedVariable):
            local_acc_updates = [local_acc_updates]
        if isinstance(global_particle, theano.compile.SharedVariable):
            global_particle = [global_particle]

        new_global = []
        new_local = []
        new_acc_updates = []
        for lp, lau, gp in zip(local_particle, local_acc_updates, global_particle):
            global_acc_updates = AllReduceSum(lau, inplace=True)
            if self.average:
                global_acc_updates /= self.worker.global_size
            new_global.append(gp + global_acc_updates)
            new_local.append(new_global[-1])
            new_acc_updates.append(basic.zeros_like(lau))

        updates = list(zip(local_particle, new_local)) + \
            list(zip(local_acc_updates, new_acc_updates)) + \
            list(zip(global_particle, new_global))

        self._fn = theano.function([], [], updates=updates, accept_inplace=True)
