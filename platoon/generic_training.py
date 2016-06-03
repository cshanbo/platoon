class Sampler(object):
    """
    Delivers training samples in the form of batches to the caller.

    #  Objects of this interface are supposed to be used in separate processes in
    #  multi-process/multi-nose systems.
    Can be specified to return batches for one or more particles.

    A concrete implementation must define the functionality (1), as described in
    `GenericTraining`, and use the `Worker` interface for working out a
    distributed strategy for training sampling with other `Sampler` objects in other
    processes.

    """
    # TODO: As in platoon examples, will these objects act as generator?

    def make_samples(self):
        """
        The main function to be implemented which represents a sampling strategy.

        Returns
        -------
        x : tensor-like
            Model inputs
        mask : tensor-like
            Sequence mask
        y : tensor-like
            Targets

        """
        raise NotImplementedError()

    def __call__(self):
        return self.make_samples()


class LocalOptimizingDynamics(object):
    """
    Describes the dynamics that govern the updates of a particle's own parameters.

    Provides an interface that calculates information about local paramater updates
    (Δθ) and an interface to perform updates on a particle.
    #  Objects of this interface are supposed to be used in separate processes in
    #  multi-process/multi-nose systems.

    In general, to call the object the user provides `x`, `mask` and `y`
    variables (representing the mini-batch) and gets cost function's value
    and the dynamics' first derivative, `dupdates`, (containing the local update on
    the particle's parameters `tparams`) as a shared variable.

    A concrete implementation must define the functionality (2), as described in
    `GenericTraining`.

    """

    def __init__(self, tparams, x, mask, y, *cderivs):
        """Constructor of a `LocalOptimizingDynamics` object.

        For parameter documentation, see :func:`define_rules`.

        Raises
        ------
        ValueError
            If `cderivs` does not have length at least 2 or
            if `cderivs` contains Theano variables that do not have
            tparams, x, mask and y as inputs (??).

        """

        if len(cderivs) < 2:
            raise ValueError("cost and grad were not specified in cderivs")
        self.dupdates = []
        self.tparams = tparams
        self.x = x
        self.mask = mask
        self.y = y
        self.cost = cderivs[0]
        self.grad = cderivs[1]
        self.cderivs = cderivs
        self._f_grad_shared, self._f_update = \
            self.define_rules(tparams, x, mask, y, *cderivs)

    def define_rules(self, tparams, x, mask, y, *cderivs):
        """
        The main function to be implemented which represents the local optimizer.

        Parameters
        ----------
        tparams : Theano SharedVariable
            Model parameters

        x : Theano variable
            Model inputs
        mask : Theano variable
            Sequence mask
        y : Theano variable
            Targets

        cderivs : tuple of Theano variables
            Functions which represent orders of the objective function. The 0th order
            is the cost function itself, the 1st order is grad of cost function
            w.r.t. to `tparams` and so forth.


        Notes
        -----
        * Generic optimization dynamics require mathematically the first order
        (grad) of the cost function, but they do not specify explicitly that higher
        orders cannot be used (i.e. in the Newton method). The 0th order is given in
        order to calculate the cost at each update step for tracking training
        progress. Thus, `cderivs` tuple should be at least of length 2, containing
        a cost function at `cderivs[0]` and its grad function at `cderivs[1]`. In
        case that this does not hold a ValueError shall be raised.
        * `cderivs[0]` can be referenced through `self.cost` and `cderivs[1]` can be
        referenced through `self.grad`.
        * Optimization hyperparameters, such as learning rate, should be defined and
        managed by each concrete implementation.

        Returns
        -------
        f_grad_shared : Theano function
            Function that computes gradients of the optimization dynamics
            (using list of Theano SharedVariables `self.dderivs`) for a mini-batch
            input, but does not updates the weights.
            Returns cost value at `x`, `mask`, `y`.
        f_update :  Theano function
            Function that updates the weights from the previously computed gradient
            (at `self.dderivs`). Returns None.

        """
        raise NotImplementedError()

    def __call__(self, xv, maskv, yv):
        """
        Wrapper call to `self._f_grad_shared`
        """
        return self._f_grad_shared(xv, maskv, yv)

    def update_local(self):
        """
        Wrapper call to `self._f_update`
        """
        return self._f_update()

    def share_updates(self):
        """
        Returns
        -------
        dupdates: list of Theano SharedVariables
            Updates which describe the changes in optimization dynamics'
            states in this step. Includes the update in particle's local parameters
            `tparams`.

        """
        return self.dupdates

    def share_params(self):
        """
        Returns
        -------
        tparams: Theano SharedVariable
            Local model parameters

        """
        return self.tparams


class SyncingCondition(object):
    """
    Describes a scheduler of updates using global information (or syncing).

    This abstract class defines the interface to a condition used for deciding when
    global updates should occur.

    """

    def is_to_sync(self, ltparams, ldupdates, gtparams, gdupdates):
        """
        Responsible for checking whether a particle is ready to participate in
        a syncing operation or not

        Parameters
        ----------
        ltparams : Theano SharedVariable
            a particle's own model parameters
        ldupdates : list of Theano SharedVariables
            a particle's local dynamics' updates
        gtparams : Theano SharedVariable (?, gpuarray?)
            global information about model parameters
        gdupdates : list of Theano SharedVariables
            global information about parameter dynamics' updates

        Returns
        -------
        syncing : bool
            if particles will participate in global optimization dynamics
            in this iteration

        """
        raise NotImplementedError()

    def __call__(self, ltparams, ldupdates, gtparams, gdupdates):
        return self.is_to_sync(ltparams, ldupdates, gtparams, gdupdates)


class FinishingCondition(object):
    """
    This abstract class defines the interface to a condition used for deciding when
    a particle's parameter training has ended.

    """

    def has_ended(self, ldupdates, gdupdates, cost):
        """Responsible for checking whether a particle is ready to participate in
        a syncing operation or not

        Parameters
        ----------
        lparams : Theano SharedVariable
            a particle's own model parameters
        dupdates : list of Theano SharedVariables
            a particle's local dynamics' updates
        cost : scalar
            cost in current training iteration's particle position

        Returns
        -------
        syncing : bool
            if particles will participate in global optimization dynamics
            in this iteration

        """
        raise NotImplementedError()

    def __call__(self, ldupdates, gdupdates=None, cost=None):
        return self.has_ended(ldupdates, gdupdates, cost)


class GenericTraining(object):
    """
    Describes an abstract training procedure using a distributed (or not)
    optimization algorithm.

    This abstract class defines the interface which implementations should follow
    in order to be a generic usable training algorithm. In specific, it is expected
    that objects, built from this interface, will be able to:
        a. Provide in high level language the whole distributed optimization algo.
        b. Be agnostic to communication framework.
        c. Be able to change easily into similar algorithms by user, allowing
           flexible experimentations in optimization algorithms.

    This is attempted by recognising that training algorithms in general are composed
    of 5 independent functionalities:
        1. A way to get a (mini) batch.
        2. An update function of a (worker) particle's local information.
        3. A condition which dictates when information from all particles
           will be combined.
        4. An update function which combines information from all particles.
        5. A condition which dictates when the training has finished.

    Objects of this interface are supposed to be used in separate processes in
    multi-process/multi-node systems.

    A concrete implementation must define the functionality (4) and use the `Worker`
    interface for exchanging information with other `GenericTraining` objects in
    other processes.

    """

    def __init__(self, worker, update_before=True,
                 workon_params=True, workon_updates=False):
        super(GenericTraining, self).__init__()
        self._get_batch = None  # (1)
        self._optimize_local = None  # (2)
        self._is_to_sync = None  # (3)
        self._is_to_finish = None  # (5)

        # Initialized `Worker` object which corresponds to this process.
        # `Worker` object exposes a backend-agnostic API to express distributed
        # (multi-worker) operations.
        self.worker = worker

        self.update_before = update_before
        self.workon_params = workon_params
        self.workon_updates = workon_updates

    @property
    def sampler(self):
        return self._get_batch

    @sampler.setter
    def sampler(self, s):
        if not isinstance(s, Sampler):
            raise TypeError("is not an instance of type {}".format(
                Sampler.__name__))
        self._get_batch = s

    @property
    def optimizer(self):
        return self._optimize_local

    @optimizer.setter
    def optimizer(self, s):
        if not isinstance(s, LocalOptimizingDynamics):
            raise TypeError("is not an instance of type {}".format(
                LocalOptimizingDynamics.__name__))
        self._optimize_local = s

    @property
    def syncing(self):
        return self._is_to_sync

    @syncing.setter
    def syncing(self, s):
        if not isinstance(s, SyncingCondition):
            raise TypeError("is not an instance of type {}".format(
                SyncingCondition.__name__))
        self._is_to_sync = s

    @property
    def sampler(self):
        return self._is_to_finish

    @sampler.setter
    def sampler(self, s):
        if not isinstance(s, FinishingCondition):
            raise TypeError("is not an instance of type {}".format(
                FinishingCondition.__name__))
        self._is_to_finish = s

    def sync(self, ltparams, ldupdates, gtparams, gdupdates):
        """
        Performs distributed calculations, which combine information from
        `LocalOptimizingDynamics` objects, in order to have distributed optimizing
        dynamics considering all the particles.

        An implementation of `GenericTraining` must implement this function and
        use compiled Theano functions in conjuction with a `Worker`'s interface.

        Parameters
        ----------
        ltparams : Theano SharedVariable
            a particle's own model parameters
        ldupdates : list of Theano SharedVariables
            a particle's local dynamics' updates
        gtparams : Theano SharedVariable (?, gpuarray?)
            global information about model parameters
        gdupdates : list of Theano SharedVariables
            global information about parameter dynamics' updates

        """
        raise NotImplementedError()

    def train(self):
        """
        Describes a general distributed (or not) training procedure.

        Returns
        -------
        TODO

        """

        if not self._get_batch:
            raise TypeError("has not been specified a {} instance to use".format(
                Sampler.__name__))

        if not self._optimize_local:
            raise TypeError("has not been specified a {} instance to use".format(
                LocalOptimizingDynamics.__name__))

        if not self._is_to_sync:
            raise TypeError("has not been specified a {} instance to use".format(
                SyncingCondition.__name__))

        if not self._is_to_finish:
            raise TypeError("has not been specified a {} instance to use".format(
                FinishingCondition.__name__))

        #  A. Get worker local and global variables
        ltparams = self._optimize_local.share_tparams()
        ldupdates = self._optimize_local.share_updates()
        gtparams = None
        if self.workon_params:
            pass
            # "Shared" variable which refers to global model parameters
            #  gtparams = worker.new_shared_array()
            # init_shared_params ???
        gdupdates = None
        if self.workon_updates:
            pass
            # "Shared" variable which refers to global optimizing dynamics' updates
            #  gdupdates = worker.new_shared_array()
            # init_shared_params ???

        # TODO  B. Workers signal ready to train and
        # and block w. timeout to wait for all (to be error-prune so far)

        while True:
            while True:
                x, mask, y = self._get_batch()
                cost = self._optimize_local(x, mask, y)
                if self.update_before:
                    self._optimize_local.update_local()
                if self._is_to_sync(ltparams, ldupdates, gtparams, gdupdates):
                    break
            self.sync(ltparams, ldupdates, gtparams, gdupdates)
            if not self.update_before:
                self._optimize_local.update_local()
            if self._is_to_finish(ldupdates, gdupdates, cost):
                break
        # TODO On finish properly exit
        # TODO Return best model parameter
        # TODO Training curves

    # TODO add general function for validation

"""
Thought notes:
* Define struct for common referencing arrays (gpuarray, theano array, shared array)
* Define API in Worker for nccl/MPI collectives using CommonArray

Each node owns a unique controller        ->   Controllers speak with MPI or nccl+MPI
Each controller manages several processes ->   Controllers share memory among processes (posix)
                                               And defines communication between them (pyzmq + posix + nccl)
Each process owns a unique worker         ->
Each worker manages a single device       ->

!
Workers belong to a local ncclCommWorld in a common node or
or to a regional ncclCommWorld in a supported multi-node conf.

Then representative controllers are selected from the ncclCommWorld of workers.

Representative controllers belong to a global mpiCommWorld.
!

Action starts from Worker API
1. A collective is called on Worker API
2. Using pygpu, a nccl collective is called on gpuarrays in a regional/local ncclCommWorld
3. Using pygpu, a single worker is elected to copy from gpuarray to node's posix shared memory
4. Using pyzmq, a blocking call is made to its controller in order to continue the collective
5. Using mpi4py, a mpi collective is called on nodes' posix shared memory in the global mpiCommWorld
6. Controllers complete and respond to workers, they copy from node's posix shared memory to gpuarray
7. Using pygpu, ncclBroadcast is called on worker's gpuarray in regional/local ncclCommWorld

Fini
"""
