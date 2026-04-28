# Stub implementation of the OpenFilter pipeline framework by Plainsight.
# The production library handles multiprocessing, GPU scheduling, and network
# stream management - none of which we needed for a single-camera hackathon demo.
# All filters are written against this interface, so swapping in the real
# OpenFilter later would require zero changes to them.
#
# Design pattern: Pipeline / Chain of Responsibility.
# Each filter receives a shared data dict, enriches it, and passes it on.
# Filters are decoupled - none knows what comes before or after it.


class Filter:
    """
    Abstract base class for all pipeline stages.
    Defines the contract every filter must honour: accept a data dict in run(),
    return it modified, or return None to halt the pipeline.
    """

    # Nothing to initialise at this level - exists so subclasses can safely call super().__init__() without hitting a TypeError.
    def __init__(self):
        pass

    # Hook for mid-run graceful stop - not used in our pipeline but kept
    # to match the real OpenFilter interface contract.
    def stop(self):
        pass

    # Default passthrough - a filter that forgets to override run() won't
    # silently corrupt the pipeline, data just flows through unchanged.
    def run(self, data):
        return data

    # Called during pipeline shutdown so filters can release resources like
    # file handles or model memory. Safe no-op default means filters that
    # need no cleanup don't have to override it.
    def close(self):
        pass


class Graph:
    """
    Owns and orchestrates the ordered filter chain.
    The only class that knows about filter ordering - individual filters have
    zero knowledge of their neighbours, keeping the pipeline loosely coupled.
    """

    # Store the filter chain in the order provided - order is significant,
    # each filter assumes its dependencies are already in the data dict.
    def __init__(self, filters):
        self.filters = filters
        self._out = {}

    # Run one complete frame through every filter in sequence.
    # Creates a fresh data dict each call so no stale data leaks between frames.
    # Returns True on success, False if any filter signals end-of-stream via None.
    def step(self):
        data = {}

        for f in self.filters:
            data = f.run(data)
            # Fail-fast: stop immediately if a filter returns None rather than
            # passing it downstream and getting a confusing crash in the next filter.
            if data is None:
                return False

        # Persist output so the main loop can retrieve the annotated frame
        # via last_output() without coupling to pipeline internals.
        self._out = data
        return True

    # Accessor for the last completed frame's data dict.
    def last_output(self):
        return self._out

    # Shut down every filter cleanly - hasattr guard handles filters that
    # don't implement close() without raising an AttributeError.
    def close(self):
        for f in self.filters:
            if hasattr(f, "close"):
                f.close()