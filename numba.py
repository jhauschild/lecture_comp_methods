"""Provide dummy alternative for `numba.jit`.

The python package `numba <http://numba.pydata.org/>` provides just-in-time compilation,
which can often give a huge speed-up. However, it is not included into the python standard library.
Although e.g. the python distributions provided by anaconda already come with numba, it might not
be available on your laptops, so we provide a "dummy" alternative for the "@jit" decorator used
in some of the python scripts.
Moreover, this dummy is also usefull for debugging, since the "@jit" often leads to confusing,
lengthy tracebacks and error messages.

If you get an "ImportError: No module named 'numba'" or want to disable the @jit for debugging,
simply copy this file in the folder where you run the scripts.
Note that you need might need to restart the kernel to force a new import, simply calling the cell
with the "import" is not enough.


Example
-------
The usage is as with the jit decorator of the "real" numba package,
but the @jit simply doesn't do anything::

    from numba import jit

    @jit
    def example_function(a, b):
        return a + b

    @jit(nopython=True)
    def another_example_function(a, b):
        return a * b

"""
warnings.warn("did not import numba, but dummy jit. Code will work but run slowly!")

def jit(*args, **kwargs):
    """Dummy decorator, doing nothing, for replacing numba.jit if not available"""
    if len(args) > 0:
        return args[0]
    else:
        def dummy_decorator(func):
            """Dummy decorator"""
            return func
        return dummy_decorator
