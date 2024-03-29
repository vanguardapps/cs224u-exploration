import os, sys


def relative_path(filepath):
    """
    Get relative path with respect to whatever file the calling code is in.
    """
    caller__file__ = sys._getframe(1).f_globals["__file__"]
    caller_dirname = os.path.dirname(caller__file__)
    return os.path.join(caller_dirname, filepath)


def get_basic_object():
    """
    Literally just get an object that you can use like `object.some_key = 3`.
    This should not be this obtuse. JavaScript is far superior with things
    like this.
    """
    return type("basic_object", (object,), {})()
