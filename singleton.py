class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.

    To get the singleton instance, use the `Instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    """

    def __init__(self, decorated):
        self._decorated = decorated

    def Instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self.__instance
        except AttributeError:
            self.__instance = self._decorated()
            return self.__instance

    def __call__(self):
        raise TypeError("Singletons must be accessed through `Instance()`.")

    def ___instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


class SingleClass(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        str_args = str(args)
        str_kwargs = str(kwargs)
        key = str(cls)+str_args+str_kwargs
        if key not in cls._instances:
            cls._instances[key] = super(
                SingleClass, cls).__call__(*args, **kwargs)
        return cls._instances[key]
