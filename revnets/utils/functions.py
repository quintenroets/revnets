def always_return_tuple(function):
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)
        if not isinstance(result, tuple):
            result = (result,)
        return result

    return wrapper
