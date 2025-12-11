from omegaconf import OmegaConf
from typing import Dict, Any, Callable, Optional
from loguru import logger
import time
import importlib
from functools import wraps

def create_object(config: Dict) -> Any:
    '''
    Factory function to create an object, for dependency injection usage.
    '''
    omgcfg = OmegaConf.create(config)
    module_path, class_name = omgcfg["__object__"]["path"], omgcfg["__object__"]["name"]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    args = omgcfg["__object__"].get("args", {})
    return cls(**args)

def timeit(func: Optional[Callable] = None, *, enabled: bool = True) -> Callable:
    '''
    Decorator to measure execution time of a function.

    Can be used as a plain decorator or with a parameter.
    Usage:
        @timeit
        def f(...):
            ...

        @timeit(enabled=False)
        def g(...):
            ...
    '''
    # Support both @timeit and @timeit(enabled=False)
    if func is None:
        def decorator(f: Callable) -> Callable:
            return timeit(f, enabled=enabled)
        return decorator

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not enabled:
            return func(*args, **kwargs)
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.debug(f"Function {func.__name__} took {elapsed_time:.4f} seconds to execute.")
        return result

    return wrapper
