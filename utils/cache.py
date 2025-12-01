import json
from datetime import timedelta
from functools import wraps
from typing import Any, Callable

from redis import Redis

from const import REDIS_URL


RedisCache = Redis.from_url(REDIS_URL)


def key_generator(*args, **kwargs) -> str:
    key = "::".join((*[f"{k}::{hash(v)}" for k, v in kwargs.items()], *[v.__str__() for v in args]))
    return key


def wrap_cache(cache: Redis, ttl: timedelta | None = None, key_generator: Callable = key_generator) -> Callable:
    def decorator(function: Callable) -> Callable:
        @wraps(function)
        def wrapper(*args, **kwargs) -> Any:
            key = key_generator(function.__name__, *args, **kwargs)
            value = cache.get(key)
            if value is None:
                value = function(*args, **kwargs)
                cache.set(key, json.dumps(value), ex=ttl)
            else:
                print(f"Using cache for {key[:10]}")
                value = json.loads(value)

            return value

        return wrapper

    return decorator
