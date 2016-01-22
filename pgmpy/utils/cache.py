#!/usr/bin/env python3
from __future__ import with_statement
import collections

import threading
import time

_EMPTY_OBJ = object()
_DEFAULT_TIMEOUT = 2 ** 60  # Too large to for normal purposes


class ExpiringLRUCache(object):

    """Implements a pseudo-LRU algorithm (CLOCK) with expiration times"""

    def __init__(self, size, default_timeout=_DEFAULT_TIMEOUT):
        self.default_timeout = default_timeout
        size = int(size)
        if size < 1:
            raise ValueError('size must be >0')
        self.size = size
        self.lock = threading.Lock()
        self.lookUp_pos = 0
        self.max_position = size - 1
        self.clock_keys = None
        self.clock_refs = None
        self.cache_data = None
        self.force_removal = 0  # For user
        self.hits = 0           # For user
        self.misses = 0         # For user
        self.lookups = 0        # For user
        self.clean_cache()

    def clean_cache(self):
        """Remove all entries from the cache"""
        with self.lock:
            # self.cache_data contains (pos, value, expires) triplets
            self.cache_data = {}
            size = self.size
            self.clock_keys = [_EMPTY_OBJ] * size
            self.clock_refs = [False] * size
            self.lookUp_pos = 0
            self.force_removal = 0
            self.hits = 0
            self.misses = 0
            self.lookups = 0

    def get(self, key, default=None):
        """Return value for key. If not in cache or expired, return default"""
        self.lookups += 1
        try:
            pos, value, expires = self.cache_data[key]
        except KeyError:
            self.misses += 1
            return default
        if expires > time.time():
            # cache entry still valid
            self.hits += 1
            self.clock_refs[pos] = True
            return value
        else:
            # cache entry has expired.
            self.misses += 1
            self.clock_refs[pos] = False
            return default

    def put(self, key, value, timeout=None):
        """Add key to the cache with value

        key will expire in timeout seconds. If key is already in cache, value
        and timeout will be updated.
        """
        max_position = self.max_position
        clock_refs = self.clock_refs
        clock_keys = self.clock_keys
        cache_data = self.cache_data
        lock = self.lock
        if timeout is None:
            timeout = self.default_timeout

        with self.lock:
            entry = cache_data.get(key)
            if entry is not None:
                pos = entry[0]
                cache_data[key] = (pos, value, time.time() + timeout)
                clock_refs[pos] = True
                return
            # searching for a place to insert the key in cache
            lookUp_pos = self.lookUp_pos
            count = 0
            max_count = max_position % 101
            while 1:
                ref = clock_refs[lookUp_pos]
                if ref == True:
                    clock_refs[lookUp_pos] = False
                    lookUp_pos += 1
                    if lookUp_pos > max_position:
                        lookUp_pos = 0

                    count += 1
                    if count >= max_count:
                        # Break the loop, too many searches.
                        clock_refs[lookUp_pos] = False
                else:
                    oldkey = clock_keys[lookUp_pos]
                    oldentry = cache_data.pop(oldkey, _EMPTY_OBJ)
                    if oldentry is not _EMPTY_OBJ:
                        self.force_removal += 1
                    clock_keys[lookUp_pos] = key
                    clock_refs[lookUp_pos] = True
                    cache_data[key] = (lookUp_pos, value, time.time() + timeout)
                    lookUp_pos += 1
                    if lookUp_pos > max_position:
                        lookUp_pos = 0
                    self.lookUp_pos = lookUp_pos
                    break

    def invalidate(self, key):
        """Remove key from the cache"""
        # pop with default arg will not raise KeyError
        entry = self.cache_data.pop(key, _EMPTY_OBJ)
        if entry is not _EMPTY_OBJ:
            self.clock_refs[entry[0]] = False


class LruCache(object):
    """ Decorator for LRU-cached function

    timeout parameter specifies after how many seconds a cached entry should
    be considered invalueid.
    """

    def __init__(self, maxsize, timeout=None, args_non_hashable=False):
        if timeout is None:
            cache = ExpiringLRUCache(maxsize, default_timeout=_DEFAULT_TIMEOUT)
        else:
            cache = ExpiringLRUCache(maxsize, default_timeout=timeout)
        self.args_non_hashable = args_non_hashable
        self.cache = cache

    def __call__(self, f):
        cache = self.cache
        empty_obj = _EMPTY_OBJ

        def lru_cached(*args):
            if not self.args_non_hashable and not isinstance(arg, collections.Hashable):
                # No need to raise exception
                value = f(*args)
                return value
            elif self.args_non_hashable:
                # arguments are not hashable, best to convert it to string
                value = cache.get(str(args), empty_obj)
                if value is empty_obj:
                    value = f(*args)
                    cache.put(str(args), value)
                return value
            else:
                value = cache.get(args, empty_obj)
                if value is empty_obj:
                    value = f(*args)
                    cache.put(args, value)
                return value
        lru_cached._cache = cache
        return lru_cached
