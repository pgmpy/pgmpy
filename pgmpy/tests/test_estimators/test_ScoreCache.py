import unittest
from mock import Mock, MagicMock, call
from pgmpy.estimators.ScoreCache import LRUCache, ScoreCache
from pgmpy.estimators import BicScore
import pandas as pd


class TestScoreCache(unittest.TestCase):
    def test_caching(self):
        function_mock = Mock(side_effect=[1, 2])
        cache = LRUCache(function_mock, max_size=2)

        self.assertEqual(cache("key1"), 1)
        self.assertEqual(cache("key2"), 2)

        # Test will fail if the cache calls the function mock again
        self.assertEqual(cache("key2"), 2)
        self.assertEqual(cache("key1"), 1)
        self.assertEqual(cache("key1"), 1)

    def test_small_cache(self):
        function_mock = Mock(side_effect=lambda key: {"key1": 1, "key2": 2}[key])
        cache = LRUCache(function_mock, max_size=1)

        cache("key1")
        cache("key1")  # cached
        cache("key1")  # cached
        cache("key2")
        cache("key2")  # cached
        cache("key1")

        expected_function_calls = [call("key1"), call("key2"), call("key1")]
        function_mock.assert_has_calls(expected_function_calls, any_order=False)

    def test_remove_least_recently_used(self):
        function_mock = Mock(
            side_effect=lambda key: {"key1": 1, "key2": 2, "key3": 3}[key]
        )
        cache = LRUCache(function_mock, max_size=2)

        cache("key1")
        cache("key2")
        cache("key3")  # kicks out 'key1'
        cache("key2")  # cached
        cache("key1")

        expected_function_calls = [
            call("key1"),
            call("key2"),
            call("key3"),
            call("key1"),
        ]
        function_mock.assert_has_calls(expected_function_calls, any_order=False)

    def test_score_cache_invalid_scorer(self):
        with self.assertRaises(AssertionError) as e:
            ScoreCache("invalid_scorer", None)

    def test_score_cache(self):
        def local_scores(*args):
            if args == ("key1", ["key2", "key3"]):
                return -1
            if args == ("key2", ["key3"]):
                return -2

            assert False, "Unhandled arguments"

        data = pd.DataFrame({"key1": [1, 2], "key2": [1, 2]})
        base_scorer = MagicMock(spec=BicScore)
        base_scorer.local_score = Mock(side_effect=local_scores)
        cache = ScoreCache(base_scorer, data)

        self.assertEqual(cache.local_score("key1", ["key2", "key3"]), -1)
        self.assertEqual(cache.local_score("key1", ["key2", "key3"]), -1)  # cached
        self.assertEqual(cache.local_score("key2", ["key3"]), -2)
        self.assertEqual(cache.local_score("key2", ["key3"]), -2)  # cached

        expected_function_calls = [
            call("key1", ["key2", "key3"]),
            call("key2", ["key3"]),
        ]
        base_scorer.local_score.assert_has_calls(
            expected_function_calls, any_order=False
        )
