"""Tests for benchmark dataset loaders and metadata helpers."""

import gzip

import pandas as pd
import pytest

import pgmpy.datasets.benchmark as benchmark


def _gzip_bif_bytes() -> bytes:
    bif_text = """
network unknown {
}

variable A {
  type discrete [ 2 ] { no, yes };
}

probability ( A ) {
  table 0.5, 0.5;
}
""".strip()
    return gzip.compress(bif_text.encode("utf-8"))


class _MockResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def read(self):
        return self.payload


def test_alarm_loader_download_cache_and_force_download(monkeypatch, tmp_path):
    """Benchmark files are cached and only re-downloaded when forced."""
    payload = _gzip_bif_bytes()
    checksum = benchmark.hashlib.sha256(payload).hexdigest()

    monkeypatch.setitem(
        benchmark._BENCHMARK_SOURCES,
        "alarm",
        benchmark._BenchmarkSource(
            name="alarm",
            source_url="https://example.invalid/alarm.bif.gz",
            checksum_sha256=checksum,
            citation="test-citation",
            license="CC-BY 4.0",
            num_nodes=1,
            num_edges=0,
            variables=("A",),
            description="test",
        ),
    )
    monkeypatch.setattr(benchmark, "PGMPY_DATA_HOME", str(tmp_path))

    call_count = {"n": 0}

    def _urlopen(*args, **kwargs):
        call_count["n"] += 1
        return _MockResponse(payload)

    monkeypatch.setattr(benchmark, "urlopen", _urlopen)

    raw1 = benchmark._download_if_needed(dataset_name="alarm")
    assert isinstance(raw1, bytes)
    assert call_count["n"] == 1

    raw2 = benchmark._download_if_needed(dataset_name="alarm")
    assert raw1 == raw2
    assert call_count["n"] == 1

    benchmark._download_if_needed(dataset_name="alarm", force_download=True)
    assert call_count["n"] == 2


def test_asia_loader_is_deterministic_with_sample_id(monkeypatch, tmp_path):
    """Using the same sample_id should produce deterministic samples."""
    class _DummyModel:
        def simulate(self, n_samples, seed=None, show_progress=False):
            return pd.DataFrame({"A": [seed] * n_samples})

    monkeypatch.setattr(benchmark, "_load_model", lambda *args, **kwargs: _DummyModel())

    df1 = benchmark.load_asia(n_samples=20, sample_id="fold-1")
    df2 = benchmark.load_asia(n_samples=20, sample_id="fold-1")
    assert df1.equals(df2)


def test_invalid_n_samples_raises_value_error(monkeypatch, tmp_path):
    """Non-positive n_samples must raise a clear error."""
    class _DummyModel:
        def simulate(self, n_samples, seed=None, show_progress=False):
            return pd.DataFrame({"A": [seed] * n_samples})

    monkeypatch.setattr(benchmark, "_load_model", lambda *args, **kwargs: _DummyModel())

    with pytest.raises(ValueError, match="n_samples must be a positive integer"):
        benchmark.load_alarm(n_samples=0)


def test_get_benchmark_metadata():
    """Metadata helper returns all expected stable fields."""
    metadata = benchmark.get_benchmark_metadata("alarm")
    assert metadata["name"] == "alarm"
    assert "source_url" in metadata
    assert "citation" in metadata
    assert "license" in metadata
    assert "num_nodes" in metadata
    assert "num_edges" in metadata
    assert "variables" in metadata
    assert "description" in metadata


def test_invalid_sample_id_raises_value_error(monkeypatch, tmp_path):
    """Invalid sample_id values should be rejected."""
    class _DummyModel:
        def simulate(self, n_samples, seed=None, show_progress=False):
            return pd.DataFrame({"A": [seed] * n_samples})

    monkeypatch.setattr(benchmark, "_load_model", lambda *args, **kwargs: _DummyModel())

    with pytest.raises(ValueError, match="sample_id must be"):
        benchmark.load_alarm(n_samples=10, sample_id=-1)


def test_checksum_mismatch_message_contains_force_download_hint(monkeypatch, tmp_path):
    """Checksum mismatch errors should include refresh guidance."""
    payload = _gzip_bif_bytes()

    monkeypatch.setitem(
        benchmark._BENCHMARK_SOURCES,
        "alarm",
        benchmark._BenchmarkSource(
            name="alarm",
            source_url="https://example.invalid/alarm.bif.gz",
            checksum_sha256="badchecksum",
            citation="test-citation",
            license="CC-BY 4.0",
            num_nodes=1,
            num_edges=0,
            variables=("A",),
            description="test",
        ),
    )
    monkeypatch.setattr(benchmark, "PGMPY_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(benchmark, "urlopen", lambda *args, **kwargs: _MockResponse(payload))

    with pytest.raises(ValueError, match="force_download=True"):
        benchmark.load_alarm(n_samples=10)
