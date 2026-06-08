"""Representation proxy benchmarks backed by cached frozen ESM embeddings."""

__all__ = ["BenchmarkRunner", "DEFAULT_PROBE_SPECS"]


def __getattr__(name):
    # Keep package import side effects minimal to avoid circular imports in
    # training paths that only need submodules (e.g., probe_benchmark.interplm).
    if name in {"BenchmarkRunner", "DEFAULT_PROBE_SPECS"}:
        from .pipeline import DEFAULT_PROBE_SPECS, BenchmarkRunner

        if name == "BenchmarkRunner":
            return BenchmarkRunner
        return DEFAULT_PROBE_SPECS
    raise AttributeError(name)
