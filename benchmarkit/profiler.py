import time
import statistics
from collections.abc import Iterable
from itertools import groupby
from typing import Any, Callable, Dict, List, Literal, Tuple

import matplotlib.pyplot as plt
import multiprocess
import pandas
from memory_profiler import memory_usage
from multiprocess.managers import BaseProxy
from tqdm import tqdm


class ProfiledProcess(multiprocess.Process):
    """
    ProfiledProcess is a wrapper around multiprocessing.Process
    that allows to profile the exeecution time and memory consumption.
    """

    def __init__(
        self,
        target,
        args=(),
        kwargs={},
        nb_runs: int = 3,
        name: str = None,
        proxy: BaseProxy = None,
    ):
        super().__init__(target=target, args=args, kwargs=kwargs, name=name)
        self._name = name if name else target.__name__
        self._proxy = proxy if proxy is not None else multiprocess.Manager().dict()
        self._nb_runs = nb_runs

    def run(self):
        """
        Run the process and store the result in the proxy.
        """

        runs_results = []

        for _ in range(self._nb_runs):
            start_time = time.time()
            memory_result = memory_usage(
                proc=(self._target, self._args, self._kwargs),
                include_children=True,
                multiprocess=True,
                max_iterations=1,
                max_usage=True,
                timestamps=False,
            )
            end_time = time.time()

            time_result = end_time - start_time

            runs_results.append((time_result, memory_result))

        self._proxy[self._name] = (
            statistics.mean([t for t, _ in runs_results]),
            statistics.mean([m for _, m in runs_results]),
        )

    def result(self):
        return self._proxy.get(self._name)

    def profile(self):
        """
        Profile the process and return the result.
        """

        with self:
            self.start()
            self.join()

        return self.result()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.terminate()


class BenchmarkResults:

    __figures__ = {
        "time": {
            "title": "Time",
            "ylabel": "time (s)",
        },
        "memory": {
            "title": "Memory",
            "ylabel": "memory (MB)",
        },
    }

    def __init__(self, results: Dict):
        self._results = results

    def table(self) -> pandas.DataFrame:
        """
        Return a pandas DataFrame with the results.
        """
        table = pandas.DataFrame.from_dict(
            {
                (func, run_label): res
                for func, values in self._results.items()
                for run_label, res in values.items()
            },
            orient="index",
        )

        table.index.names = ["function", "run"]

        return table

    def dict(self, by: Literal["benchmark", "function"] = "function") -> dict:
        """
        Return a dictionary with the results.

        Parameters
        ----------
        by : Literal["benchmark", "function"], default="function"
            The key to use to group the results.
        returns : dict
            A dictionary with the results.
        """
        if by == "function":
            return self._results

        if by == "benchmark":
            return {
                "time": self._benchmark_dict("time"),
                "memory": self._benchmark_dict("memory"),
            }

        return self._results

    def _benchmark_dict(self, benchmark: Literal["time", "memory"]):

        benchmark_dict = {}

        for k, group in groupby(
            {
                (func, label): value
                for func, func_results in self._results.items()
                for label, benchmarks in func_results.items()
                for benchmark_label, value in benchmarks.items()
                if benchmark_label == benchmark
            }.items(),
            key=lambda x: x[0][0],
        ):
            benchmark_dict[k] = {label: value for (_, label), value in group}

        return benchmark_dict

    def plot(
        self,
        theme: str = "default",
        figsize: tuple = (6, 6),
        figures: List[str] = ["time", "memory"],
    ):
        """
        Plot the results.

        Parameters
        ----------
        theme : str, default="default"
            The theme to use for the plot.
        figsize : tuple, default=(12, 6)
            The size of the figure.
        """

        data = {k: v for k, v in self.dict(by="benchmark").items() if k in figures}
        figsize = (figsize[0] * len(data), figsize[1])

        plt.style.use(theme)

        figure, axs = plt.subplots(1, len(figures), figsize=figsize, sharex=True)

        for ax, (benchmark, results) in zip(
            axs if isinstance(axs, Iterable) else [axs], data.items()
        ):
            for func, values in results.items():
                ax.plot(*(values.keys(), values.values()), label=func)

            ax.set_ylabel(self.__figures__[benchmark]["ylabel"])
            ax.set_title(self.__figures__[benchmark]["title"])
            ax.legend()

        figure.tight_layout(pad=5.0)


class BenchmarKit:
    @staticmethod
    def run(
        func: Callable,
        args: Tuple[Any] = (),
        kwargs: Dict[str, Any] = {},
        proxy: BaseProxy = None,
        res_label: str = None,
        precision: int = None,
    ) -> Tuple[float, float]:
        """
        Run a function and return the time and memory usage

        Prameters
        ---------
        func: Callable
            The function to run.
        args: tuple
            The arguments to pass to the function.
        kwargs: dict
            The keyword arguments to pass to the function.
        proxy: BaseProxy
            The proxy to use to store the results.

        Returns
        -------
        Tuple[float, float]
            The time (in seconds) and memory usage (in mb) of the function.
        """

        res_label = res_label if res_label else func.__name__

        results = ProfiledProcess(
            target=func, args=args, kwargs=kwargs, proxy=proxy, name=res_label
        ).profile()

        if precision:
            results = tuple(round(x, precision) for x in results)

        return results

    @staticmethod
    def benchmark(
        funcs: List[Callable],
        args: List[Any],
        labels: List[str] = [],
        precision: int = None,
    ):
        """
        Benchmark a list of functions with a list of arguments.

        Prameters
        ---------
        funcs: List[Callable]
            List of functions to benchmark.
        args: List[Any]
            List of arguments to pass to the functions.
        labels: List[str]

        Returns
        -------
        BenchmarkResults
            A BenchmarkResults object containing the results of the benchmark.
        """

        results = {}
        funcs = funcs if isinstance(funcs, Iterable) else [funcs]
        labels = labels if labels else [f"run {i}" for i in range(1, len(args) + 1)]

        with multiprocess.Manager() as manager:

            proxy = manager.dict()

            with tqdm(total=len(funcs) * len(args), colour="green") as pbar:

                for func in funcs:
                    results[func.__name__] = {}

                    for label, arg in zip(labels, args):
                        exec_time, memory = BenchmarKit.run(
                            func,
                            args=arg,
                            proxy=proxy,
                            res_label=f"{func.__name__}.{label}",
                            precision=precision,
                        )

                        results[func.__name__][label] = {
                            "time": exec_time,
                            "memory": memory,
                        }

                        pbar.update()

        return BenchmarkResults(results)
