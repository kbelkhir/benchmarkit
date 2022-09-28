import unittest
from typing import List, Tuple

import numpy
import pandas
from benchmarkit import BenchmarKit, BenchmarkResults


class BenchmarKitTestCase(unittest.TestCase):
    def test_benchmark_function(self):
        data = list(numpy.random.randint(10, size=10**6))

        def func(l: List[int]):
            s = ",".join([str(x) for x in l])

        exec_time, memory = BenchmarKit.run(func, (data,))

        self.assertTrue(isinstance(exec_time, float))
        self.assertTrue(isinstance(memory, float))

    def test_benchmarking_single_function_with_multiples_inputs(self):
        def func(data):
            _ = ",".join([str(x) for x in data])

        results = BenchmarKit.benchmark(
            funcs=[func],
            args=[
                (list(numpy.random.randint(9, size=10)),),
                (list(numpy.random.randint(9, size=10**3)),),
                (list(numpy.random.randint(9, size=10**6)),),
            ],
            labels=["10", "1000", "1000000"],
        )

        self.assertBenchmarkCombinations(
            combinations=[
                (func.__name__, label) for label in ["10", "1000", "1000000"]
            ],
            results=results,
        )

    def test_benchmarking_multiple_functions_with_multiples_inputs(self):
        def func1(l: List[int]):
            _ = ",".join([str(x) for x in l])

        def func2(l: List[int]):
            _ = ",".join([str(x) for x in l])

        results = BenchmarKit.benchmark(
            funcs=[func1, func2],
            args=[
                (list(numpy.random.randint(9, size=10)),),
                (list(numpy.random.randint(9, size=10**3)),),
                (list(numpy.random.randint(9, size=10**6)),),
            ],
            labels=["10", "1000", "1000000"],
        )

        self.assertBenchmarkCombinations(
            combinations=[
                (f.__name__, l)
                for f in [func1, func2]
                for l in ["10", "1000", "1000000"]
            ],
            results=results,
        )

    def assertBenchmarkCombinations(
        self, combinations: List[Tuple], results: BenchmarkResults
    ):

        table = results.table()

        self.assertTrue(
            all(
                pandas.MultiIndex.from_tuples(
                    combinations,
                    names=["function", "run"],
                )
                == table.index
            )
        )

        for combination in combinations:
            self.assertTrue(isinstance(table.loc[combination].time, float))
            self.assertTrue(isinstance(table.loc[combination].memory, float))
