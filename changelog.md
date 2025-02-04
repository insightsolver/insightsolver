# Changelog

## 0.1.1 (2025-02-04)

- The class `InsightSolver` has a new method `to_dict` to export its content to a Python dictionary.
- The class `InsightSolver` has a new method `to_json_string` to export its content to a jsonpickle string.
- The class `InsightSolver` has a new method `to_dataframe` to export its content to a Pandas DataFrame.
- The class `InsightSolver` has a new method `to_csv` to export its content to a csv file and/or a csv string.
- Fixed a bug for the print of the benchmark scores when using mpmath p-values.
- New parameters `n_benchmark_original` and `n_benchmark_shuffle` and attribute `benchmark_scores` for the class `InsightSolver`.
- Integration of the Westfal-Young ratio (`wy_ratio`) to measure how robust the rules are against a shuffled target.
- Integration of the Cohen-d score (`cohen_d`) to measure how robust the rules are against a shuffled target.

## 0.1.0 (2024-11-18)

- First version of the InsightSolver API shared on GitHub.