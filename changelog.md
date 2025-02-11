# Changelog

## 0.1.5 (2025-02-11)

*Improvements:*

- Fixed a bug in `solver.print()` where the shown number of tests against shuffled data was wrong.

## 0.1.4 (2025-02-10)

*Improvements:*

- New function `hash_string` in the script `api_utilities.py`.
- The function `search_best_ruleset_from_API_dict ` in `api_utilities.py` has a new parameter `user_email`. When running inside a Google Cloud Run container, the hash of the user's email (`email_hash`) is sent to the server instead of informations from a service key. This is useful when using the InsightSolver API from a web app deployed in GCP without a service key.
- The function `search_best_ruleset_from_API_public` in `insightsolver.py` has a new parameter `user_email`.
- The method `InsightSolver.fit()` has a new parameter `user_email`.

## 0.1.3 (2025-02-07)

*Improvements:*

- Now the service key is only required for remote cloud computation when the API client is running outside a Google Cloud Run container.

## 0.1.2 (2025-02-05)

*Bug fixing:*

- Fixed a bug in the method `to_csv` where the generated string was filled with `np.float64` and `np.int64`.

## 0.1.1 (2025-02-04)

*Class `InsightSolver`:*

- New method `to_dict` to export its content to a Python dictionary.
- New method `to_json_string` to export its content to a jsonpickle string.
- New method `to_dataframe` to export its content to a Pandas DataFrame.
- New method `to_csv` to export its content to a csv file and/or a csv string.
- New parameter and attribute `filtering_score` to specify the filtering score used.
- New parameters `n_benchmark_original` and `n_benchmark_shuffle`.
- New attribute `benchmark_scores` that contains the Westfal-Young ratio (`wy_ratio`) and the Cohen-d score to measure how much the rules are robust against a shuffled target.

*Bug fixing:*

- Fixed a bug for the print of the benchmark scores when using mpmath p-values.

*Improvements:*

- The function `validate_class_integrity` now has a new parameter `filtering_score`.

## 0.1.0 (2024-11-18)

- First version of the InsightSolver API shared on GitHub.