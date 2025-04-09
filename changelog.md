# Changelog

## 0.1.12 (2025-04-08)

*Improvements:*

- New function `api_utilities/compute_credits_from_df` that computes the amount of credits consumed for a given DataFrame.
- New function `api_utilities/request_cloud_credits_infos` that makes a request to get informations about credits available.
- New method `solver.get_credits_needed_for_computation` that computes the amount of credits needed for the computation involved in fitting the solver. 
- New method `solver.get_df_credits_infos` that retrieves from the API server the transactions involving credits.
- New method `solver.get_credits_available` that retrieves from the API server the number of credits available.
- The method `solver.fit` now checks if there is enough credits available before sending data to the API server.

## 0.1.11 (2025-04-01)

*Improvements:*

- New methods `to_excel` and `to_excel_string` for the class `InsightSolver`.

## 0.1.10 (2025-03-26)

*Improvements:*

- Added a parameter `do_compute_memory_usage` to the method `fit`.

## 0.1.9 (2025-03-14)

*Improvements:*

- Fixed a bug in the function `S_to_index_points_in_rule`.

## 0.1.8 (2025-02-24)

*Improvements:*

- New function `S_to_index_points_in_rule` that takes a rule `S` and extracts the index of the points of `df` that lie inside `S`.
- New method `S_to_index_points_in_rule` for the class `InsightSolver`. This method generates the Pandas Index of the points of `df` that lie inside `S`.
- New method `S_to_s_points_in_rule` for the class `InsightSolver`. This method generates a boolean Pandas Series that tells if a point lies inside `S`.
- New method `S_to_df_filtered` for the class `InsightSolver`. This method generates a Pandas DataFrame that obtained by keeping only the rows of `df` that lie inside `S`.

## 0.1.7 (2025-02-21)

*Improvements:*

- A new attribute `other_modalities` is added to the class `InsightSolver`. This attribute contains a dict that tells which modalities are sent to the string `'other'` during the pre-processing phase. The new attribute `other_modalities` is compatible with the methods `ingest_dict` and `to_dict`.
- The method `ingest_dict` now ingests also `columns_descr`.
- The method `to_dict` now exports also `columns_descr`.

## 0.1.6 (2025-02-19)

*Improvements:*

- New attribute `monitoring_metadata` in the class `InsightSolver` that contains `p_value_min`, `Z_score_max`, `F_score_max` and `precision_p_values`. These are useful to benchmark the rules against shuffled data.
- The method `print` can now print the content of the attribute `monitoring_metadata`.
- The method `to_dict` not exports also the attribute `monitoring_metadata`.
- The method `ingest_dict` now imports also the attribute `monitoring_metadata`.

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