# Changelog

## 0.1.40 (2025-08-28)

*Improvements:*

- Improved the error message in the function `validate_class_integrity` when the target goal is not valid.
- Now the function `validate_class_integrity` deduces the type of the target variable based on the specified target goal.

## 0.1.39 (2025-08-22)

*Improvements:*

- Improved the function `compute_admissible_btypes` so that it handles the dtype `bool` and gives a warning when the dtype is unknown.
- Improved the function `compute_columns_names_to_admissible_btypes` so that it uses the last version of the function `compute_admissible_btypes`.

## 0.1.38 (2025-08-21)

*Improvements:*

- New function `p_value_to_p_text` in the viz script.
- The function `p_value_to_p_text` is now used in the function `generate_insightsolver_banner`. This fixes a problem where the p-value is mpmath and not float64.

## 0.1.37 (2025-07-30)

*Improvements:*

- The method `i_to_print` has a new boolean parameter `do_print_shuffling_scores` to show a DataFrame of the shuffling scores of the rule at position `i`.
- The method `print` has a new boolean parameter `do_print_shuffling_scores`.
- The method `print_light` has a new boolean parameter `do_print_shuffling_scores`.
- The method `print_dense` has a new boolean parameter `do_print_shuffling_scores`.
- Simplified the implementation of the method `print_dense`.
- New method `plot` which is an alias for the method `show_all_feature_contributions_and_distributions`.
- Added the new `solver.plot()` method in the quick start of the `README.txt`.
- Updated the pdf documentation to include the new `print_dense` result with the `cohen_d`.

## 0.1.36 (2025-07-25)

*Improvements:*

- The method `print` has a new boolean parameter `do_print_last_separator`.
- The method `print_light` has a new boolean parameter `do_print_last_separator`.

## 0.1.35 (2025-07-24)

*Improvements:*

- Now the function `visualization/show_feature_distributions_of_S` handles better the situation where the feature is made of integers formatted as floats with useless `.0`.

## 0.1.34 (2025-07-24)

*Improvements:*

- New function `resolve_language` that validates the language provided by the user.
- Now the method `.fit` resolve the LLM language provided by the user.

## 0.1.33 (2025-07-23)

*Improvements:*

- Now the function `ingest_dict` does not overwrites the *btype* of the target variable by the one coming from the server. This fixes the problem where the target is continuous but the server converts it to a binary variable.

## 0.1.32 (2025-07-21)

*Improvements:*

- The function `show_all_feature_contributions` has a new parameter `bar_annotations`.
- The method `show_all_feature_contributions` has a new parameter `bar_annotations`.
- The function `show_feature_contributions_and_distributions_of_i` has a new parameter `bar_annotations`.
- The function `show_all_feature_contributions_and_distributions` has a new parameter `bar_annotations`.
- The method `show_all_feature_contributions_and_distributions` has a new parameter `bar_annotations`.
- Fixed another bug in the function `generate_insightsolver_banner`.

## 0.1.31 (2025-07-21)

*Improvements:*

- The function `show_feature_contributions_of_i` now has a new parameter `bar_annotations` which lets the user choose the kind of number at the end of the bars (it can be `p_value_ratio` or `p_value_contribution` or `None`).
- The method `show_feature_contributions_of_i` now has a new parameter `bar_annotations`.
- Fixed a bug in the function `generate_insightsolver_banner`.

## 0.1.30 (2025-07-18)

*Improvements:*

- Now the function `validate_class_integrity` is a little more strict on allowed modalities of the parameter `target_goal`.

## 0.1.29 (2025-07-18)

*Improvements:*

- The method `ingest_dict` now handles the *btypes* coming from the server. The *btypes* coming from the server overwrite the *btypes* in the solver. If there's a mismatch between both sides a warning is now shown. This new behaviour will help fill the missing *btypes* in the client-side.

## 0.1.28 (2025-07-17)

*Improvements:*

- The method `i_to_subrules_dataframe` is now compatible with `KL_divergence`.
- The method `i_to_print` is now compatible with `KL_divergence`.
- The method `to_dataframe` is now compatible with `KL_divergence`.

## 0.1.27 (2025-07-09)

*Improvements:*

- Updated the file `THIRD_PARTY_LICENSES.csv`.
- The class `InsightSolver` has a new boolean attribute `_is_fitted` that tells if the solver is fitted. By default it is set to `False`. When the fitting is done it is set to `True`.
- The class `InsightSolver` has a new callable method `.is_fitted()` that tells if the solver is fitted.

## 0.1.26 (2025-06-17)

*Improvements:*

- Completed the docstring of the method `solver.fit()`.
- Added a file `THIRD_PARTY_LICENSES.csv` to the repo.
- Updated the `README.md` file to mention the new file `THIRD_PARTY_LICENSES.csv`.

## 0.1.25 (2025-06-05)

*Improvements:*

- New function `get_credits_available` which is both independent of the solver and accessible via `from insightsolver import get_credits_available`.

## 0.1.24 (2025-06-05)

*Improvements:*

- New method `solver.convert_target_to_binary` to convert the target variable to a binary-valued Pandas Series.
- New method `solver.compute_mutual_information` to compute the mutual information between the features and the target variable. This can help the user decide which features to use.
- New method `solver.show_all_mutual_information` to show a bar plot of the mutual information between the features and the target variable.
- New function `show_all_mutual_information` in the file `visualization.py`.
- Fixed a bug in the file `requirements.txt` where `PIL` is now `Pillow`.
- Improved the docstrings and the pdf documentation.

## 0.1.23 (2025-06-03)

*Improvements:*

- Updated the readme: specifications about credits consumption.
- Updated the pdf documentation: specifications about credits consumption.
- Fixed a bug occuring in the `pip install .` of the module.

## 0.1.22 (2025-05-27)

*Improvements:*

- Updated the docstring of the class `InsightSolver`.
- Updated the docstring of the function `validate_class_integrity`.
- The function `validate_class_integrity` now has a new boolean parameter `do_strict_types` that determines the behaviour regarding specified types that are not admissible.
- The function `validate_class_integrity` now returns a dict of adjusted columns types.

## 0.1.21 (2025-05-08)

*Improvements:*

- The serialization `jsonpickle` has been completely removed both from the API client and server for security purposes.
- There are four new functions `encode_obj`, `convert_dict_to_json_string`, `decode_obj` and `convert_json_string_to_dict` in `api_utilities.py`. These functions are used as a safe substitute to `jsonpickle`.
- The key `'requested_action'` is now mandatory in the dict sent to the server.

## 0.1.20 (2025-05-06)

*Improvements:*

- The function `request_cloud_credits_infos` in `api_utilities` is now independent of the function `untransform_dict`. This update is meant to simplify the architecture of the API.

## 0.1.19 (2025-04-29)

*Improvements:*

- Added a new key `'requested_action'` in the dict sent to the server from the three functions `request_cloud_credits_infos`, `request_cloud_public_keys` and `request_cloud_computation`. This new key makes the API client and server more flexible and futur proof for eventual new kinds of requests other than cloud credits infos, cloud public keys or cloud computation. For now this new key is optional (to maintain compatibility with versions ≤0.1.18) but eventually it'll become mandatory.

## 0.1.18 (2025-04-28)

*Improvements:*

- Fixed a bug in the method `solver.to_dataframe()` where the new column `llm` was missing.

## 0.1.17 (2025-04-28)

*Improvements:*

- Fixed a bug in the function `show_feature_contributions_of_i` where `mpmath` was not imported.
- Fixed a bug in the function `search_best_ruleset_from_API_public` where for a local computation without service key, the LLM is now deactivated by default to avoid an error.

## 0.1.16 (2025-04-25)

*Improvements:*

- Updated `__init__.py` to avoid a bug where Pandas would be imported before being installed during the `pip install .`.
- New method `solver.i_to_S` that returns the rule `S` at position `i`.
- Fixed a bug in the function `S_to_index_points_in_rule`.
- Added some libraries in the file `requirements.txt`.
- Fixed a bug in the function `search_best_ruleset_from_API_public`.
- The documentation is now up to date.
- New file `version.py` that'll track the current version of the module.

*New visualization functions:*

- New folder `assets` that contains `.png` files used as headers for some figures.
- New file `visualization.py` that'll contain visualization functions for the InsightSolver module.
- New function `classify_variable_as_continuous_or_categorical` in `visualization.py`. This function is meant to classify a variable as continuous or categorical, which has an impact on how to plot the distribution of the variable.
- New function `compute_feature_label` in `visualization.py`. This function is meant to compute the label of a feature in a rule `S`.
- New function `show_feature_distributions_of_S` in `visualization.py`. This functions is meant to show the distributions of the points in the rule.
- New function `generate_insightsolver_banner` in `visualization.py`. This function is meant to generate the InsightSolver banner, which is used as a header in some figures.
- New function `show_feature_contributions_of_i` in `visualization.py`. This function is meant to generate a figure of feature contributions of the rule at index `i`.
- New function `show_all_feature_contributions` in `visualization.py`. This functions is meant to generate the feature contribution figures for all rules identified in the solver.
- New function `show_feature_contributions_and_distributions_of_i` in `visualization.py`. This functions is meant to generate the feature contribution figures and the feature distribution figures for the rule at position `i` in the solver.
- New function `show_all_feature_contributions_and_distributions` in `visualization.py`. This functions is meant to generate the feature contribution figures and the feature distribution figures for all rules found in a fitted solver.

*New InsightSolver methods:*

- Most of the new visualization functions are now also implemented as new methods for the class `InsightSolver`: `show_feature_distributions_of_S`, `show_feature_contributions_of_i`, `show_all_feature_contributions` and `show_all_feature_contributions_and_distributions`.

## 0.1.15 (2025-04-23)

*Improvements:*

- The function `search_best_ruleset_from_API_public` now has six new parameters `columns_names_to_descr`, `do_llm_readable_rules`, `llm_source`, `llm_language`, `do_store_llm_cache`, `do_check_llm_cache`. These parameters are meant to convert the identified rules to human readable text.
- The method `solver.fit` now has these last five new parameters.
- New method `solver.i_to_readable_text` meant to show the human readable text of the rule `i` if it is available.
- The methods `solver.i_to_print` and `solver.print` now show the human readable text of the rules if it is available.
- Now it is possible to do `insightsolver.__version__` to get the current version of the module.
- The header of `__init__.py` is slightly improved.
- Fixed a bug in the documentation.
- The parameter `threshold_M_max` can now be set up to `40000` instead of `10000`.

## 0.1.14 (2025-04-14)

*Improvements:*

- Fixed a bug in the function `S_to_index_points_in_rule` that caused it to crash when the type of a feature was not specified in the solver.
- Fixed a bug in the function `S_to_index_points_in_rule` that caused it to crash when a modality was not in the filtered DataFrame.
- Fixed a bug in the function `S_to_index_points_in_rule` that caused it to not return the good index under some circumstances.

## 0.1.13 (2025-04-09)

*Improvements:*

- The function `validate_class_integrity` not validates more parameters of the solver before fitting it via the API.
- New function `compute_admissible_btypes` that computes the admissible *btypes* for a given column.
- New function `compute_columns_names_to_admissible_btypes` that computes the admissible *btypes* for each columns.

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