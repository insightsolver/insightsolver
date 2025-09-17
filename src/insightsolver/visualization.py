"""
* `Organization`:  InsightSolver Solutions Inc.
* `Project Name`:  InsightSolver
* `Module Name`:   insightsolver
* `File Name`:     visualization.py
* `Authors`:       Noé Aubin-Cadot <noe.aubin-cadot@insightsolver.com>,
				   Arthur Albo <arthur.albo@insightsolver.com>
* `Last Updated`:  2025-09-10
* `First Created`: 2025-04-24

Description
-----------
This file contains some visualization functions, some of which are integrated as a method of the InsightSolver class.

Functions provided
------------------

- show_all_mutual_information
- classify_variable_as_continuous_or_categorical
- compute_feature_label
- truncate_label
- show_feature_distributions_of_S_feature
- show_feature_distributions_of_S
- p_value_to_p_text
- generate_insightsolver_banner
- wrap_text_with_word_boundary
- show_feature_contributions_of_i
- show_all_feature_contributions
- show_feature_contributions_and_distributions_of_i
- show_all_feature_contributions_and_distributions

License
-------
Exclusive Use License - see `LICENSE <license.html>`_ for details.

----------------------------

"""

################################################################################
################################################################################
# Import some libraries

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from typing import Optional, Union, Dict, Sequence

################################################################################
################################################################################
# Defining some visualization functions

def show_all_mutual_information(
	solver,
	n_samples:Optional[int] = 1000,
	n_cols:Optional[int]    = 20,
	kind: str               = 'barh',
):
	"""
	This function generates a bar plot of the mutual information between the features and the target variable.

	Parameters
	----------
	n_samples: int
		An integer that specifies the number of data rows to use in the computation of the mutual information.
	n_cols: int
		An integer that specifies the maximum number of features to show
	kind: str
		Kind of plot ('bar' or 'barh')
	"""
	# Make sure the parameter kind is valid
	if kind not in ['bar','barh']:
		raise ValueError(f"ERROR (show_all_mutual_information): The parameter kind='{kind}' must be either 'bar' or 'barh'.")

	# Compute the mutual information
	s_mi = solver.compute_mutual_information(
		n_samples = n_samples,
	)
	# Keep only the top variables
	if n_cols and len(s_mi)>n_cols:
		s_mi = s_mi.head(n_cols)
	# For a horizontal barplot we must sort to have big values on top of the figure
	if kind=='barh':
		s_mi.sort_values(ascending=True,inplace=True)

	# Generate the figure
	import matplotlib.pyplot as plt
	plt.figure(figsize=(12, 6))
	ax = s_mi.plot(
		kind      = kind,
		edgecolor = 'black',
	)
	plt.title('Mutual Information between the features and the target variable')
	plt.ylabel('Mutual Information')
	plt.xlabel('Feature')
	plt.xticks(rotation=45, ha='right')
	for idx, value in enumerate(s_mi):
		# Compute the position
		if kind=='bar':
			x = idx
			y = value + max(s_mi) * 0.01
			ha = 'center'
			va = 'bottom'
		elif kind=='barh':
			x = value + max(s_mi) * 0.005
			y = idx
			ha = 'left'
			va = 'center'
		ax.text(
			x        = x, 
			y        = y,  # small offset
			s        = f"{value:.4f}", 
			ha       = ha, 
			va       = va, 
			fontsize = 8
		)
	plt.tight_layout()
	# Show the figure
	plt.show()

def classify_variable_as_continuous_or_categorical(
	s: pd.Series,
	unique_ratio_threshold: float = 0.1,
	max_categories: int           = 20,
) -> str:
	"""
	Classify a pandas Series as 'continuous' or 'categorical'.

	Heuristic
	---------
	- If dtype is object/string/bool → categorical
	- If all values are equal → categorical
	- If all values are integers:
	  - Few unique values (<= max_categories) → categorical
	  - Low unique ratio (<= unique_ratio_threshold) → categorical
	- Otherwise → continuous

	Parameters
	----------
	s : pd.Series
		Input series.
	unique_ratio_threshold : float, optional
		Threshold for ratio (#unique / #non-missing) to treat integers as categorical.
	max_categories : int, optional
		Absolute cap for number of unique categories to treat as categorical.

	Returns
	-------
	str
		"categorical" or "continuous"
	"""

	# On vérifie le dtype
	if s.dtype in ["object", "string", "bool"]:
		return "categorical"

	# On élimine les valeurs manquantes
	s = s.dropna()

	# On regarde s'il est de longueur nulle
	if s.empty:
		return "categorical"

	# On regarde s'il est constant
	if s.nunique() == 1:
		return "categorical"

	# On regarde s'il ne contient que des entiers
	all_integers = all(s.astype(float).apply(float.is_integer))

	# Calculer le nombre de valeurs uniques
	unique_values = s.nunique()

	# Calculer la proportion de valeurs uniques sur la longueur de s
	unique_ratio = unique_values / len(s)

	if all_integers:
		if unique_values <= max_categories:
			return "categorical"
		if unique_ratio <= unique_ratio_threshold:
			return "categorical"

	return "continuous"

def compute_feature_label(
	solver,              # The solver
	feature_name: str,   # The name of the feature
	S: dict,             # The rule S
)->[str,str]:
	"""
	This function computes the label of a feature in a rule S.

	Parameters
	----------
	solver: InsightSolver
		The solver.
	feature_name: str
		The name of the feature.
	S: dict
		The rule S.

	Returns
	-------
	feature_label: str
		The label of the feature.
	feature_relationship: str
		The relationship of the feature to the constraints.
	"""
	# Make sure feature_name is in S
	if feature_name not in S.keys():
		raise Exception(f"ERROR (compute_feature_label): feature_name={feature_name} is not in the keys of S.")
	# Look at the type of data
	if isinstance(S[feature_name],list):
		# If it's a continuous feature
		# Take the boundaries specified by the continuous feature
		if isinstance(S[feature_name][0],list):
			# If it's a continuous feature with NaNs
			[[rule_min,rule_max],rule_nan] = S[feature_name]
		else:
			# If it's a continuous feature without NaNS
			rule_min,rule_max = S[feature_name]
			rule_nan = None
		# Take the min and max according to the data
		min_value = solver.df[feature_name].min()
		max_value = solver.df[feature_name].max()
		# Depending on the rule and the data we compute the label
		if (rule_min==min_value)&(rule_max==max_value):
			# If both boundaries seem meaningless
			if rule_min==rule_max:
				# If only one value is legitimate
				feature_label = f"{feature_name} = {rule_max}"
				feature_relationship = '='
			else:
				feature_label = f"{feature_name} ∈ ℝ"
				feature_relationship = '∈'
		elif rule_min==min_value:
			# If only the lower boundary is meaningless
			feature_label = f"{feature_name} ≤ {rule_max}"
			feature_relationship = '≤'
		elif rule_max==max_value:
			# If only the upper boundary is meaningless
			feature_label = f"{feature_name} ≥ {rule_min}"
			feature_relationship = '≥'
		else:
			# If both boundaries are meaningful
			feature_label = f"{feature_name} ∈ {[rule_min,rule_max]}"
			feature_relationship = '∈'
		if rule_nan:
			feature_label += f", {rule_nan}"
	elif isinstance(S[feature_name],set):
		# If it's a binary or multiclass feature with at least one possible value
		feature_label = f"{feature_name} ∈ {S[feature_name]}"
		feature_relationship = '∈'
	else:
		# If it's a binary or multiclass feature with only one possible value
		feature_label = f"{feature_name} = {S[feature_name]}"
		feature_relationship = '='
	# Return the feature label and the feature relationship
	return feature_label,feature_relationship

def truncate_label(
	label,
	max_length = 30,
	asterisk   = False,
):
	"""
	This function truncates a string if it exceeds a specified length, adding an ellipsis.

	Parameters
	----------
	label: string
		the feature rule's modalities.
	max_length: int
		the maximum number of character accepted.
	asterisk: bool
		whether we want an asterisk to appear after the truncation.

	Returns
	-------
	truncated_label: str
		The truncated label.
		
	"""
	if len(label) > max_length:
		truncated_label = label[:max_length-1] + '…'
		if asterisk:
			truncated_label += '*'
	else:
		truncated_label = label
	return truncated_label

def show_feature_distributions_of_S_feature(
	solver,
	df_filtered: pd.DataFrame,
	S: dict,
	feature_name: str,
	missing_value: str           = False,
	ax: str                      = None,
	language: str                = 'en',
	padding_y: int               = 5,
	do_show_kde: bool            = False,
	do_show_vertical_lines: bool = False,
)->None:
	"""
	This function generates bar plots of the distributions of the points in the specified rule S for a given feature.

	Parameters
	----------
	solver : InsightSolver
		The solver object.
	df_filtered: pd.DataFrame
		The filtered data according to the rule S.
	S : dict
		The rule S that we wish to visualize.
	feature_name : str
		The name of the column
	missing_value: bool
		If we want to show the graph for the present values or the missing values.
	ax: matplotlib.axes
		Axes to be used if provided.
	language: str
		Language to be used.
	padding_y: int
		The padding used for the ylim.
	do_show_kde: bool
		Boolean to show the KDE of the continuous features.
	do_show_vertical_lines: bool
		If we want to show vertical lines.
	"""

	# Determine if a new figure needs to be created
	if ax is None:
		# Take the size of a pixel instead of inches
		px = 1/plt.rcParams['figure.dpi']
		if missing_value:
			fig, ax = plt.subplots(figsize=(200*px, 4)) 
		else:
			fig, ax = plt.subplots(figsize=(1000*px, 4))
		do_early_show = True
	else:
		do_early_show = False

	# Take the DataFrame that contains the data
	df = solver.df
	# Take the Pandas Series of the feature data
	s_unfiltered = df[feature_name]
	# Take the data without the missing values
	s_unfiltered_dropna = s_unfiltered.dropna()
	# Take the Pandas Series of the filtered feature data
	s_filtered   = df_filtered[feature_name]
	# Take the filtered data without the missing values
	s_filtered_dropna = s_filtered.dropna()

	# Take the btype of the feature
	if isinstance(S[feature_name],list):
		column_btype = 'continuous'
	else:
		column_btype = 'multiclass'
	# Determine if the variable is to be shown as a continuous (i.e. histogram) or as a categorical (i.e. bars)
	if column_btype in ['binary','multiclass']:
		categorical_or_continuous = 'categorical'
	elif column_btype=='continuous':
		categorical_or_continuous = classify_variable_as_continuous_or_categorical(
			s = s_unfiltered,
		)
	else:
		raise Exception(f"ERROR: feature_name='{feature_name}' has a btype='{column_btype}' which is illegal.")

	# Look at the type of feature
	if categorical_or_continuous=='continuous':
		# If the feature is continuous

		# Calculate the inter quartile range (IQR)
		Q1 = s_unfiltered_dropna.quantile(0.25)
		Q3 = s_unfiltered_dropna.quantile(0.75)
		IQR = Q3 - Q1
		# Take the number of observations
		n_rows = len(s_unfiltered_dropna)
		# Look at the min and max values
		min_value = s_unfiltered_dropna.min()
		max_value = s_unfiltered_dropna.max()
		# Compute the widths of the bins
		if IQR>0:
			# Freedman-Diaconis formula
			step_bins = 2 * IQR * n_rows ** (-1 / 3)
		elif min_value<max_value:
			# Sturges formula
			step_bins = (max_value - min_value) / (1 + np.log2(n_rows))
		else:
			# 1 by default
			step_bins = 1
		# Calculate the number of bins based on the range and the step size
		num_bins = round((max_value - min_value) / step_bins)  # Nombre de bins correct
		if num_bins==0:
			num_bins = 1
		# Limit the total number of bins to avoid an over segmentation
		max_bins = 30
		num_bins = min(num_bins, max_bins)
		# Adjust the width of the bins to the limited number of bins
		if min_value<max_value:
			step_bins = (max_value - min_value) / num_bins
		else:
			step_bins = 1
		# Create the bin edges for the histograms
		bin_edges = np.arange(
			min_value,
			max_value + step_bins,
			step_bins,
		)

	if missing_value:
		
		# Create a Pandas Series of the missing values of the unfiltered data
		s_unfiltered_na = s_unfiltered[s_unfiltered.isna()].replace({np.nan: "nan"})
		# Create a Pandas Series of the missing values of the filtered data
		s_filtered_na   = s_filtered[s_filtered.isna()].replace({np.nan: "nan"})
		# First grey bar for the number of missing values in the original data
		sns.countplot(
			x     = s_unfiltered_na,
			color = 'grey',
			alpha = 0.6,
			ax    = ax,
		)
		# Superpose a second green bar for the number of missing values in the filtered data
		sns.countplot(
			x     = s_filtered_na,
			color = 'green',
			alpha = 0.6,
			ax    = ax,
		)
		# Remove legend
		if ax.get_legend() is not None:
			ax.get_legend().remove()
		# Hide the title and xlabel and ylabel
		ax.set(
			title  = '',
			xlabel = '',
			ylabel = '',
		)

	else:
		# If we are not in the scenario of showing missing values

		# Look at the type of feature
		if categorical_or_continuous=='continuous':
			# First histplot for the distribution of the original variable
			sns.histplot(
				data  = s_unfiltered,
				kde   = do_show_kde,
				bins  = bin_edges,
				color = 'grey',
				alpha = 0.6,
				ax    = ax,
			)
			# Second plot for the distribution of the filtered variable by the rule
			sns.histplot(
				data  = s_filtered,
				bins  = bin_edges,
				color = 'green',
				alpha = 0.6,
				ax    = ax,
			)
			# Rotate the bin edges
			ax.set_xticks(bin_edges)
			# Adjust the xlim
			ax.set_xlim(s_unfiltered.min() - step_bins, s_unfiltered.max()+step_bins)

		elif categorical_or_continuous=='categorical':
			# Take the Pandas Series to show in the countplot

			# If the data seems to be integers formatted as floats with useless .0, remove the .0 to improve the figure
			if pd.api.types.is_float_dtype(s_unfiltered_dropna) and np.all(s_unfiltered_dropna == s_unfiltered_dropna.astype(int)):
				s_unfiltered_dropna = s_unfiltered_dropna.astype(int).copy()
				s_filtered_dropna   = s_filtered_dropna.astype(int).copy()
			# Take the non numerical columns
			non_num_cols = df.select_dtypes(exclude='number').columns
			# If the feature is a non numerical column
			if feature_name in non_num_cols:
				# Ensure we only get unique values from the original data
				unique_categories = s_unfiltered_dropna.astype(str).unique() # Convert to string for consistent sorting
				sorted_categories = sorted(unique_categories)
			# First countplot for the distribution of the original variable
			sns.countplot(
				x     = s_unfiltered_dropna,
				color = 'grey',
				alpha = 0.6,
				label = "Unfiltered",
				order = sorted_categories if feature_name in non_num_cols else None, # Apply alphabetical order
				ax    = ax,
			)
			# Second plot for the distribution of the filtered variable by the rule
			sns.countplot(
				x     = s_filtered_dropna,
				color = 'green',
				alpha = 0.6,
				label = "Filtered",
				order = sorted_categories if feature_name in non_num_cols else None, # Apply alphabetical order
				ax    = ax,
			)
		
		if do_show_vertical_lines:
			# Take the boundaries specified by the continuous feature
			if isinstance(S[feature_name],list):
				# Generate the feature label and the feature relationship
				_,feature_relationship = compute_feature_label(
					solver       = solver,
					feature_name = feature_name,
					S            = S,
				)
				# Take the rule
				if isinstance(S[feature_name][0],list):
					# If it's a continuous feature with NaNs
					[[rule_min,rule_max],rule_nan] = S[feature_name]
				else:
					# If it's a continuous feature without NaNS
					rule_min,rule_max = S[feature_name]
				# Add a vertical line
				if feature_relationship=='≥':
					# Add a vertical line at the lower boundary
					ax.axvline(rule_min, color='green', linestyle='--', label=feature_name+' min')
				elif feature_relationship=='≤':
					# Add a vertical line at the upper boundary
					ax.axvline(rule_max, color='green', linestyle='--', label=feature_name+' max')
				elif feature_relationship=='∈':
					# Add vertical lines at both boundaries
					ax.axvline(rule_min, color='green', linestyle='--', label=feature_name+' min')
					ax.axvline(rule_max, color='green', linestyle='--', label=feature_name+' max')
				   
		# Generate the title
		if language=='fr':
			title = f"Distribution de la variable: {feature_name}"
		elif language=='en':
			title = f"Distribution Plot for {feature_name}"
		else:
			title = f"Distribution Plot for {feature_name}"
		ax.set_title(title)
		# Generate the xlabel
		plt.xlabel(feature_name)

		# Add custom legend
		import matplotlib.patches as mpatches
		grey_patch  = mpatches.Patch(color="grey",  alpha=0.6, label="Hors de la règle" if language == 'fr' else "Outside the rule")
		green_patch = mpatches.Patch(color="green", alpha=0.6, label="Dans la règle" if language == 'fr' else "Inside the rule")
		ax.legend(handles=[grey_patch, green_patch])

		# Get the current x-axis tick locations and labels
		locs, labels = ax.get_xticks(), ax.get_xticklabels()
		# Apply the truncation function to each label
		truncated_labels = [truncate_label(label.get_text()) for label in labels]
		# Set the xticks positions
		ax.set_xticks(locs)
		# Rotate x-axis tick labels diagonally
		ax.set_xticklabels(truncated_labels, rotation=30, ha="right")

	# Adjust the ylim so that the ylim is the same for the left and the right picture
	if categorical_or_continuous=='continuous':
		# Count the number of points per bin
		counts, _ = np.histogram(
			a    = s_unfiltered,
			bins = bin_edges,
		)
		# Take the maximum number of point found in a bin
		max_count_left = counts.max()
	elif categorical_or_continuous=='categorical':
		# If the feature is categorical
		max_count_left = s_unfiltered.value_counts().iloc[0]
	# Look at if there is any missing value in the original data	
	if s_unfiltered.isna().any():
		# Take the number of missing values
		max_count_right = s_unfiltered.isna().sum()
		# Update the maximum count
		max_count = max(max_count_left, max_count_right)
	else:
		max_count = max_count_left
	# Adjust y-lim
	ax.set_ylim(
		0,
		max_count + padding_y,
	)

	# If we want to show the plot now
	if do_early_show:
		# Tight layout
		plt.tight_layout()
		# Show the figure
		plt.show()

def show_feature_distributions_of_S(
	solver,
	S: dict,
	language: str                = 'en',
	padding_y: int               = 5,
	do_show_kde: bool            = False,
	do_show_vertical_lines: bool = False,
)->None:
	"""
	This function generates bar plots of the distributions of the points in the specified rule S.

	Parameters
	----------
	solver : InsightSolver
		The solver object.
	S : dict
		The rule S that we wish to visualize.
	language: str
		Language to use.
	padding_y: int
		The padding used for the ylim.
	do_show_kde: bool
		Boolean to show the KDE of the continuous features.
	do_show_vertical_lines: bool
		If we want to show some vertical lines.
	"""

	# Take the DataFrame that contains the data
	df = solver.df
	# Filter the data to the points that are in the rule S
	df_filtered = solver.S_to_df_filtered(S=S)
	# Take the size of a pixel instead of inches
	px = 1/plt.rcParams['figure.dpi']
	# Loop over the features in the rule S
	for feature_name in S.keys():
		# One figure will be created per feature name
		# Look at if the data of the feature contains any missing value
		if solver.df[feature_name].isna().any():
			# If the feature contains any missing value
			# Create two graphs (one for the present values and one for the missing values)
			fig, axes = plt.subplots(
				figsize     = (1446*px, 4),
				nrows       = 1,
				ncols       = 2,
				gridspec_kw = {
					'width_ratios': [15, 1],
				},
			)
			# Plot the graph for the present values to the left
			show_feature_distributions_of_S_feature(
				solver                 = solver,
				df_filtered            = df_filtered,
				S                      = S,
				feature_name           = feature_name,
				missing_value          = False,   # Plot for the present values
				ax                     = axes[0], # Left figure
				language               = language,
				padding_y              = padding_y,
				do_show_kde            = do_show_kde,
				do_show_vertical_lines = do_show_vertical_lines,
			)
			# Plot the graph for the missing values to the right
			show_feature_distributions_of_S_feature(
				solver                 = solver,
				df_filtered            = df_filtered,
				S                      = S,
				feature_name           = feature_name,
				missing_value          = True,    # Plot for the missing values
				ax                     = axes[1], # Right figure
				language               = language,
				padding_y              = padding_y,
				do_show_kde            = do_show_kde,
				do_show_vertical_lines = do_show_vertical_lines,
			)
		else:
			# If the feature does not contain any missing value
			# Create a single graph for the present values
			fig, ax = plt.subplots(
				figsize = (1446*px, 4),
			)
			# Plot the graph for the present values
			show_feature_distributions_of_S_feature(
				solver                 = solver,
				df_filtered            = df_filtered,
				S                      = S,
				feature_name           = feature_name,
				missing_value          = False, # Plot for the present values
				ax                     = ax,
				language               = language,
				padding_y              = padding_y,
				do_show_kde            = do_show_kde,
				do_show_vertical_lines = do_show_vertical_lines,
			)
		# Tight layout
		plt.tight_layout()
		# Show the figure
		plt.show()

def p_value_to_p_text(
	p_value,
	precision_p_values: str,
)->str:
	"""
	This function converts the p-value to a string.

	Parameters
	----------
	p_value: float or mpmath.mpf
		The p-value to convert.
	precision_p_values: str
		The precision of the p-values.

	Returns
	-------
	p_text: str
		The p_value formatted as a string.
	"""
	import mpmath
	if precision_p_values=='float64':
		# If the precision is float64
		if abs(p_value) >= 0.001: # If the p_value is big
			p_text = f"{p_value:.4f}"  # normal decimals
		else:
			p_text = f"{p_value:.2e}"  # scientific notation
	elif precision_p_values=='mpmath':
		# If the precision is mpmath
		if abs(p_value) >= 0.001: # If the p_value is big
			p_text = mpmath.nstr(p_value, n=5, strip_zeros=True)
		else:
			# Scientific notation : 2 significant numbers
			p_text = mpmath.nstr(p_value, n=2, min_fixed=0, max_fixed=0)
	else:
		raise Exception(f"ERROR: precision_p_values='{precision_p_values}' is invalid. It must be either 'float64' or 'mpmath'.")
	# Return the result
	return p_text

def generate_insightsolver_banner(
	solver,
	i:int,
	loss:Optional[float] = None,
):
	"""
	This function returns an image containing the parameters for p-value, purity, lift, coverage, size and loss value of a specified rule.

	Parameters
	----------
	solver: InsightSolver
		The solver.
	i: int
		Index of the rule.
	loss: float
		Some loss to show in the banner.

	Returns
	-------
	image: Image
		Image of the banner (with the values).
	"""
	
	from PIL import Image, ImageDraw
	
	# Take the rule at position i
	rule_i = solver.i_to_rule(i=i)

	# Take some scores of the rule
	p_value  = rule_i['p_value']
	purity   = rule_i['mu_rule']
	lift     = rule_i['lift']
	coverage = rule_i['coverage']
	size     = rule_i['m']

	# Take the precision of the p-values
	if 'precision_p_values' in solver.monitoring_metadata.keys():
		precision_p_values = solver.monitoring_metadata['precision_p_values']
	else:
		precision_p_values = 'float64'
	if precision_p_values=='mpmath':
		import mpmath

	# Convert the p-value to a string
	p_text = p_value_to_p_text(
		p_value            = p_value,
		precision_p_values = precision_p_values,
	)

	# Generate the banner
	if loss==None:
		# If loss is not specified

		# Import the banner
		from importlib.resources import files
		with (files("insightsolver.assets") / "insightbanner_no_loss.png").open("rb") as f:
			banner = Image.open(f).convert("RGBA").copy()
		# Draw the banner
		draw = ImageDraw.Draw(banner)
		# Font size
		font_size = 30
		# Draw the Id of the insight
		insight_id_text = "Insight #" + str(i+1)
		insight_id_position = (90, 20)
		draw.text(insight_id_position, insight_id_text, font_size=font_size, fill="black")
		# Draw the p-value
		p_position = (355, 20)
		draw.text(p_position, p_text, font_size=font_size, fill="black")
		# Draw the purity
		purity_text = str(round(purity*100, 2))+'%'
		purity_position = (595, 20)
		draw.text(purity_position, purity_text, font_size=font_size, fill="black")
		# Draw the lift
		lift_text = str(round(lift, 2))
		lift_position = (850, 20)
		draw.text(lift_position, lift_text, font_size=font_size, fill="black")
		# Draw the coverage
		cov_text = str(round(coverage*100,2))+'%'
		cov_position = (1080, 20)
		draw.text(cov_position, cov_text, font_size=font_size, fill="black")
		# Draw the size
		size_text = str(size)
		size_position = (1310, 20)
		draw.text(size_position, size_text, font_size=font_size, fill="black")
	
	else:
		# If loss is specified

		# Import the banner
		from importlib.resources import files
		with (files("insightsolver.assets") / "insightbanner_with_loss.png").open("rb") as f:
			banner = Image.open(f).convert("RGBA").copy()
		# Draw the banner
		draw = ImageDraw.Draw(banner)
		# Font size
		font_size = 25
		# Draw the Id of the Insight
		insight_id_text = "Insight #" + str(i+1)
		insight_id_position = (90, 22)
		draw.text(insight_id_position, insight_id_text, font_size=font_size, fill="black")
		# Draw the p-value
		p_position = (320, 22)
		draw.text(p_position, p_text, font_size=font_size, fill="black")
		# Draw the purity
		purity_text = str(round(purity*100, 2))+'%'
		purity_position = (555, 22)
		draw.text(purity_position, purity_text, font_size=font_size, fill="black")
		# Draw the lift
		lift_text = str(round(lift, 2))
		lift_position = (770, 22)
		draw.text(lift_position, lift_text, font_size=font_size, fill="black")
		# Draw the coverage
		cov_text = str(round(coverage*100,2))+'%'
		cov_position = (950, 22)
		draw.text(cov_position, cov_text, font_size=font_size, fill="black")
		# Draw the size
		size_text = str(size)
		size_position = (1160, 22)
		draw.text(size_position, size_text, font_size=font_size, fill="black")
		# Draw the loss
		loss_text = str(loss)
		loss_position = (1315, 22)
		draw.text(loss_position, loss_text, font_size=font_size, fill="black")

	# Return the banner
	return banner

def wrap_text_with_word_boundary(
	text: str,                  # The original string to modify.
	max_line_length: int = 150, # The character threshold for insertion.
) -> str:
	"""
	Wraps a text string into multiple lines by inserting line breaks 
	around a target character width, while preserving word boundaries 
	whenever possible.

	- If the next word would cause the line to exceed `max_line_length`,
	  a line break is inserted *before* that word.
	- If a single word is longer than `max_line_length`, the word is split
	  with a hyphen followed by a line break.

	Parameters
	----------
	text : str
		The input text to wrap.
	max_line_length : int, optional
		The maximum allowed line length before wrapping occurs (default is 150).
										   
	Returns
	-------
	str
		The wrapped string, with line breaks (and occasional hyphenation)
		inserted at appropriate positions.    
	"""

	# If the text is not a string, convert it to a string
	if not isinstance(text, str):
		text = str(text)
	# If the text is empty, return an empty text
	if text=='':
		return ''
	# Take the list of words
	words = text.split()
	# Create a list of strings
	strings = []
	# The current line
	current_len = 0
	# Looping over the words
	for word in words:
		# Case 1: the word longer than a single line and needs to be chunked down
		while len(word) > max_line_length:
			# Take the first chunk
			chunk = word[:max_line_length - 1] + "-"
			# Append the first chunk
			strings.append(chunk + "\n    ")
			# Take the last part of the word (stripped from the first chunk)
			word = word[max_line_length - 1:]
			# Reset the line because we are on a new line
			current_len = 0
		# Case 2: normal situation
		if current_len == 0:
			# If we are at the start of the line
			# append the word at the start of the string
			strings.append(word)
			# We moved a bit to the right of the line
			current_len = len(word)
		elif current_len + 1 + len(word) <= max_line_length:
			# If the word is not too long
			# we append the word to the strings
			strings.append(" " + word)
			# We moved a bit to the right of the line
			current_len += 1 + len(word)
		else:
			# If the word is too long
			# Normal jump of line
			strings.append("\n    " + word)
			current_len = len(word)
	# Join the resulting strings
	string = " ".join(strings)
	# Return the resulting string
	return string

def show_feature_contributions_of_i(
	solver,
	i: int,                        # Index of the rule to show
	a: float              = 0.5,   # Height per bar
	b: float              = 1,     # Height for the margins and other elements
	fig_width: float      = 12,    # Width of the figure
	language: str         = 'en',  # Language of the figure
	do_grid: bool         = True,  # If we want to show a vertical grid
	do_title: bool        = False, # If we want a title automatically generated
	do_banner: bool       = True,  # If we want to show the banner
	bar_annotations: str  = 'p_value_ratio', # Type of values to show at the end of the bars (can be 'p_value_ratio', 'p_value_contribution' or None)
	loss: Optional[float] = None,  # If we want to show a loss
)->None:
	"""
	This function generates a horizontal bar plots of the feature constributions of a specified rule ``S``.
	
	Parameters
	----------
	solver: InsightSolver
		The fitted solver that contains the identified rules.
	i: int
		The index of the rule to show.
	a: float
		Height per bar.
	b: float
		Added height to the figure.
	fig_width: float
		Width of the figure
	language: str
		Language of the figure ('fr' or 'en').
	do_grid: bool
		If we want to show a vertical grid behind the horizontal bars.
	do_title: bool
		If we want to show a title.
	do_banner: bool
		If we want to show the banner.
	bar_annotations: str
		Type of values to show at the end of the bars (can be 'p_value_ratio', 'p_value_contribution' or None)
	loss: float
		If we want to show a loss.
	"""
	# Take the rule i
	rule_i = solver.i_to_rule(i=i)
	# Take the rule S
	S = rule_i['rule_S']
	# Take the contributions of the features
	df_feature_contributions_S = solver.i_to_feature_contributions_S(
		i                      = i,
		do_rename_cols         = False,
	)
	# Append the p_value_ratio
	d_p_value_ratios_S = rule_i['p_value_ratio_S']
	df_feature_contributions_S["p_value_ratio"] = df_feature_contributions_S.index.map(d_p_value_ratios_S)
	# Append the labels
	feature_names = df_feature_contributions_S.index.to_list() # List of features names of the rule S
	feature_labels = [] # List of feature labels
	for feature_name in feature_names:
		feature_label,_ = compute_feature_label(
			solver       = solver,
			feature_name = feature_name,
			S            = S,
		)
		feature_labels.append(feature_label)
	df_feature_contributions_S['feature_label'] = feature_labels
	# Make sure numbers are float (they can be 'mpmath')
	df_feature_contributions_S['p_value_contribution'] = df_feature_contributions_S['p_value_contribution'].astype(float)
	# Sort by p_value_contribution descending
	df_feature_contributions_S.sort_values(
		by        = 'p_value_contribution',
		ascending = False,
		inplace   = True,
	)
	# Take back the sorted feature labels
	feature_labels = df_feature_contributions_S['feature_label'].to_list()
	# Convert the p_value_contribution to percentages
	df_feature_contributions_S['p_value_contribution'] *= 100
	# Take the precision of the p-values
	if 'precision_p_values' in solver.monitoring_metadata.keys():
		precision_p_values = solver.monitoring_metadata['precision_p_values']
	else:
		precision_p_values = 'float64'
	if precision_p_values=='mpmath':
		import mpmath
	# Take the complexity of the rule
	complexity = len(S)
	# Compute the dpi
	dpi = 1446 / fig_width # so that 1446px (width of the banner) = 12 inches (width of the figure)
	# Create the banner as a separate figure
	if do_banner:
		# Create the banner
		banner = generate_insightsolver_banner(
			solver = solver,
			i      = i,
			loss   = loss,
		)
		# Define the height of the banner
		fig_height_banner_inches = banner.height / dpi
		# Create a figure for the banner
		fig_banner = plt.figure(figsize=(fig_width, fig_height_banner_inches), dpi=dpi)
		ax_banner = fig_banner.add_subplot(111)
		ax_banner.imshow(banner)
		ax_banner.axis("off")
		plt.show()
	# Create a bar plot as a separate figure
	fig_height_plot_inches = a * complexity + b
	fig_plot = plt.figure(figsize=(fig_width, fig_height_plot_inches), dpi=dpi)
	ax_plot = fig_plot.add_subplot(111)
	# Create the barplot
	ax = sns.barplot(
		ax      = ax_plot,
		data    = df_feature_contributions_S,
		x       = 'p_value_contribution',
		y       = 'feature_label',
		hue     = 'feature_label',
		palette = 'viridis',
		dodge   = False,
		legend  = False, # We do not show the legend
		zorder  = 3,     # So that the vertical lines are behind the horizontal bars
	)
	# Set the xlabel and the ylabel according to the language
	if language=='fr':
		ax.set_xlabel('Contribution de la variable (%)')
		ax.set_ylabel('Variable')
	elif language=='en':
		ax.set_xlabel('Feature Contribution (%)')
		ax.set_ylabel('Feature')
	# Set the xlim
	ax.set_xlim(0, 100)
	# Set the xticks
	ax.set_xticks(range(0, 101, 5))
	# Truncate the yticks labels
	locs, labels = plt.yticks() # # Get the current y-axis tick locations and labels
	truncated_labels = [truncate_label(label.get_text(), max_length=55) for label in labels] # Apply the truncation function to each label
	plt.yticks(locs, truncated_labels) # Set the new truncated labels and locations on the y-axis
	# Set the grid
	if do_grid:
		ax.grid(
			visible   = True,
			axis      = 'x',
			color     = 'gray',
			linestyle = '--',
			linewidth = 0.5,
			alpha     = 0.7,
			zorder    = 0,
		)
	# Set the title
	if do_title:
		if i==None:
			if language=='fr':
				title = "Contribution des variables"
			elif language=='en':
				title = "Contribution of the features"
			else:
				title = "Contribution of the features"
		else:
			if language=='fr':
				title  = f"Contribution de chaque variable à la puissance statistique de l'insight #{i+1}"
			elif language=='en':
				title  = f"Contribution of each variable to the statistical power of the insight #{i+1}"
			else:
				title  = f"Contribution of each variable to the statistical power of the insight #{i+1}"
			p_value    = rule_i['p_value']  # Take the p-value
			lift       = rule_i['lift']     # Take the lift
			coverage   = rule_i['coverage'] # Take the coverage
			if precision_p_values=='mpmath':
				formatted_p_value = mpmath.nstr(p_value, 2, strip_zeros=False)
				title += f"\np-value : {formatted_p_value}, lift : {lift:.2f},  coverage : {coverage* 100:.2f}%"
			else:
				title += f"\np-value : {p_value:.2e}, lift : {lift:.2f},  coverage : {coverage* 100:.2f}%"
		ax.set_title(title,size=12)

	# Add annotations
	if bar_annotations is not None:
		valid_bar_annotations = [
			'p_value_ratio',
			'p_value_contribution',
		]
		if bar_annotations not in valid_bar_annotations:
			raise Exception(f"ERROR: valid_bar_annotations='{valid_bar_annotations}' is not a valid value. It must be either None or in {valid_bar_annotations}.")
		
		for y, (x, value) in enumerate(zip(
				df_feature_contributions_S['p_value_contribution'],
				df_feature_contributions_S[bar_annotations])):
			bar_width        = ax.transData.transform((x/100,       0))[0] - ax.transData.transform((0,     0))[0] # Width in pixels of the bar from the origin to x
			annotation_width = ax.transData.transform((x/100 + 0.1, 0))[0] - ax.transData.transform((x/100, 0))[0] # Width in pixels of the annotation to show (approximation)
			if bar_width > annotation_width:
				# If the annotation is larger than the bar, we put the annotation to the right of the tip of the bar
				color = 'white'
				ha    = 'right'
			else:
				# If the annotation is shorter than the bar, we put the annotation to the left of the tip of the bar
				color = 'black'
				ha    = 'left'
			if bar_annotations=='p_value_ratio':
				if precision_p_values=='mpmath':
					s = ' '+mpmath.nstr(value, 2, strip_zeros=False)+' '
				else:
					s = f' {value:.2e} '
			elif bar_annotations=='p_value_contribution':
				s = f' {value:.2f} % '
			# Put the text
			ax.text(
				x        = x,
				y        = y,
				s        = s,
				color    = color,
				ha       = ha,
				va       = 'center',
				fontsize = 9,
			)

	# Generating the feature labels
	if any(len(feature_label) > 55 for feature_label in feature_labels):
		# If any feature label is too long, we add this details section
		# Add a text box underneath the plot using figtext
		if language=='fr':
			details_title = 'Détails'
		elif language=='en':
			details_title = 'Details'
		else:
			details_title = 'Details'
		# Create a new list to store the modified labels
		wrapped_feature_labels = []
		for feature_label in feature_labels:
			feature_label = '• ' + feature_label
			wrapped_label = wrap_text_with_word_boundary(
				text            = feature_label,
				max_line_length = 200,
			)
			wrapped_feature_labels.append(wrapped_label)		
		# Join the title with the prepared labels, each starting on a new line
		# (the LaTeX style string is to specify that only details_title is shown in bold)
		feature_label_text = "\n".join(
			[r"$\bf{" + f"{details_title}:" + "}$"] + wrapped_feature_labels
		) 
		# computing the number of rows the text contains
		n_rows = int(len(df_feature_contributions_S)) + int(feature_label_text.count('\n') + 1)
		fig_feature_label = plt.figure(figsize=(fig_width,  (0.05 * n_rows)))
		ax_feature_label = fig_feature_label.add_subplot(111)
		plt.figtext(
			x                 = 0.005,
			y                 = 0.005,
			s                 = feature_label_text, 
			wrap              = True,     # This helps for very long words that don't have commas
			fontsize          = 9, 
			verticalalignment = 'bottom', # Align text from the bottom edge of the figtext box
		)
		ax_feature_label.axis("off")
		plt.show()		

	# Tight layout
	plt.tight_layout()
	# Show the figure
	plt.show()

def show_all_feature_contributions(
	solver,
	a:float             = 0.5,   # Height per bar
	b:float             = 1,     # Height for the margin and other elements
	fig_width:float     = 12,    # Width of the figure
	language:str        = 'en',  # Language of the figure
	do_grid:bool        = True,  # If we want to show a grid
	do_title:bool       = False, # If we want to show a title which is automatically generated
	do_banner:bool      = True,  # If we want to show the banner
	bar_annotations:str = 'p_value_ratio', # Type of values to show at the end of the bars (can be 'p_value_ratio', 'p_value_contribution' or None)
)->None:
	"""
	This function generates a horizontal bar plot of the feature contributions for each rule found in a solver.
	
	Parameters
	----------
	solver: InsightSolver
		The fitted solver that contains the identified rules.
	a: float
		Height per bar.
	b: float
		Added height to the figure.
	fig_width: float
		Width of the figure
	language: str
		Language of the figure ('fr' or 'en').
	do_grid: bool
		If we want to show a vertical grid behind the horizontal bars.
	do_title: bool
		If we want to show a title.
	do_banner: bool
		If we want to show the banner.
	bar_annotations: str
		Type of values to show at the end of the bars (can be 'p_value_ratio', 'p_value_contribution' or None)
	"""
	# Take the list of rule index available in the solver
	range_i = solver.get_range_i()
	# Looping over the index
	for i in range_i:
		# Show the contributions of the rule i
		show_feature_contributions_of_i(
			solver          = solver,
			i               = i,
			a               = a,
			b               = b,
			fig_width       = fig_width,
			language        = language,
			do_grid         = do_grid,
			do_title        = do_title,
			do_banner       = do_banner,
			bar_annotations = bar_annotations,
		)

def show_feature_contributions_and_distributions_of_i(
	solver,
	i:int,
	language: str         = 'en',            # Language to use
	do_banner: bool       = True,            # If we want to show the banner
	loss: Optional[float] = None,            # Some loss number
	bar_annotations: str  = 'p_value_ratio', # Type of values to show at the end of the bars (can be 'p_value_ratio', 'p_value_contribution' or None)
)->None:
	"""
	This function returns a bar plot of the feature contributions and a distribution of the points in the rule i.
	
	Parameters
	----------
	solver: InsightSolver
		The fitted solver that contains the identified rules.
	i: int
		The index of the rule to show.
	language: str
		Language to use.
	do_banner: bool
		If we want to show the banner.
	loss: float
		If we want to show a loss.
	bar_annotations: str
		Type of values to show at the end of the bars (can be 'p_value_ratio', 'p_value_contribution' or None)
	"""
	# Generate the feature contributions figure
	show_feature_contributions_of_i(
		solver          = solver,
		i               = i,
		do_banner       = do_banner,
		loss            = loss,
		bar_annotations = bar_annotations,
		language        = language,
	)
	# Take the rule S at position i
	S = solver.i_to_S(i=i)
	# Generate the feature distributions of the rule S
	show_feature_distributions_of_S(
		solver   = solver,
		S        = S,
		language = language,
	)

def show_all_feature_contributions_and_distributions(
	solver,
	language: str        = 'en',            # Language to use
	do_banner: bool      = True,            # If we want to show the banner
	bar_annotations: str = 'p_value_ratio', # Type of values to show at the end of the bars (can be 'p_value_ratio', 'p_value_contribution' or None)
)->None:
	"""
	This function generates the feature contributions and feature distributions for all rules found in a fitted solver.
	
	Parameters
	----------
	solver: InsightSolver
		The fitted solver that contains the identified rules.
	language: str
		Language to use.
	do_banner: bool
		If we want to show the banner.
	bar_annotations: str
		Type of values to show at the end of the bars (can be 'p_value_ratio', 'p_value_contribution' or None)
	"""
	# Take the list of rule index available in the solver
	range_i = solver.get_range_i()
	# Looping over the index
	for i in range_i:
		# Show the contributions and distributions of the rule i
		show_feature_contributions_and_distributions_of_i(
			solver          = solver,
			i               = i,
			language        = language,
			do_banner       = do_banner,
			bar_annotations = bar_annotations,
		)

################################################################################
################################################################################
