"""
* `Organization`:  InsightSolver
* `Project Name`:  InsightSolver
* `Module Name`:   insightsolver
* `File Name`:     visualization.py
* `Author`:        Noé Aubin-Cadot
* `Email`:         noe.aubin-cadot@insightsolver.com
* `Last Updated`:  2025-04-25
* `First Created`: 2025-04-24

Description
-----------
This file contains some visualization functions, some of which are integrated as a method of the InsightSolver class.

Functions provided
------------------

- classify_variable_as_continuous_or_categorical
- compute_feature_label
- show_feature_distributions_of_S
- generate_insightsolver_banner
- show_feature_contributions_of_i
- show_all_feature_contributions
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

def classify_variable_as_continuous_or_categorical(
	s:pd.Series,
	unique_ratio_threshold:float = 0.1,
)->str:
	"""
	This function is meant to classify a series as continuous or categorical.
	This is used to decide which plot to do with it.

	Parameters
	----------
	s: pd.Series
		Series that needs to be classified.
	unique_ratio_threshold: float
		Threshold on the `unique_ratio`.

	Returns
	-------
	categorical_or_continuous: str
		The classification of the Series.
	"""
	# Take a look at the dtype of the Series
	if s.dtype=="object":
		categorical_or_continuous = "categorical"
	else:
		# Criterion 1: Verify if the Series contains decimals
		has_decimals = not all(s.dropna().apply(lambda x: float(x).is_integer()))
		
		# Criterion 2: Compute the number of unique values
		unique_values = s.nunique()
	
		# Criterion 3: Compare the unique number of values to the length of the series
		unique_ratio = unique_values / len(s.dropna())
		
		if has_decimals or (unique_ratio > unique_ratio_threshold):
			# If the Series contains decimals or a big number of modalities
			categorical_or_continuous = "continuous"
		else:
			# If the Series looks like an ordinal variable with few modalities
			categorical_or_continuous = "categorical"
	
	# Return the classification
	return categorical_or_continuous

def compute_feature_label(
	solver,            # The solver
	feature_name: str, # The name of the feature
	S: dict,           # The rule S
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

def show_feature_distributions_of_S(
	solver,
	S:dict,
	padding_y:int = 5,
	do_show_kde:bool = False,
	do_show_vertical_lines:bool = False,
)->None:
	"""
	This function generates bar plots of the distributions of the points in the specified rule S.

	Parameters
	----------
	solver : InsightSolver
		The solver object.
	S : dict
		The rule S that we wish to visualize.
	padding_y: int
		The padding used for the ylim.
	do_show_kde: bool
		Boolean to show the KDE of the continuous features.
	"""
	# Take the DataFrame that contains the data
	df = solver.df

	# Filter the data to the points that are in the rule S
	df_filtered = solver.S_to_df_filtered(S=S)

	# Take the size of a pixel instead of inches
	px = 1/plt.rcParams['figure.dpi']

	# Loop over the features of the rule
	for column_name in S.keys():
		# One bar plot will be created per feature name

		# Create the figure
		fig, ax = plt.subplots(figsize=(1446*px, 4))

		# On prend le btype de cette colonne
		if isinstance(S[column_name],list):
			column_btype = 'continuous'
		else:
			column_btype = 'multiclass'
		
		# On détermine si c'est une variable continu ou catégorielle
		if column_btype in ['binary','multiclass']:
			categorical_or_continuous = 'categorical'
		elif column_btype=='continuous':
			categorical_or_continuous = classify_variable_as_continuous_or_categorical(
				s = df[column_name],
			)
		else:
			raise Exception(f"ERROR: column_name='{column_name}' has a btype='{column_btype}' which is illegal.")
		
		# Look at the type of feature
		if categorical_or_continuous=='continuous':
			# If the feature is continuous

			# Calculate IQR for column using the Freedman-Diaconis rule
			Q1 = df[column_name].quantile(0.25)
			Q3 = df[column_name].quantile(0.75)
			IQR = Q3 - Q1
			step_bins = (2 * IQR) * ((len(df[column_name])) ** (-1 / 3))
			
			# Calculate the number of bins based on the range and the step size
			bin_count = int(np.ceil((df[column_name].max() - df[column_name].min()) / step_bins))
			
			# Limit the number of bins to 20 if it's greater than 20
			max_bins = 50
			bin_count = min(bin_count, max_bins)

			# Recalculate step_bins to fit the limited number of bins
			step_bins = (df[column_name].max() - df[column_name].min()) / bin_count

			# Create the bin edges
			bin_edges = np.arange(
				df[column_name].min(),
				df[column_name].max() + step_bins,
				step_bins
			)

			# First histplot for the distribution of the original variable
			sns.histplot(
				data  = df[column_name],
				kde   = do_show_kde,
				bins  = bin_edges,
				color = 'grey',
				alpha = 0.6,
			)
			
			# Take the maximum height of the bins
			max_count = max(patch.get_height() for patch in ax.patches)

			# Second plot for the distribution of the filtered variable by the rule
			sns.histplot(
				data  = df_filtered[column_name],
				bins  = bin_edges,
				color = 'green',
				alpha = 0.6,
			)
			
			# Rotate the bin edges
			plt.xticks(bin_edges, rotation=45)
			
			# Adjust the xlim
			plt.xlim(df[column_name].min() - step_bins, df[column_name].max()+step_bins)

			# Adjust the ylim
			plt.ylim(0, max_count + padding_y)

		else:

			# First countplot for the distribution of the original variable
			sns.countplot(
				data  = df,
				x     = column_name,
				color = 'grey',
				alpha = 0.6,
				label = "Unfiltered"
			)
			
			# Second plot for the distribution of the filtered variable by the rule
			sns.countplot(
				data  = df_filtered,
				x     = column_name,
				color = 'green',
				alpha = 0.6,
				label = "Filtered",
			)

			# Adjust the ylim
			most_frequent_count = df[column_name].value_counts().iloc[0]
			plt.ylim(0, most_frequent_count + padding_y)

		# Generate the feature label and the feature relationship
		feature_label,feature_relationship = compute_feature_label(
			solver       = solver,
			feature_name = column_name,
			S            = S,
		)

		if do_show_vertical_lines:
			# Take the boundaries specified by the continuous feature
			if isinstance(S[column_name],list):
				if isinstance(S[column_name][0],list):
					# If it's a continuous feature with NaNs
					[[rule_min,rule_max],rule_nan] = S[column_name]
				else:
					# If it's a continuous feature without NaNS
					rule_min,rule_max = S[column_name]
				# Add a vertical line
				if feature_relationship=='≥':
					# Add a vertical line at the lower boundary
					plt.axvline(rule_min, color='green', linestyle='--', label=column_name+' min')
				elif feature_relationship=='≤':
					# Add a vertical line at the upper boundary
					plt.axvline(rule_max, color='green', linestyle='--', label=column_name+' max')
				elif feature_relationship=='∈':
					# Add vertical lines at both boundaries
					plt.axvline(rule_min, color='green', linestyle='--', label=column_name+' min')
					plt.axvline(rule_max, color='green', linestyle='--', label=column_name+' max')

		# Generate the title
		if column_name in solver.columns_descr.keys():
			title = f"{solver.columns_descr[column_name]}\n{feature_label}"
		else:
			title = feature_label

		# Show the title
		plt.title(title)

		# Generate the xlabel
		plt.xlabel(column_name)

		# Hide the legend
		legend = plt.gca().get_legend()
		if legend is not None:
			legend.remove()

		# Tight layout
		plt.tight_layout()
		# Show the figure
		plt.show()

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
		p_text = str(p_value).split('e')[0][0:4] + 'e' + str(p_value).split('e')[1]
		p_position = (355, 20)
		draw.text(p_position, p_text, font_size=font_size, fill="black")
		# Draw the purity
		pure_text = str(round(purity*100, 2))+'%'
		pure_position = (595, 20)
		draw.text(pure_position, pure_text, font_size=font_size, fill="black")
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
		p_text = str(p_value).split('e')[0][0:4] + 'e' + str(p_value).split('e')[1]
		p_position = (320, 22)
		draw.text(p_position, p_text, font_size=font_size, fill="black")
		# Draw the purity
		pure_text = str(round(purity*100, 2))+'%'
		pure_position = (555, 22)
		draw.text(pure_position, pure_text, font_size=font_size, fill="black")
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

def show_feature_contributions_of_i(
	solver,
	i:int,                        # Index of the rule to show
	a:float              = 0.5,   # Height per bar
	b:float              = 1,     # Height for the margins and other elements
	fig_width:float      = 12,    # Width of the figure
	language:str         = 'en',  # Language of the figure
	do_grid:bool         = True,  # If we want to show a vertical grid
	do_title:bool        = False, # If we want a title automatically generated
	do_banner:bool       = True,  # If we want to show the banner
	loss:Optional[float] = None,  # If we want to show a loss
)->None:
	"""
	This function returns a horizontal bar plots of the feature constributions of a specified rule S.
	
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
		feature_label,feature_relationship = compute_feature_label(
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
	# Convert the p_value_contribution to percentages
	df_feature_contributions_S['p_value_contribution'] = df_feature_contributions_S['p_value_contribution']*100
	# Take the precision of the p-values
	if 'precision_p_values' in solver.monitoring_metadata.keys():
		precision_p_values = solver.monitoring_metadata['precision_p_values']
	else:
		precision_p_values = 'float64'
	if precision_p_values=='mpmath':
		import mpmath
	# Take the complexity of the rule
	complexity = len(S)
	# Compute the figure height
	fig_height = a*complexity+b
	# Create the figure
	if do_banner:
		# If we want to add a banner as a header of the figure
		# Create the banner
		banner = generate_insightsolver_banner(
			solver = solver,
			i      = i,
			loss   = loss,
		)
		# Add some height for the banner
		dpi = 1446 / fig_width  # so that 1446px (width of the banner) = 12 inches (width of the figure)
		banner_height_inches = banner.height / dpi
		fig_height += banner_height_inches
		# Create a figure
		fig = plt.figure(figsize=(fig_width,fig_height), dpi=dpi)
		# Calculate the height ratio for GridSpec
		ratio_banner = 2*banner_height_inches / fig_height  # Fraction of the figure's height for the banner (the 2x is for retina)
		ratio_plot = 1 - ratio_banner  # Fraction of the figure's height for the plot
		# Append the banner
		import matplotlib.gridspec as gridspec
		gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[ratio_banner, ratio_plot])
		ax_img = fig.add_subplot(gs[0])
		ax_img.imshow(banner)
		ax_img.axis('off')
		ax_plot = fig.add_subplot(gs[1])
	else:
		# If we do not want to add a banner as a header of the figure
		# Create a figure
		fig = plt.figure(figsize=(fig_width,fig_height))
		ax_plot = fig.add_subplot(111)
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
			if language=='fr':
				title  = f"Contribution de chaque variable à la puissance statistique de l'insight #{i+1}"
			elif language=='en':
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
	for y, (x, value) in enumerate(zip(df_feature_contributions_S['p_value_contribution'], df_feature_contributions_S['p_value_ratio'])):
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
		if precision_p_values=='mpmath':
			s = ' '+mpmath.nstr(value, 2, strip_zeros=False)+' '
		else:
			s = f' {value:.2e} '
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
	# Tight layout
	plt.tight_layout()
	# Show the figure
	plt.show()

def show_all_feature_contributions(
	solver,
	a:float         = 0.5,   # Height per bar
	b:float         = 1,     # Height for the margin and other elements
	fig_width:float = 12,    # Width of the figure
	language:str    = 'en',  # Language of the figure
	do_grid:bool    = True,  # If we want to show a grid
	do_title:bool   = False, # If we want to show a title which is automatically generated
	do_banner:bool  = True,  # If we want to show the banner
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
	"""
	# Take the list of rule index available in the solver
	range_i = solver.get_range_i()
	# Looping over the index
	for i in range_i:
		# Show the contributions of the rule i
		show_feature_contributions_of_i(
			solver    = solver,
			i         = i,
			a         = a,
			b         = b,
			fig_width = fig_width,
			language  = language,
			do_grid   = do_grid,
			do_title  = do_title,
			do_banner = do_banner,
		)

def show_feature_contributions_and_distributions_of_i(
	solver,
	i:int,
	do_banner:bool       = True, # If we want to show the banner
	loss:Optional[float] = None, # Some loss number
)->None:
	"""
	This function returns a bar plot of the feature contributions and a distribution of the points in the rule i.
	
	Parameters
	----------
	solver: InsightSolver
		The fitted solver that contains the identified rules.
	i: int
		The index of the rule to show.
	do_banner: bool
		If we want to show the banner.
	loss: float
		If we want to show a loss.
	"""
	# Generate the feature contributions figure
	show_feature_contributions_of_i(
		solver    = solver,
		i         = i,
		do_banner = do_banner,
		loss      = loss,
	)
	# Take the rule S at position i
	S = solver.i_to_S(i=i)
	# Generate the feature distributions of the rule S
	show_feature_distributions_of_S(
		solver = solver,
		S      = S,
	)

def show_all_feature_contributions_and_distributions(
	solver,
	do_banner:bool = True, # If we want to show the banner
)->None:
	"""
	This function generates the feature contributions and feature distributions for all rules found in a fitted solver.
	
	Parameters
	----------
	solver: InsightSolver
		The fitted solver that contains the identified rules.
	do_banner: bool
		If we want to show the banner.
	"""
	# Take the list of rule index available in the solver
	range_i = solver.get_range_i()
	# Looping over the index
	for i in range_i:
		# Show the contributions and distributions of the rule i
		show_feature_contributions_and_distributions_of_i(
			solver    = solver,
			i         = i,
			do_banner = do_banner,
		)

################################################################################
################################################################################
