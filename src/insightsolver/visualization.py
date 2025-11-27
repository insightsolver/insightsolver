"""
* `Organization`:  InsightSolver Solutions Inc.
* `Project Name`:  InsightSolver
* `Module Name`:   insightsolver
* `File Name`:     visualization.py
* `Authors`:       Noé Aubin-Cadot <noe.aubin-cadot@insightsolver.com>,
                   Arthur Albo <arthur.albo@insightsolver.com>

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
- svg_to_pil
- generate_insightsolver_img_banner
- generate_insightsolver_fig_banner
- generate_insightsolver_img_legend
- generate_insightsolver_fig_legend
- wrap_text_with_word_boundary
- show_feature_contributions_of_i
- show_all_feature_contributions
- show_feature_contributions_and_distributions_of_i
- show_all_feature_contributions_and_distributions
- show_mosaic_plot_of_i
- show_all_mosaic_plot_of_i

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

from typing import Optional, Union, Dict, Sequence, List

################################################################################
################################################################################
# Defining some global variables

# Width of the figures
FIG_WIDTH_IN = 12
# Dots per inch
DPI = 300
# InsightSolver blue
HEX_INSIGHTSOLVER = "#0530AD"

################################################################################
################################################################################
# Defining some visualization functions

def show_all_mutual_information(
    solver,
    n_samples:Optional[int] = 1000,
    n_cols:Optional[int]    = 20,
    kind: str               = 'barh',
    do_show: bool           = True,
    fig_width: float        = FIG_WIDTH_IN, # Width of the figure in inches
)->Optional["matplotlib.figure.Figure"]:
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
    do_show: bool
        Show the figure if True, return the figure if False.
    fig_width: float
        Width of the figure in inches

    Returns
    -------
    fig : matplotlib.figure.Figure, optional
        The matplotlib Figure object if `do_show=False`; otherwise None.
    """
    if not do_show:
        # Non interactive backend
        import matplotlib
        matplotlib.use("Agg")
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
    # Determine the colors of the bars
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "insightsolver_cmap",
        ["white", HEX_INSIGHTSOLVER]
    )
    # Normalize according to the values
    norm = mcolors.Normalize(
        vmin=0,
        vmax=s_mi.max()
    )
    # Color of each bar
    bar_colors = [cmap(norm(v)) for v in s_mi]
    # Generate the figure
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(
        figsize = (fig_width, 6),
    )
    s_mi.plot(
        kind      = kind,
        edgecolor = 'black',
        color     = bar_colors,
        linewidth = 0.8, # Thinner border
        ax        = ax,
    )
    plt.title('Mutual Information between the features and the target variable')
    plt.xlabel('Mutual Information')
    plt.ylabel('Feature')
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
    # Tight layout
    plt.tight_layout()
    # Return or show the figure
    if do_show:
        # Show the figure
        plt.show()
        return None
    else:
        # Return the figure
        return fig

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
    fig_width: float             = FIG_WIDTH_IN, # Width of the figure in inches
    verbose: bool                = False,
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
    fig_width: float
        Width of the figure in inches
    verbose: bool, default False
        Verbosity.
    """
    # Determine if a new figure needs to be created
    if ax is None:
        # Take the size of a pixel instead of inches
        if missing_value:
            fig, ax = plt.subplots(
                figsize = ((1/6)*fig_width, 4),
            )
        else:
            fig, ax = plt.subplots(
                figsize = (5/6*fig_width, 4),
            )
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

    if verbose:
        print("column_btype =",column_btype)
        print("categorical_or_continuous =",categorical_or_continuous)

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
        # Superpose a second blue bar for the number of missing values in the filtered data
        sns.countplot(
            x     = s_filtered_na,
            color = HEX_INSIGHTSOLVER,
            alpha = 1.0,
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
                color = HEX_INSIGHTSOLVER,
                alpha = 1.0,
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
            # Hangle the other modalities
            if feature_name in solver.other_modalities and len(solver.other_modalities[feature_name])>0:
                if verbose:
                    print("Other modalities found:",len(solver.other_modalities[feature_name]))
                other_mods = set(solver.other_modalities[feature_name])
                # Replace all modalities present in other_mods by "Other"
                s_unfiltered_dropna = s_unfiltered_dropna.apply(lambda x: "other" if x in other_mods else x)
                s_filtered_dropna   = s_filtered_dropna.apply(lambda x: "other" if x in other_mods else x)
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
                color = HEX_INSIGHTSOLVER,
                alpha = 1.0,
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
                    ax.axvline(rule_min, color=HEX_INSIGHTSOLVER, linestyle='--', label=feature_name+' min')
                elif feature_relationship=='≤':
                    # Add a vertical line at the upper boundary
                    ax.axvline(rule_max, color=HEX_INSIGHTSOLVER, linestyle='--', label=feature_name+' max')
                elif feature_relationship=='∈':
                    # Add vertical lines at both boundaries
                    ax.axvline(rule_min, color=HEX_INSIGHTSOLVER, linestyle='--', label=feature_name+' min')
                    ax.axvline(rule_max, color=HEX_INSIGHTSOLVER, linestyle='--', label=feature_name+' max')
                   
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
        grey_patch = mpatches.Patch(
            color = "grey",
            alpha = 0.6,
            label = "Hors de la règle" if language == 'fr' else "Outside the rule",
        )
        blue_patch = mpatches.Patch(
            color = HEX_INSIGHTSOLVER,
            alpha = 1.0,
            label = "Dans la règle" if language == 'fr' else "Inside the rule",
        )
        ax.legend(handles=[grey_patch, blue_patch])

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
    do_show: bool                = True,
    fig_width: float             = FIG_WIDTH_IN, # Width of the figure in inches
)->Optional[List["matplotlib.figure.Figure"]]:
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
    do_show: bool
        If True, displays the figures. If False, returns a list of matplotlib Figure objects.
    fig_width: float
        Width of the figure in inches

    Returns
    -------
    figs : list of matplotlib.figure.Figure or None
        List of figures if `do_show=False`. Otherwise None.
    """
    if not do_show:
        # Non interactive backend
        import matplotlib
        matplotlib.use("Agg")
    # Create a list of figures
    figs = []
    # Take the DataFrame that contains the data
    df = solver.df
    # Filter the data to the points that are in the rule S
    df_filtered = solver.S_to_df_filtered(S=S)
    # Loop over the features in the rule S
    for feature_name in S.keys():
        # One figure will be created per feature name
        # Look at if the data of the feature contains any missing value
        if solver.df[feature_name].isna().any():
            # If the feature contains any missing value
            # Create two graphs (one for the present values and one for the missing values)
            fig, axes = plt.subplots(
                figsize     = (fig_width, 4),
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
                figsize = (fig_width, 4),
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
        # Show or append to the list of figures
        if do_show:
            # Show the figure
            plt.show()
        else:
            # Append to the list of figures
            figs.append(fig)
    # Return the list of figures if the figures are not shown
    if not do_show:
        # Return the list of figures
        return figs
    else:
        # Return nothing
        return None

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

def svg_to_pil(
    svg_filename,
    assets_package = "insightsolver.assets",
    subfolder      = "google_fonts_icons",
    size           = (80,80),
):
    """
    Convert SVG to PIL Image with specified size.
    """
    from importlib.resources import files
    import cairosvg
    import io
    from PIL import Image
    svg_file = files(assets_package) / subfolder / svg_filename
    with svg_file.open("rb") as f:
        svg_bytes = f.read()
    png_bytes = cairosvg.svg2png(
        bytestring    = svg_bytes,
        output_width  = size[0],
        output_height = size[1],
    )
    return Image.open(io.BytesIO(png_bytes)).convert("RGBA")

def generate_insightsolver_img_banner(
    solver,
    i: int,
    loss: float                = None,
    fig_width: float           = 12,   # inches
    dpi: int                   = 200,
    icon_size: tuple[int, int] = (80, 80),
):
    """
    Generate a dynamic InsightSolver banner composed of SVG icons and text.

    Parameters
    ----------
    solver : InsightSolver
        The solver containing the rules.
    i : int
        Index of the rule to display.
    loss : float, optional
        Optional loss value to display.
    fig_width : float
        Width of the banner (in inches).
    dpi : int
        DPI resolution (pixels per inch).
    icon_size : tuple
        Icon size in pixels (width, height).

    Returns
    -------
    PIL.Image
        The generated banner.
    """
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    from importlib.resources import files

    # --- Extract rule data ---
    rule_i = solver.i_to_rule(i=i)
    p_value           = rule_i["p_value"]
    purity            = rule_i["mu_rule"]
    lift              = rule_i["lift"]
    coverage_relative = rule_i["coverage"]
    coverage_absolute = rule_i["m"]
    cohen_d           = rule_i["shuffling_scores"]["p_value"]["cohen_d"]

    precision_p_values = solver.monitoring_metadata.get("precision_p_values", "float64")
    if precision_p_values == "mpmath":
        import mpmath

    p_text = p_value_to_p_text(
        p_value=p_value,
        precision_p_values=precision_p_values,
    )

    # --- Icon mapping ---
    icons_map = {
        "insight_id_text":        "network_intelligence.svg",
        "p_text":                 "offline_bolt.svg",
        "purity_text":            "timelapse.svg",
        "lift_text":              "gondola_lift.svg",
        "coverage_relative_text": "zoom_out_map.svg",
        "coverage_absolute_text": "select_all.svg",
        "cohen_d_text":           "shuffle.svg",
        "loss_text":              "sell.svg",
    }

    # --- Values to display ---
    values_all = [
        ("insight_id_text",        f"Insight #{i+1}"),
        ("p_text",                 p_text),
        ("purity_text",            f"{round(purity * 100, 2)} %"),
        ("lift_text",              f"{round(lift, 2)}"),
        ("coverage_relative_text", f"{round(coverage_relative * 100, 2)} %"),
        ("coverage_absolute_text", str(coverage_absolute)),
        ("cohen_d_text",           f"{cohen_d:.2f}"),
    ]
    if loss is not None:
        values_all.append(("loss_text", str(loss)))
        font_ratio = 0.38
    else:
        font_ratio =  0.45

    # --- Banner dimensions ---
    banner_width  = int(fig_width * dpi)
    banner_height = 120
    img_banner    = Image.new("RGBA", (banner_width, banner_height), "white")
    draw          = ImageDraw.Draw(img_banner)

    # --- Load Roboto font ---
    font_size           = int(icon_size[1] * font_ratio)
    roboto_regular_path = files("insightsolver.assets") / "google_fonts_icons" / "Roboto-Regular.ttf"
    roboto_bold_path    = files("insightsolver.assets") / "google_fonts_icons" / "Roboto-Bold.ttf"
    font_regular        = ImageFont.truetype(str(roboto_regular_path), size=font_size)
    font_bold           = ImageFont.truetype(str(roboto_bold_path), size=font_size)

    # --- Fixed horizontal layout ---
    n_blocks = len(values_all)
    margin = 20     # Margin around the cells, in pixels
    gap = margin*2  # Horizontal gap between cells, in pixels
    total_gap = gap * (n_blocks - 1)
    usable_width = banner_width - 2 * margin - total_gap
    space_per_block = usable_width / n_blocks
    x_positions = [int(margin + i * (space_per_block + gap)) for i in range(n_blocks)]

    # --- Vertical icon placement ---
    y_icon = (banner_height - icon_size[1]) // 2

    # --- Shadow parameters ---
    shadow_offset = 2            # Slight offset in x,y
    shadow_radius = 4            # Blur radius
    shadow_color = (0, 0, 0, 60) # Semi transparent black for the shadow

    # Colorisation du cohen_d
    if cohen_d>2:
        cohen_d_color = "#d4edda" # Light greed background
    elif cohen_d>0:
        cohen_d_color = "#fff3cd" # Light yellow background
    else:
        cohen_d_color = "#f8d7da" # Light red background

    # --- Draw icons and text ---
    for (key, text), x in zip(values_all, x_positions):

        # Define the bounding box of the block
        block_x0 = x
        block_x1 = int(x + space_per_block)
        pad      = 10  # Internal margin
        block_y0 = pad
        block_y1 = banner_height - pad

        # --- Draw shadow using Gaussian blur ---
        shadow = Image.new("RGBA", img_banner.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)

        # shadow rectangle coords (slightly offset)
        shadow_rect = [
            (block_x0 + shadow_offset, block_y0 + shadow_offset),
            (block_x1 + shadow_offset, block_y1 + shadow_offset),
        ]

        shadow_draw.rounded_rectangle(
            shadow_rect,
            radius = 12,
            fill   = shadow_color,
        )

        # Apply blur
        shadow = shadow.filter(
            ImageFilter.GaussianBlur(
                radius = shadow_radius,
            ),
        )

        # Paste shadow onto img_banner
        img_banner.alpha_composite(shadow)

        # --- Draw grey outline rectangle ---
        fill_color = cohen_d_color if key == "cohen_d_text" else (242, 242, 242)
        draw.rounded_rectangle(
            [(block_x0 + 2, block_y0), (block_x1 - 2, block_y1)],
            outline = (213, 213, 213),
            width   = 2,
            radius  = 12,
            fill    = fill_color,
        )

        # Draw icon
        icon = svg_to_pil(
            svg_filename = icons_map[key],
            size         = icon_size,
        )
        img_banner.paste(icon, (x + pad, y_icon), mask=icon)

        font = font_bold if key == "insight_id_text" else font_regular

        # --- Horizontal centering for text within the block ---
        block_text_start_x = x + icon_size[0]
        block_text_width   = block_x1 - block_text_start_x - pad
        text_width         = draw.textlength(text, font=font)
        x_text             = block_text_start_x + (block_text_width - text_width) // 2

        # --- Vertical centering using typographic metrics ---
        ascent, descent = font.getmetrics()
        text_height     = ascent + descent
        icon_center_y   = y_icon + icon_size[1] // 2
        y_text          = icon_center_y - text_height // 2

        # Draw text
        draw.text(
            (x_text, y_text),
            text,
            fill = "black",
            font = font,
        )
    
    return img_banner

def generate_insightsolver_fig_banner(
    solver,
    i: int,
    loss: float                = None,
    fig_width: float           = 12,   # inches
    dpi: int                   = 200,
    icon_size: tuple[int, int] = (80, 80),
    do_show: bool              = False,
):
    # Create the banner image
    img_banner = generate_insightsolver_img_banner(
        solver    = solver,
        i         = i,
        loss      = loss,
        fig_width = fig_width,
        dpi       = dpi,
        icon_size = icon_size,
    )
    # Size in pixels of the banner image
    height_px = img_banner.height
    width_px  = img_banner.width
    # Take the ratio height/width
    ratio = height_px / width_px
    # Height of the banner in inches
    fig_height = fig_width * ratio
    # Create a figure for the banner
    fig_banner = plt.figure(
        figsize = (fig_width, fig_height),
        dpi     = dpi,
    )
    ax = fig_banner.add_subplot(111)
    ax.imshow(img_banner)
    ax.axis("off")
    if do_show:
        plt.show()
    # Return the figure of the banner
    return fig_banner

def generate_insightsolver_img_legend(
    do_show_loss: bool         = True,
    fig_width: float           = 12,   # inches
    dpi: int                   = 200,
    icon_size: tuple[int, int] = (80, 80),
    language: str              = 'en',
    verbose: bool              = False,
):
    """
    Generate a legend that explains what the icons of the legend represent.

    Parameters
    ----------
    do_show_loss : bool
        If we want to describe the symbol for the loss in the legend.
    fig_width : float
        Width of the legend (in inches).
    dpi : int
        DPI resolution (pixels per inch).
    icon_size : tuple
        Icon size in pixels (width, height).
    verbose: bool
        Verbosity.

    Returns
    -------
    PIL.Image
        The generated legend.
    """

    # Create a DataFrame of labels, icons and texts
    data = {
        "insight_id": (
            "network_intelligence.svg",
            "Number of the insight, starting from 1.",
            "Numéro de l'insight, en commençant par 1.",
        ),
        "p_value": (
            "offline_bolt.svg",
            "p-value.",
            "p-value.",
        ),
        "purity": (
            "timelapse.svg",
            "Purity.",
            "Pureté.",
        ),
        "lift": (
            "gondola_lift.svg",
            "Lift.",
            "Lift.",
        ),
        "coverage_relative": (
            "zoom_out_map.svg",
            "Relative coverage.",
            "Couverture relative.",
        ),
        "coverage_absolute": (
            "select_all.svg",
            "Absolute coverage.",
            "Couverture absolue.",
        ),
        "cohen_d": (
            "shuffle.svg",
            "Shuffling score (Cohen's d).",
            "Score de permutations (Cohen's d).",
        ),
        "loss": (
            "sell.svg",
            "Loss",
            "Coût",
        ),
    }
    columns = [
        'svg_filename',
        'en',
        'fr',
    ]
    df = pd.DataFrame.from_dict(
        data    = data,
        orient  = 'index',
        columns = columns,
    )
    df.index.name = 'label'
    # Handle the loss
    if not do_show_loss:
        df.drop(index=['loss'],inplace=True)
    # Import libraries
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    from importlib.resources import files
    # Number of blocks, one for "Legend:" and one per row of df
    n_blocks = 1 + len(df)
    if verbose:
        print("n_blocks = ",n_blocks)
    # Legend height per block
    height_per_block = 120 # 120 pixels per block
    if verbose:
        print("height_per_block =",height_per_block)
    # Legend dimensions
    legend_width  = int(fig_width * dpi)
    legend_height = height_per_block*n_blocks
    if verbose:
        print("legend_width =",legend_width)
        print("legend_height =",legend_height)
    # Create the image
    img_legend = Image.new("RGBA", (legend_width, legend_height), "white")
    draw       = ImageDraw.Draw(img_legend)
    # Load font
    font_ratio = 0.5
    font_size  = int(icon_size[1] * font_ratio)
    font_path_regular  = files("insightsolver.assets") / "google_fonts_icons" / "Roboto-Regular.ttf"
    font_path_bold     = files("insightsolver.assets") / "google_fonts_icons" / "Roboto-Bold.ttf"
    font_regular       = ImageFont.truetype(str(font_path_regular), size=font_size)
    font_bold          = ImageFont.truetype(str(font_path_bold), size=font_size)
    if verbose:
        print("icon_size =",icon_size)
        print("font_size =",font_size)
    # Compute text height
    ascent, descent = font_regular.getmetrics()
    text_height     = ascent + descent
    if verbose:
        print("ascent =",ascent)
        print("descent =",descent)
        print("text_height =",text_height)    
    # Vertical layout
    margin          = 20          # Vertical margin around the blocks, in pixels
    gap             = margin*2    # Vertical gap between blocks, in pixels
    total_gap       = gap * (n_blocks - 1) # Total vertical gap
    usable_height   = legend_height - 2 * margin - total_gap # Usable vertical height
    usable_height_per_block = usable_height // n_blocks # Usable vertical height per block
    y_positions     = [int(margin + i * (usable_height_per_block + gap)) for i in range(n_blocks)] # Vertical positions of the blocks
    if verbose:
        print("margin =",margin)
        print("gap =",gap)
        print("total_gap =",total_gap)
        print("usable_height =",usable_height)
        print("usable_height_per_block =",usable_height_per_block)
        print("y_positions =",y_positions)
    # Add "Legend:" in bold in the first block
    x_text_legend = margin
    y_text_legend = y_positions[0] + text_height // 2
    if language=='fr':
        text_legend = "Légende :"
    else:
        text_legend = "Legend:"
    draw.text(
        xy   = (
            x_text_legend,
            y_text_legend,
        ),
        text = text_legend,
        fill = "black",
        font = font_bold,
    )
    if verbose:
        print("x_text_legend =",x_text_legend)
        print("y_text_legend =",y_text_legend)
    # Internal padding of the block
    padding = 0
    if verbose:
        print("padding =",padding)
    
    # Draw icons and text
    for i,label in enumerate(df.index):
        if verbose:
            print(f"\n{i} : {label}")
        # Take the icon and the text
        svg_filename = df.loc[label,'svg_filename']
        text = df.loc[label,language]
        # Position of the block
        x_block = margin
        y_block = y_positions[i+1]
        if verbose:
            print("x_block =",x_block)
            print("y_block =",y_block)
        # Horizontal icon placement
        x_icon = x_block + padding
        if verbose:
            print("x_icon =",x_icon)
        # Vertical icon placement
        y_icon = (int(y_block + usable_height_per_block) - icon_size[1])
        if verbose:
            print("y_icon =",y_icon)
        # Draw icon
        icon = svg_to_pil(
            svg_filename = svg_filename,
            size         = icon_size,
        )
        img_legend.paste(
            icon,
            (x_icon, y_icon),
            mask = icon,
        )
        # Position of the text within the block
        x_text = x_block + icon_size[0] + 20
        y_text  = y_block + text_height // 2 - 10
        if verbose:
            print("x_text =",x_text)
            print("y_text =",y_text)
        # Draw text
        draw.text(
            xy   = (
                x_text,
                y_text,
            ),
            text = text,
            fill = "black",
            font = font_regular,
        )
    # Return the image
    return img_legend

def generate_insightsolver_fig_legend(
    do_show_loss: bool         = False,
    fig_width: float           = 12,   # inches
    dpi: int                   = 200,
    icon_size: tuple[int, int] = (80, 80),
    language: str              = 'en',
    verbose: bool              = False,
    do_show: bool              = False,
):
    # Create the legend image
    img_legend = generate_insightsolver_img_legend(
        do_show_loss = do_show_loss,
        fig_width    = fig_width,
        dpi          = dpi,
        icon_size    = icon_size,
        language     = language,
        verbose      = verbose,
    )
    # Size in pixels of the legend image
    legend_height_px = img_legend.height
    legend_width_px  = img_legend.width
    # Take the ratio height/width
    legend_ratio = legend_height_px / legend_width_px
    # Height of the legend in inches
    fig_height = fig_width * legend_ratio
    # Create a figure for the legend
    fig_legend = plt.figure(
        figsize = (fig_width, fig_height),
        dpi     = dpi,
    )
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.imshow(img_legend)
    ax_legend.axis("off")
    if do_show:
        plt.show()
    # Return the figure of the legend
    return fig_legend

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
    fig_width: float      = FIG_WIDTH_IN, # Width of the figure in inches
    language: str         = 'en',  # Language of the figure
    do_grid: bool         = True,  # If we want to show a vertical grid
    do_title: bool        = False, # If we want a title automatically generated
    do_banner: bool       = True,  # If we want to show the banner
    bar_annotations: str  = 'p_value_ratio', # Type of values to show at the end of the bars (can be 'p_value_ratio', 'p_value_contribution' or None)
    loss: Optional[float] = None,  # If we want to show a loss
    do_show: bool         = True,  # If we want to show the figure or return it
)->Optional[List["matplotlib.figure.Figure"]]:
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
        Width of the figure in inches
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
    do_show: bool
        If we want to show the figure or return it

    Returns
    -------
    figs : List of matplotlib.figure.Figure or None
        List of Figure if `do_show=False`. Otherwise None.
    """
    figs = []
    if not do_show:
        # Non interactive backend
        import matplotlib
        matplotlib.use("Agg")
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
    # Take the dpi
    dpi = DPI
    # Create the banner as a separate figure
    if do_banner:
        fig_banner = generate_insightsolver_fig_banner(
            solver    = solver,
            i         = i,
            loss      = loss,
            fig_width = fig_width,
            dpi       = dpi,
            do_show   = do_show,
        )
        figs.append(fig_banner)
    # Create a bar plot as a separate figure
    fig_height_plot_inches = a * complexity + b
    fig_plot = plt.figure(
        figsize = (fig_width, fig_height_plot_inches),
        dpi     = dpi,
    )
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

    # Change the colors of the bars and their contours
    import matplotlib.colors as mcolors
    vals = df_feature_contributions_S['p_value_contribution'].values
    normalized_values = vals / vals.max()  # 0 → white, max → blue
    cmap = mcolors.LinearSegmentedColormap.from_list("white_to_blue", ["#FFFFFF", HEX_INSIGHTSOLVER])
    bar_colors = [cmap(v) for v in normalized_values]
    for bar, bar_color in zip(ax.patches, bar_colors):
        bar.set_facecolor(bar_color)
        bar.set_edgecolor('black')
        bar.set_linewidth(0.8)
    
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

    # Define a function that maps a RGB color to a level of luminosity
    def relative_luminance(rgb):
        # Ignore alpha if present
        r, g, b = rgb[:3]
        # Return the luminance (sRGB norm)
        return 0.2126*r + 0.7152*g + 0.0722*b
    
    # Add annotations
    if bar_annotations is not None:
        valid_bar_annotations = [
            'p_value_ratio',
            'p_value_contribution',
        ]
        if bar_annotations not in valid_bar_annotations:
            raise Exception(f"ERROR: valid_bar_annotations='{valid_bar_annotations}' is not a valid value. It must be either None or in {valid_bar_annotations}.")
        
        for y, (x, value, bar_color) in enumerate(zip(
                df_feature_contributions_S['p_value_contribution'],
                df_feature_contributions_S[bar_annotations],
                bar_colors, # Colors of the bars
        )):
            bar_width        = ax.transData.transform((x/100,       0))[0] - ax.transData.transform((0,     0))[0] # Width in pixels of the bar from the origin to x
            annotation_width = ax.transData.transform((x/100 + 0.1, 0))[0] - ax.transData.transform((x/100, 0))[0] # Width in pixels of the annotation to show (approximation)
            if bar_width > annotation_width:
                # If the annotation is larger than the bar, we put the annotation to the right of the tip of the bar
                ha    = 'right'
                # Handle the color of the annotation
                lum = relative_luminance(bar_color)
                if lum<0.5:
                    color = 'white'
                else:
                    color = 'black'
            else:
                # If the annotation is shorter than the bar, we put the annotation to the left of the tip of the bar
                ha    = 'left'
                # Handle the color of the annotation
                color = 'black'
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

    figs.append(fig_plot)

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
        if do_show:
            plt.show()
        else:
            figs.append(fig_feature_label)

    # Tight layout
    plt.tight_layout()
    # Show of return the figure
    if do_show:
        # Show the figure
        plt.show()
        return None
    else:
        # Return the figure
        return figs

def show_all_feature_contributions(
    solver,
    a:float             = 0.5,   # Height per bar
    b:float             = 1,     # Height for the margin and other elements
    fig_width:float     = FIG_WIDTH_IN, # Width of the figure in inches
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
    do_legend: bool      = True,            # If we want to show the legend
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
    do_legend: bool
        If we want to show the legend.
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
    # If we want to show the legend
    if do_legend:
        generate_insightsolver_fig_legend(
            language = language,
            do_show  = True,
        )

def show_mosaic_plot_of_i(
    solver,
    i: int,
    feature_name: Optional[str] = None,
    ax                          = None,
):
    # Determine if a new figure needs to be created
    do_show = False
    if ax == None:
        fig, ax = plt.subplots(figsize=(4, 4))
        do_show = True
    # Take the rule S at position i
    S = solver.i_to_S(i=i)
    # Take the target_name
    target_name = solver.target_name
    # Take some global statistics
    M   = solver.M
    M0  = solver.M0
    M1  = solver.M1
    # Take some rule statistics
    if feature_name is None:
        # Take the rule at position i
        rule_i = solver.i_to_rule(i=i)
        # Take some statistics
        m   = rule_i['m']
        m1  = rule_i['m1']
    else:
        # Create a subrule
        S = {feature_name:S[feature_name]}
        # Take the index of the points in the subrule
        index_points_in_rule = solver.S_to_index_points_in_rule(S=S)
        # Compute the statistics of the subrule
        s_y = solver.convert_target_to_binary().loc[index_points_in_rule]
        m = len(s_y)
        m1 = s_y.sum()
    m0 = m-m1
    mc  = M-m
    m0c = M0-m0
    m1c = M1-m1
    # Coverage
    coverage_rule = m/M if M>0 else 0
    coverage_comp = 1-coverage_rule
    # Purities
    mu1_pop  = M1/M if M>0 else 0
    mu0_rule = m0/m if m>0 else 0
    mu1_rule = m1/m if m>0 else 0
    mu0_comp = m0c/mc if mc>0 else 0
    mu1_comp = m1c/mc if mc>0 else 0
    # Create a pandas Series with a MultiIndex
    data = pd.Series(
        data  = [m1,m0,m1c,m0c],
        index = pd.MultiIndex.from_product([['in', 'out'], ['1', '0']]),
    )
    # Define a coloring function
    def color_func(key):
        """
        Returns a dictionary of colors based on the key tuple.
        """
        # Define colors for each combination
        if key == ('in', '1'):
            # Inside the rule, class 1
            return {'color': HEX_INSIGHTSOLVER}
        elif key == ('in', '0'):
            # Inside the rule, class 0
            return {'color': 'grey'}
        elif key == ('out', '1'):
            # Outsite the rule, class 1
            return {'color': HEX_INSIGHTSOLVER, 'alpha':0.5}
        elif key == ('out', '0'):
            # Outside the rule, class 0
            return {'color': 'grey', 'alpha':0.5}
        else:
            # Default color
            return {}
    # Custom labelizer to show nothing
    def empty_labelizer(key):
        return ""
    # Create the plot with your existing parameters
    from statsmodels.graphics.mosaicplot import mosaic
    mosaic(
        data       = data,
        ax         = ax,
        properties = color_func,
        labelizer  = empty_labelizer,
        gap        = 0.02,
        title      = "",
        statistic  = False,
        axes_label = True,
    )
    # Edit the mosaic plot
    import matplotlib.ticker as mticker
    # Edit the xlabel and ylabel
    ax.set_xlabel("Coverage (%)", fontsize=12)
    ax.set_ylabel("Purity (%)", fontsize=12)
    # Edit the xlim and ylim
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_xlim())
    # Edit the xticks and yticks
    ax.set_xticks(np.linspace(0, 1, 6)) # Set ticks at 0%, 20%, 40%, 60%, 80%, 100%
    ax.set_yticks(np.linspace(0, 1, 6)) # Set ticks at 0%, 20%, 40%, 60%, 80%, 100%
    # Format the ticks as percentages
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    # Manually set x-axis tick locations and labels
    ticker_feature_location_in = float(coverage_rule / 2)
    ticker_feature_location_out = float((1-coverage_rule) / 2) + (ticker_feature_location_in*2)
    x_tick_locations = [ticker_feature_location_in, ticker_feature_location_out]
    x_tick_labels = ['in', 'out']
    ax_top = ax.twiny()
    ax_top.set_xticks(x_tick_locations)
    ax_top.set_xticklabels(x_tick_labels)
    # Title of the figure
    if not feature_name:
        title = f'Insight #{i + 1}'
    else:
        # Take the feature label
        feature_label, _ = compute_feature_label(
            solver       = solver,       # The solver
            feature_name = feature_name, # The name of the feature
            S            = S,            # The rule S
        )
        # Truncate the feature label
        feature_label = truncate_label(
            label      = feature_label,
            max_length = 50,
        )
        # The title is the feature_label
        title = feature_label
    ax_top.set_xlabel(
        title,
        fontsize   = 12,
        fontweight = 'bold',
    )
    #alpha = 0
    alpha = 0.2
    #alpha = 0.35
    backgroundcolor = (1,1,1,alpha)
    fontsize = 9
    if mu1_rule != 0:
        ax.text(
            coverage_rule/2,
            mu1_rule+0.02,
            f"{mu1_rule:.1%}",
            fontsize = fontsize,
            ha       = 'center',
            va       = 'center',
            backgroundcolor = backgroundcolor,
        )
    if mu1_comp != 0:
        ax.text(
            coverage_rule+(coverage_comp/2),
            mu1_comp+0.02,
            f"{mu1_comp:.1%}",
            fontsize = fontsize,
            ha       = 'center',
            va       = 'center',
            backgroundcolor = backgroundcolor,
        )
    # Add a dashed line for the purity of the population
    ax.axhline(
        y         = mu1_pop,
        linewidth = 1,
        color     = 'r',
        linestyle = ":",
    )
    # Add a text over the dashed line
    ax.text(
        x        = 0.5,
        y        = mu1_pop+0.01,
        s        = f"{mu1_pop:.1%}",
        fontsize = fontsize,
        color    = 'r',
        ha       = "center",
        backgroundcolor = backgroundcolor,
    )
    if do_show:
        # Only tight_layout if showing an individual plot
        plt.tight_layout()
        plt.show()

def show_all_mosaic_plot_of_i(
    solver,
    i: int,
    ncols: int = 3,
):
    """
    This function shows the mosaic plot of the rule at position i.
    One figure is generated for the whole rule and then one figure is generated per feature in the rule.

    Parameters
    ----------
    solver: InsightSolver
        The solver.
    i: int
        Index of the rule.
    ncols: int
        Number of figures per row (should be 1, 2, 3, 4).
    """
    # Make sure the solver if fitted
    if not solver._is_fitted:
        return None
    # Make sure i is valid
    if i not in range(len(solver)):
        return None
    # Make sure ncols is valid
    if ncols not in [1,2,3,4]:
        return None
    # Take the feature names in the rule, sorted by contribution, descending
    feature_names = solver.i_to_feature_names(
        i       = i,
        do_sort = True,
    )
    # Number of features
    n_features = len(feature_names)
    # Compute the number of rows (one for then target then rows for the features)
    nrows = 1 + (1+ (n_features-1)//ncols)
    # Determine the size of the figure
    fig_width  = 12 # 12 inch wide, 3 figure per row, 4 inch per figure
    fig_height = 4 * nrows # Each figure is a square of size (4 inch x 4 inch)
    figsize = (
        fig_width,
        fig_height,
    )
    # Create the subplot grid
    fig, axes = plt.subplots(
        nrows   = nrows,
        ncols   = ncols,
        figsize = figsize,
    )
    # Flatten the 2D array of axes into a 1D array
    axes = axes.flatten()
    # Add the mosaic plot of the whole rule on the first row
    show_mosaic_plot_of_i(
        solver        = solver,
        i             = i,
        feature_name  = None,
        ax            = axes[0],
    )
    # Add the mosaic plot of the feature rules in the other rows
    for k, feature_name in enumerate(feature_names):
        show_mosaic_plot_of_i(
            solver        = solver,
            i             = i,
            feature_name  = feature_name,
            ax            = axes[ncols + k],
        )
    # Hide any remaining, unused subplots
    for k in range(1, ncols):
        axes[k].set_visible(False)    
    total_plots = n_features + 1
    for k in range(total_plots + ncols-1, len(axes)):
        axes[k].set_visible(False)
    # Adjusts subplot params for a tight layout
    plt.tight_layout()
    # Display the figure
    plt.show()

def show_mosaic_plot_pop_vs_rule_of_i(
    solver,
    i: int,
    ax = None,
    do_show_comp:bool = True,
):
    # Take the range of rules
    range_i = solver.get_range_i()
    # Make sure i is valid
    if i not in range_i:
        raise Exception(f"ERROR (show_mosaic_plot_pop_vs_rule_of_i): i={i} is not valid.")

    # Determine if a new figure needs to be created
    do_show = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
        do_show = True
        
    # Population statistics
    M  = solver.M
    M1 = solver.M1
    M0 = solver.M0
    # Rule statistics
    rule_i = solver.i_to_rule(i=i)
    m  = rule_i['m']
    m1 = rule_i['m1']
    m0 = rule_i['m0']
    # Complement statistics
    mc  = M-m
    m1c = M1-m1
    m0c = M0-m0

    # Computing the various purities
    mu1_pop  = M1/M if M>0 else 0
    mu0_pop  = M0/M if M>0 else 0
    mu1_rule = m1/m if m>0 else 0
    mu0_rule = m0/m if m>0 else 0
    mu1_comp = m1c/mc if mc>0 else 0
    mu0_comp = m0c/mc if mc>0 else 0

    # Create a pandas Series with a MultiIndex
    if do_show_comp:
        data = pd.Series(
            data = [mu1_pop,mu0_pop,mu1_rule,mu0_rule,mu1_comp,mu0_comp],
            index = pd.MultiIndex.from_product([['Population', 'Rule', 'Complement'], ['1', '0']]),
        )
    else:
        data = pd.Series(
            data = [mu1_pop,mu0_pop,mu1_rule,mu0_rule],
            index = pd.MultiIndex.from_product([['Population', 'Rule'], ['1', '0']]),
        ) 

    def color_func(key):
        """
        Returns a dictionary of colors based on the key tuple.
        Key format: (in rule or not, class 0 or 1)
        """
        # Define colors for each combination
        if key == ('Population', '1'):
            return {'color': 'grey'}
        elif key == ('Population', '0'):
            return {'color': 'grey', 'alpha':0.5}
        elif key == ('Rule', '1'):
            return {'color': HEX_INSIGHTSOLVER}
        elif key == ('Rule', '0'):
            return {'color': HEX_INSIGHTSOLVER, 'alpha':0.5}
        elif key == ('Complement', '1'):
            return {'color': 'grey'}
        elif key == ('Complement', '0'):
            return {'color': 'grey', 'alpha':0.5}
        else:
            return {}

    # Custom labelizer to show nothing
    def empty_labelizer(key):
        return ""

    # Create the plot with your existing parameters
    from statsmodels.graphics.mosaicplot import mosaic
    mosaic(
        data       = data,
        ax         = ax,
        properties = color_func,
        labelizer  = empty_labelizer,
        gap        = 0.03,
        title      = "",
        statistic  = False,
        axes_label = True,
    )

    # Add title
    ax_top = ax.twiny()
    ax_top.set_xlabel(
        xlabel     = f"Insight #{i+1}",
        fontsize   = 12,
        fontweight = 'bold',
    )
    # Add a bit of distance between the title and the figure
    ax_top.xaxis.labelpad = 10

    # Edit the xlabel and ylabel
    ax.set_xlabel("Subset", fontsize=12)
    ax.set_ylabel("Class", fontsize=12)
    
    # Mask the top tickers
    fig = ax.get_figure()
    for a in fig.axes:
        if a.xaxis.get_ticks_position() == "top":
            if not any(lbl in ["Population", "Rule", "Complement"] for lbl in a.get_xticklabels()):
                a.set_xticks([])
                a.set_xticklabels([])
                a.tick_params(top=False)

    alpha = 0.2
    backgroundcolor = (1,1,1,alpha)
    fontsize = 9
    if mu1_pop != 0:
        ax.text(
            x        = 0.166 if do_show_comp else 0.25,
            y        = mu1_pop/2,
            s        = f"{mu1_pop:.1%}",
            fontsize = fontsize,
            ha       = 'center',
            va       = 'center',
            backgroundcolor = backgroundcolor,
        )
    if mu0_pop != 0:
        ax.text(
            x        = 0.166 if do_show_comp else 0.25,
            y        = mu1_pop+(mu0_pop/2),
            s        = f"{mu0_pop:.1%}",
            fontsize = fontsize,
            ha       = 'center',
            va       = 'center',
            backgroundcolor = backgroundcolor,
        )
    if mu1_rule != 0:
        ax.text(
            x        = 0.5 if do_show_comp else 0.75,
            y        = mu1_rule/2,
            s        = f"{mu1_rule:.1%}",
            fontsize = fontsize,
            ha       = 'center',
            va       = 'center',
            backgroundcolor = backgroundcolor,
        )
    if mu0_rule != 0:
        ax.text(
            x        = 0.5 if do_show_comp else 0.75,
            y        = mu1_rule+(mu0_rule/2),
            s        = f"{mu0_rule:.1%}",
            fontsize = fontsize,
            ha       = 'center',
            va       = 'center',
            backgroundcolor = backgroundcolor,
        )
    if do_show_comp:
        if mu1_comp != 0:
            ax.text(
                x        = 0.833,
                y        = mu1_comp/2,
                s        = f"{mu1_comp:.1%}",
                fontsize = fontsize,
                ha       = 'center',
                va       = 'center',
                backgroundcolor = backgroundcolor,
            )
        if mu0_comp != 0:
            ax.text(
                x        = 0.833,
                y        = mu1_comp+(mu0_comp/2),
                s        = f"{mu0_comp:.1%}",
                fontsize = fontsize,
                ha       = 'center',
                va       = 'center',
                backgroundcolor = backgroundcolor,
            )
    
    if do_show:
        # Only tight_layout if showing an individual plot
        plt.tight_layout()
        plt.show()

def show_mosaic_plot_pop_vs_rule(
    solver,
    do_show_comp: bool = True,
    ncols:int = 3,
    fig_width: float = 12,
    verbose: bool = False,
):
    # Take the list of rule index available in the solver
    range_i = solver.get_range_i()
    # Number of rows
    nrows = 1+(len(range_i)-1)//ncols
    # Width per fig
    width_per_plot = fig_width/ncols
    # Fig size (inches)
    fig_height = width_per_plot*nrows
    figsize = (
        fig_width,
        fig_height,
    )
    if verbose:
        print("\nshow_mosaic_plot_pop_vs_rule :")
        print("- nrows      =",nrows)
        print("- ncols      =",ncols)
        print("- fig_width  =",fig_width)
        print("- fig_height =",fig_height)
    # Create a single figure with a row of subplots
    fig, axes = plt.subplots(
        nrows   = nrows,
        ncols   = ncols,
        figsize = figsize,
    )
    # Flatten the 2D array of axes into a 1D array
    axes = axes.flatten()
    for i in range_i:
        show_mosaic_plot_pop_vs_rule_of_i(
            solver       = solver,
            ax           = axes[i],
            i            = i,
            do_show_comp = do_show_comp,
        )
    # Hide any remaining, unused subplots
    for i in range(len(range_i),ncols*nrows):
        axes[i].set_visible(False)
    plt.tight_layout()
    plt.show()

################################################################################
################################################################################
