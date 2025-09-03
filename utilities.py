# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:45:47 2024

@author:       Klest Dedja
@institution:  KU Leuven
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sksurv.tree import SurvivalTree
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.metrics import concordance_index_censored as c_index


def adjust_tick_label_size(xfontsize=None, yfontsize=None):
    """
    Adjusts the font size of the tick labels for both the x-axis and y-axis.
    Parameters:
    - xfontsize: The new font size for the x-axis tick labels. If None, x-axis font size is not changed.
    - yfontsize: The new font size for the y-axis tick labels. If None, y-axis font size is not changed.
    """
    ax = plt.gca()
    if xfontsize is not None:
        for label in ax.get_xticklabels():
            label.set_fontsize(xfontsize)
    if yfontsize is not None:
        for label in ax.get_yticklabels():
            label.set_fontsize(yfontsize)


def rename_fields_dataframe(df):
    rename_map = {}
    for column in df.columns:
        # Detect binary column
        if (
            df[column].dropna().apply(float.is_integer).all()
            and df[column].nunique() == 2
        ):
            rename_map[column] = "event"
        # Detect floating-point column
        elif df[column].dtype == float or df[column].dtype == np.float64:
            rename_map[column] = "time"
    return df.rename(columns=rename_map)


def rename_fields_recarray(recarray):
    rename_map = {}
    set_dtypes = []

    # Construct new dtype array while building the rename map
    for name, dtype in recarray.dtype.descr:
        values = recarray[name]
        # Detect binary field
        if np.all(np.isin(values, [0, 1])) and len(np.unique(values)) == 2:
            new_name = "event"
            set_dtypes.append((new_name, dtype))
        # Detect floating-point field
        else:
            new_name = "time"
            set_dtypes.append((new_name, "float64"))  # Change dtype to float64 directly

        rename_map[name] = new_name

    # Create a new recarray with the new dtype
    new_recarray = np.recarray(recarray.shape, dtype=set_dtypes)

    # Copy and cast data from the old recarray to the new recarray
    for old_name, (new_name, new_type) in zip(recarray.dtype.names, set_dtypes):
        if new_name == "time" and recarray[old_name].dtype != np.float64:
            new_recarray[new_name] = recarray[old_name].astype(np.float64)
        else:
            new_recarray[new_name] = recarray[old_name]

    return new_recarray


def auto_rename_fields(data):
    if isinstance(data, pd.DataFrame):
        return rename_fields_dataframe(data)
    elif isinstance(data, np.recarray):
        return rename_fields_recarray(data)
    else:
        raise TypeError("Unsupported data type. Expected pd.DataFrame or np.recarray.")


# potentially custom scorers for scikit-learn's cross_val_score method
def concordance_index_scorer(estimator, X, y):
    event_indicator = y["event"]
    event_time = y["time"]
    risk_scores = estimator.predict(X)
    concord_index, _ = c_index(event_indicator, event_time, risk_scores)
    return concord_index


# Wrapper function to match the expected signature of make_scorer
def concordance_index_scorer_wrapper(y_true, y_pred, estimator, X):
    return concordance_index_scorer(estimator, X, y_true)


def predict_hazard_function(clf, X, event_times="auto", smooth=False):

    if event_times == "auto":
        event_times = clf.unique_times_

    # basically, take the derivative of the cumulated hazard function

    dx = np.diff(event_times, prepend=0)

    y_hazards = clf.predict_cumulative_hazard_function(X, return_array=True)

    dy = np.diff(y_hazards, axis=1, prepend=0)
    dx = dx.reshape(1, -1)
    df = dy / dx

    return df


# def take_derivative(y_original, dx, axis=1,):
#     # y_hazards = clf.predict_cumulative_hazard_function(X, return_array=True)
#     dy = np.diff(y_original, axis=axis, prepend=0)
#     dx = dx.reshape(1, -1)
#     df = dy/dx

#     return df


# from scipy.stats import gaussian_kde


def rolling_kernel(curve, kernel_size=10):


    if isinstance(kernel_size, int):
        out_kernel = np.ones(kernel_size) / kernel_size  # 1/ N, N times (uniform)
        out_kernel = out_kernel / np.sum(out_kernel)
    else:
        raise ValueError("kernel_size must be an integer.")
    # Smooth the array along the rows
    if kernel_size > 1:
        smoothed_curve = np.convolve(curve, out_kernel, mode="valid")
        pad_l = len(curve) - len(smoothed_curve)

        end_values = (curve[0], curve[-1])

        padded_curve = np.pad(
            smoothed_curve,
            pad_width=(pad_l // 2, pad_l - pad_l // 2),
            mode="linear_ramp",
            end_values=end_values,
        )
    else:
        padded_curve = curve

    return padded_curve


class SurvivalModelConverter:

    def __init__(self, clf_obj, t_start=0, t_end=None, fallback_prob_ratio=0):
        self.clf_obj = clf_obj
        self.t_start = t_start
        self.t_end = t_end
        self.fallback_prob_ratio = fallback_prob_ratio

        params_to_validate = {
            "t_start": t_start,
            "t_end": t_end,
            "fallback_prob_ratio": fallback_prob_ratio,
        }

        self._validate_inputs(**params_to_validate)

    def _validate_inputs(self, **kwargs):
        for var_name, var_value in kwargs.items():
            if not isinstance(var_value, (float, int, type(None))):
                raise TypeError(
                    f"Variable '{var_name}' must be float, integer, or None, got {type(var_value).__name__}."
                )

    def surv_tree_to_dict(self, idx, output_format):
        """
        Converts Survival Decision Trees to a dictionary format.
        Currently compatible with single-output trees only.
        """
        tree_obj = self._get_tree_object(idx)
        tree_dict = self._tree_structure_as_dict(tree_obj, output_format)

        if isinstance(tree_obj, SurvivalTree):
            tree_dict = self._handle_intervals_survival_tree(
                tree_obj, tree_dict, output_format
            )
        else:
            raise ValueError(f"Unsupported tree object type: {type(tree_obj)}")

        tree_dict["base_offset"] = tree_dict["values"][0]  # root node (0) prediction
        return tree_dict

    def tree_list_to_dict_model(
        self, tree_list, is_gradient_based=False, learning_weight=1.0
    ):
        """
        Converts a list of tree dictionaries into a model dictionary.
        """
        # Handle learning weights
        learning_weight = self._set_learning_weight(learning_weight, len(tree_list))

        # Adjust tree values and base offsets based on learning weights
        for i, t in enumerate(tree_list):
            t["values"] *= learning_weight[i]
            t["prior_values"] *= learning_weight[i]
            t["base_offset"] *= learning_weight[i]

        base_offset = self._compute_base_offset(tree_list, is_gradient_based)
        output_format, ensemble_class = self._ensure_consistency(tree_list)

        return {
            "trees": tree_list,
            "base_offset": base_offset,
            "output_format": output_format,
            "ensemble_class": ensemble_class,
            "input_dtype": np.float32,
            "internal_dtype": np.float32,
        }

    def _get_tree_object(self, idx):
        """
        Extracts the tree object from the classifier. Works for SurvivalTree in
        RandomSurvivalForest.
        TODO: make ti work for RegressionTree in GradientBosostingSurvivalAnalysis.
        """
        if isinstance(self.clf_obj, RandomSurvivalForest):
            tree_obj = self.clf_obj[idx]  # Default behaviour for accessing tree
        elif isinstance(self.clf_obj, GradientBoostingSurvivalAnalysis):
            tree_obj = self.clf_obj[
                idx
            ]  # this is now a weird np.array object with one elemenet
            assert tree_obj.shape == (1,)  # as of scikit-survival 0.22.1
            tree_obj = tree_obj[0]
        return tree_obj

    def _tree_structure_as_dict(self, tree_obj, output_format):
        """
        Extracts the structure of a tree and returns it as a dictionary.
        Integrates with some extra properties
        """
        tree = tree_obj.tree_
        return {
            "children_left": tree.children_left,
            "children_right": tree.children_right,
            "children_default": tree.children_right.copy(),  # Placeholder for future missing value handling
            "features": tree.feature,
            "thresholds": tree.threshold,
            "node_sample_weight": tree.weighted_n_node_samples,
            "n_features_in_": getattr(tree_obj, "n_features_in_", None),
            "feature_names_in_": getattr(self.clf_obj, "feature_names_in_", None),
            "unique_times_": getattr(tree_obj, "unique_times_", None),
            "is_event_time_": getattr(tree_obj, "is_event_time_", None),
            "random_state": getattr(tree_obj, "random_state", None),
            "ensemble_class": self.clf_obj.__class__.__name__,
            "learner_class": tree_obj.__class__.__name__,
            "output_format": output_format,
        }

    def _handle_intervals_survival_tree(self, tree_obj, tree_dict, output_format):
        """
        Processes survival tree-specific logic: manages the definition of intervals
        and computes the conditional probabilities.
        """
        if tree_dict["unique_times_"] is None and output_format not in ["probability"]:
            raise KeyError("Missing 'unique_times_' in SurvivalTree-based ensemble.")

        if self.t_end in ["end", "end_time"]:
            self.t_end = tree_dict["unique_times_"][-1] + 1
        if self.t_end is None and tree_dict["unique_times_"] is not None:
            # Select median time to (any) event if self.t_end not specified
            self.t_end = tree_dict["unique_times_"][
                len(tree_dict["unique_times_"]) // 2
            ]

        if self.t_end is not None and self.t_end <= self.t_start:
            raise ValueError(
                f"Endpoint {self.t_end} is not strictly greater than t_start={self.t_start}"
            )

        # Compute indices for self.t_start and self.t_end in `unique_times_`
        # identifies the last element of (sorted array) unique_times_ such that
        # element_start <= t_start and element_end < t_end. Remember argmax returns the FIRST maximum arg
        # TODO busy right here:
        index_start = np.argmax(tree_obj.unique_times_ > self.t_start) - 1
        index_end = np.argmax(tree_obj.unique_times_ > self.t_end) - 1
        # This leaves out some edge cases, treated here below:
        # if self.t_start is smaller than all `unique_times_`
        if tree_obj.unique_times_[0] > self.t_start:
            index_start = 0
        # if self.t_end is greater than all `unique_times_`: not allowed!
        if tree_obj.unique_times_[0] > self.t_end:
            raise ValueError(
                f"t_end={self.t_end:.4f} is too small. Must be greater than {min(tree_obj.unique_times_):.4f}"
            )
        if index_start >= index_end and index_end != -1:
            raise ValueError(
                f"Provided interval: [{self.t_start}, {self.t_end}) is too narrow, no `unique_times_` are available."
            )

        # MEMENTO: tree.value[node, sample, 0] represents H(t)
        #        - tree.value[node, sample, 1] represents S(t)

        if isinstance(tree_obj, SurvivalTree) and output_format in [
            "probability",
            "auto",
        ]:
            # Compute conditional probabilities (for each tree).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                conditional_prob = (
                    1
                    - tree_obj.tree_.value[:, index_end, 1]
                    / tree_obj.tree_.value[:, index_start, 1]
                ).reshape(-1, 1)
            # Handle division by zero or NaN values
            conditional_prob[np.isnan(conditional_prob)] = self.fallback_prob_ratio
            tree_dict["values"] = conditional_prob

            # store probability for which the event is conditioned upon: 1 - S(t)
            tree_dict["prior_values"] = 1 - tree_obj.tree_.value[
                :, index_start, 1
            ].reshape(-1, 1)

        elif isinstance(tree_obj, SurvivalTree) and output_format in [
            "hazard",
            "cum. hazard",
        ]:
            # Compute conditional probabilities (for each tree).
            tree_dict["values"] = (
                tree_obj.tree_.value[:, index_end, 0]
                - tree_obj.tree_.value[:, index_start, 0]
            ).reshape(-1, 1)

            # store hazard incurred until time t: H(t)
            tree_dict["prior_values"] = tree_obj.tree_.value[:, index_start, 0].reshape(
                -1, 1
            )

        elif isinstance(tree_obj, SurvivalTree) and output_format in ["survival"]:
            # Compute conditional survival probabilities (for each tree).
            conditional_surv = (
                tree_obj.tree_.value[:, index_end, 1]
                / tree_obj.tree_.value[:, index_start, 1]
            ).reshape(-1, 1)
            conditional_surv[np.isnan(conditional_surv)] = (
                1 - self.fallback_prob_ratio
            )  # S(t)= 1-P(t)
            tree_dict["values"] = conditional_surv
            # store probability of survival until time t: S(t)
            tree_dict["prior_values"] = tree_obj.tree_.value[:, index_start, 0].reshape(
                -1, 1
            )

        elif not isinstance(tree_obj, SurvivalTree):
            ValueError(f"Not implemented yet for learner {tree_obj.__name__}")
        else:
            raise ValueError(
                f"Unsupported output_format '{output_format}' for survival tree."
            )

        return tree_dict

    def _set_learning_weight(self, learning_weight, n_learners):
        """
        Takes care of the learning weight parameter, returns array of length (n_learners,)
        """
        if learning_weight is None:
            learning_weight = 1.0
        if isinstance(learning_weight, (float, int)):
            learning_weight = np.array([learning_weight] * n_learners)
        return learning_weight

    def _compute_base_offset(self, tree_list, is_gradient_based):
        """
        Computes the base offset (baseline) for the model.
        """
        if is_gradient_based:  # base offset is root of first learned  tree
            base_offset = tree_list[0]["base_offset"][0]
        else:  # sum ( or average??) over all root nodes. Since the learning rate is here set to 1/n_learners, take sum
            base_offset = np.sum([t["base_offset"] for t in tree_list])
        return base_offset

    def _ensure_consistency(self, tree_list):
        """
        Ensures that all trees in the list have consistent output formats and ensemble classes.
        """
        output_formats = {t["output_format"] for t in tree_list}
        ensemble_classes = {t["ensemble_class"] for t in tree_list}
        if len(output_formats) > 1 or len(ensemble_classes) > 1:
            raise ValueError(
                "Inconsistent output formats or ensemble classes in tree list."
            )
        return output_formats.pop(), ensemble_classes.pop()


def reduce_ensemble_object(clf, factor=10):

    for t in clf["trees"]:
        t["is_event_time_"] = t["is_event_time_"][::factor]
        t["unique_times_"] = t["unique_times_"][::factor]

    return clf


def format_SHAP_values(shap_values, clf, X):

    from sklearn.ensemble import RandomForestRegressor

    """ formats the shap values ndarray (reshape) depending on whether
         the clf  is a RF or a RSF. """

    if hasattr(clf, "predict_proba"):
        # then n_outputs==2 ( class 0 and 1) but we are only interested in class 1
        if shap_values.values.ndim == 3:  # double check it is the case
            shap_values = shap_values[:, :, 1]

    elif isinstance(clf, RandomForestRegressor):
        shap_values.base_values = shap_values.base_values.ravel()

    # for some reason, this is not always consistent. Fix here:
    if isinstance(shap_values.base_values, np.ndarray):
        shap_values.base_values = np.mean(shap_values.base_values)

    if (
        isinstance(shap_values.base_values, np.ndarray)
        and len(shap_values.base_values) > 1
    ):
        shap_values.base_values = shap_values.base_values[0]  # [0]

    return shap_values


def predict_proba_at_T(clf, X, t_start=0, t_end=None):

    assert t_start < t_end

    if t_end is None:
        index_end = int(len(clf.unique_times_) // 2)
        print("Analysing for t_end = ", clf.unique_times_[index_end])

    # to idenitify corresponding index, pick last "False" index before "True" appears
    index_start = np.argmax(clf.unique_times_ > t_start) - 1
    index_end = np.argmax(clf.unique_times_ > t_end) - 1

    # it does not work when all times are > T_bin, as it selects -1 instead of 0
    if min(clf.unique_times_) > t_start:
        index_start = 0
    if min(clf.unique_times_) > t_end:
        raise ValueError(
            f"Value for t_end {t_end} is too small. Mus be greater than {clf.unique_times_}"
        )

    y_surv = clf.predict_survival_function(X, return_array=True).astype(np.float32)
    # probability of experiencing the event by t=2 -> P(t) = 1 - S(t)
    return 1 - y_surv[:, index_start] / y_surv[:, index_end]  # elementwise division


def format_timedelta(td, time_format):
    total_seconds = td.total_seconds()
    hours, remainder = divmod(int(total_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)

    valid_formats = ["hh:mm:ss", "hh:mm", "mm:ss", "hh:mm:ss:ms", "hh:mm:ss:ms"]

    if time_format.lower() == "hh:mm:ss":
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    elif time_format.lower() == "hh:mm":
        return f"{hours:02d}:{minutes:02d}"
    elif time_format.lower() == "mm:ss":
        return f"{minutes:02d}:{seconds:02d}"
    elif time_format.lower() == "hh:mm:ss:ms":
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{milliseconds:03d}"
    elif time_format.lower() == "mm:ss:ms":
        return f"{60*hours+minutes:03d}:{seconds:02d}:{milliseconds:03d}"
    else:
        error_message = f"Invalid format. Valid formats are: {', '.join(valid_formats)}"
        raise ValueError(error_message)


# if __name__ == '__main__':

#     print('hello!')
