# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:45:47 2024

@author:       Klest Dedja
@institution:  KU Leuven
"""
import os
import warnings
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont  # stitch images together, write text etc.

from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored as c_index

from plotting_utils import (
    plot_survival_curve,
    plot_hazard_curve,
    plot_cum_hazard_curve,
)
from utilities import SurvivalModelConverter, predict_hazard_function
from utilities import auto_rename_fields
from utilities import format_SHAP_values
from utilities import save_placeholder_plot

DPI_RES = 180
DRAFT_RUN = False

if __name__ == "__main__":

    root_folder = os.getcwd()

    X = pd.read_csv(
        os.path.join(root_folder, "FLChain-single-event-imputed", "data.csv")
    )
    X.rename(columns={"sample_yr": "sample_year"}, inplace=True)
    y = pd.read_csv(
        os.path.join(root_folder, "FLChain-single-event-imputed", "targets.csv")
    ).to_records(index=False)
    y = auto_rename_fields(y)

    if DRAFT_RUN:  # faster run with less samples (almost instant)
        X = X[:500]
        y = y[:500]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    # choose whether to store in `draft-figures` folder (in .gitignore)
    # or in the normal `figures` folder
    fig_folder = (
        "draft-figures" if DRAFT_RUN else "figures"
    )  # pylint: disable=invalid-name
    figures_main_folder = os.path.join(root_folder, fig_folder)
    local_interv_figs_folder = os.path.join(figures_main_folder, "local-interval-plots")
    global_interv_figs_folder = os.path.join(
        figures_main_folder, "global-interval-plots"
    )
    survival_figs_folder = os.path.join(figures_main_folder, "survival-curves")
    local_shap_folder = os.path.join(figures_main_folder, "local-SHAP")
    global_shap_folder = os.path.join(figures_main_folder, "global-SHAP")

    clf = RandomSurvivalForest(
        n_estimators=100, min_samples_split=10, n_jobs=5, random_state=0
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    test_perf = c_index(y_test["event"], y_test["time"], y_pred)[0]
    print(f"Test performance: {test_perf:.4f}")

    unique_times = clf.unique_times_

    y_train_surv = clf.predict_survival_function(X_train, return_array=True)
    y_train_surv = pd.DataFrame(y_train_surv, columns=unique_times)
    y_pred_surv = clf.predict_survival_function(X_test, return_array=True)
    y_pred_surv = pd.DataFrame(y_pred_surv, columns=unique_times)

    IDX_PLOT = 1  # pick one index for plotting example
    FONTSIZE = 14

    y_survs = clf.predict_survival_function(X_test, return_array=True)
    y_surv = clf.predict_survival_function(X_test)[IDX_PLOT].y
    y_hazard = clf.predict_cumulative_hazard_function(X_test)[IDX_PLOT].y

    dy_hazard = predict_hazard_function(clf, X_test, event_times="auto")[IDX_PLOT]

    from utilities import rolling_kernel

    # smoothen out S(t), Lambda(t) and lambda(t) curves (prettify)
    KERNEL_SIZE = 20 if len(unique_times) > 200 else 2

    y_surv_smooth = rolling_kernel(y_surv, kernel_size=KERNEL_SIZE)
    y_hazard_smooth = rolling_kernel(y_hazard, kernel_size=KERNEL_SIZE)
    dy_hazard_smooth = rolling_kernel(dy_hazard, kernel_size=KERNEL_SIZE)

    plt = plot_survival_curve(unique_times, y_surv_smooth, fontsize=FONTSIZE)
    plt.savefig(os.path.join(root_folder, fig_folder, "survival-curve-example.pdf"))
    if DRAFT_RUN:
        plt.show(block=False)
        plt.pause(0.4)
    plt.close()

    plt = plot_cum_hazard_curve(unique_times, y_hazard_smooth, fontsize=FONTSIZE)
    plt.savefig(os.path.join(root_folder, fig_folder, "cum-hazard-curve-example.pdf"))
    if DRAFT_RUN:
        plt.show(block=False)
        plt.pause(0.4)
    plt.close()

    plt = plot_hazard_curve(unique_times, dy_hazard_smooth, fontsize=FONTSIZE)
    plt.savefig(os.path.join(root_folder, fig_folder, "hazard-curve-example.pdf"))
    if DRAFT_RUN:
        plt.show(block=False)
        plt.pause(0.4)
    plt.close()

    convert_all = SurvivalModelConverter(
        clf_obj=clf, t_start=0, t_end=max(unique_times) * 1.02
    )

    clf_dict = []
    tree_dicts = [
        convert_all.surv_tree_to_dict(idx=i, output_format="probability")
        for i in range(len(clf.estimators_))
    ]
    clf_dict = convert_all.tree_list_to_dict_model(
        tree_list=tree_dicts, learning_weight=1 / len(tree_dicts)
    )

    explainer = shap.TreeExplainer(
        model=clf_dict,
        data=None,
        model_output="raw",
        feature_perturbation="tree_path_dependent",
    )

    shap_values = explainer(X_test, check_additivity=True)
    shap_values = format_SHAP_values(shap_values, clf, X_test)

    shap.summary_plot(shap_values, max_display=10, alpha=0.7, show=False)
    plt.title("Global explanation, full interval", fontsize=16)
    plt.xlabel("SHAP value: impact on output", fontsize=14)
    for fmt in ["png", "pdf"]:
        plt.savefig(
            os.path.join(global_shap_folder, f"Global_SHAP.{fmt}"),
            bbox_inches="tight",
            dpi=DPI_RES,
        )
    if DRAFT_RUN:
        plt.show(block=False)
        plt.pause(0.4)
    plt.close()

    """
    LOCAL EXPLANATIONS HERE: split explanations in time intervals
    (loop over such intervals, binarise outputs, store resulting SHAP values)
    """
    # split timeline in intervals and explain each segment
    time_intervals = [0, 1250, 2500, 3600, 5200]  # longer version
    # ^ nice plots, but a bit too many. Shorter version below:
    # time_intervals = [0, 1825, 3600, 5100]

    # time_intervals = [0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]

    # Store interval shap values as dictionary here (intervals as keys):
    interval_shap_values = {}

    for i, t_i in enumerate(range(len(time_intervals) - 1)):

        t_start = time_intervals[t_i]
        t_end = time_intervals[t_i + 1]

        print(f"Computing SHAP values for interval: [{t_start}, {t_end}) ...")

        convert_interv = SurvivalModelConverter(
            clf_obj=clf, t_start=t_start, t_end=t_end
        )

        clf_interv = []
        tree_intervs = [
            convert_interv.surv_tree_to_dict(idx=i, output_format="auto")
            for i in range(len(clf.estimators_))
        ]
        clf_interv = convert_interv.tree_list_to_dict_model(
            tree_list=tree_intervs, learning_weight=1 / len(tree_intervs)
        )

        explainer = shap.TreeExplainer(
            model=clf_interv,
            data=None,
            model_output="raw",
            feature_perturbation="tree_path_dependent",
        )

        interval_str = f"{str(t_start)}-{str(t_end)}"

        shap_values_int = explainer(X_test, check_additivity=False)
        shap_values_int = format_SHAP_values(shap_values_int, clf, X_test)
        interval_shap_values[interval_str] = shap_values_int

        INTERVAL_PLT_NAME = f"Global_interval_SHAP_T{interval_str}"

        shap.summary_plot(shap_values_int, max_display=10, alpha=0.7, show=False)
        plt.title(f"Global explanation, interval [{t_start}, {t_end})", fontsize=16)
        plt.xlabel("SHAP value: impact on output", fontsize=14)
        for fmt in ["png", "pdf"]:
            plt.savefig(
                os.path.join(global_interv_figs_folder, f"{INTERVAL_PLT_NAME}.{fmt}"),
                bbox_inches="tight",
                dpi=DPI_RES,
            )
        if DRAFT_RUN:
            plt.show(block=False)
            plt.pause(0.4)
        plt.close()

    # examples to explain: 3 for draft run, 8 to 12 for full data
    N = 3 if DRAFT_RUN else 10

    for i in range(N):

        y_pred_pop = y_train_surv.mean(axis=0)  # sample from training data
        y_pred_pop_med = np.percentile(y_train_surv.values, q=50, axis=0)
        y_pred_pop_low = np.percentile(y_train_surv.values, q=25, axis=0)
        y_pred_pop_high = np.percentile(y_train_surv.values, q=75, axis=0)

        y_pred_i = y_pred_surv.iloc[i, :]  # sample from TEST (unseen) data

        """ survival plot here"""

        if len(time_intervals) < 2:
            raise ValueError("At least two time intervals are required for timeSHAP.")
        if len(time_intervals) == 2:
            surv_curv_figsize = (6, 5.5)
        elif len(time_intervals) == 3:
            surv_curv_figsize = (8, 5.5)
        else:
            surv_curv_figsize = (9, 5.5)

        plt.figure(figsize=surv_curv_figsize)
        plt.suptitle("Predicted survival curve", size=round(7 * (DPI_RES / 72)), y=0.98)
        plt.step(
            y_pred_i.index,
            y_pred_i.to_numpy(),
            where="post",
            label="$S(t)$",
            lw=2.4,
            color="purple",
        )
        plt.step(
            y_pred_pop.index,
            y_pred_pop.to_numpy(),
            where="post",
            label="population",
            lw=1.6,
            color="forestgreen",
        )

        plt.fill_between(
            unique_times,
            y_pred_pop_low,
            y_pred_pop_high,
            alpha=0.3,
            label="P25-P75",
            color="forestgreen",
        )

        for t in time_intervals:
            plt.axvline(x=t, color="k", linestyle="--", linewidth=1, alpha=0.7)
        plt.xlabel("time", fontsize=round(6 * (DPI_RES / 72)))
        plt.xlim([0, None])
        plt.xticks(fontsize=round(6 * (DPI_RES / 72)))
        plt.ylabel("Survival over time", fontsize=round(6 * (DPI_RES / 72)))
        plt.yticks(np.arange(0.2, 1.15, 0.2), None, fontsize=round(6 * (DPI_RES / 72)))
        plt.yticks(np.arange(0.1, 1, 0.2), None, minor=True)
        plt.legend(fontsize=round(6 * (DPI_RES / 72)))  # loc='auto' or 'upper right'
        plt.savefig(
            os.path.join(survival_figs_folder, f"survival_curve_idx{i}.png"),
            bbox_inches="tight",
            dpi=DPI_RES,
        )
        plt.close()

        """ local and global SHAP plot here: computed over entire interval """
        local_plt_name = f"Local_SHAP_idx{i}"

        # Robust check for SHAP values: any non-finite value raises an exception:
        shap_vals = shap_values[i].values
        shap_base = shap_values[i].base_values

        if (not np.any(np.isfinite(shap_vals))) or (
            not np.isfinite(shap_base) or np.unique(shap_vals).size < 2
        ):
            print(
                f"Warning: SHAP values either all equal, or are all NaN or inf. Creating empty plot."
            )
            save_placeholder_plot(
                local_shap_folder, f"{local_plt_name}.png", dpi_res=DPI_RES
            )
        else:
            fig, ax = plt.subplots(figsize=(5, 5))
            plt.sca(ax)  # make this Axes current for SHAP
            ax = shap.plots.waterfall(shap_values[i], max_display=10, show=False)
            ax.set_title(
                "Output explanation, full interval", fontsize=round(7 * (DPI_RES / 72))
            )
            for fmt in ["png", "pdf"]:
                plt.savefig(
                    os.path.join(local_shap_folder, f"{local_plt_name}.{fmt}"),
                    bbox_inches="tight",
                    dpi=DPI_RES,
                )
            plt.close(fig)  # Close the figure to free up memory
            # Check file size (png) after saving:
            png_path = os.path.join(local_shap_folder, f"{local_plt_name}.png")
            if (
                os.path.exists(png_path) and os.path.getsize(png_path) > 5 * 1024 * 1024
            ):  # > 5 MB
                warnings.warn(
                    "SHAP (full interval) waterfall plot file is too large. Creating empty plot."
                )
                os.remove(png_path)
                save_placeholder_plot(
                    local_shap_folder, f"{local_plt_name}.png", dpi_res=DPI_RES
                )

        """
        given local instance, iterate through time intervals
        (stored in previous dictionary)
        """
        ## Now generating and storing interval-specific plots

        for key, value in interval_shap_values.items():

            local_interv_plt_name = f"Local_SHAP_idx{i}_T{key}"

            t_start, t_end = [float(s) for s in key.split("-")]
            index_t_end = np.argmax(unique_times > t_end) - 1

            shap_values_use = interval_shap_values[key][i]

            ## conditional SHAP can contain NaN values:
            if np.isnan(shap_values_use.values).sum() > 0:
                shap_values_use.values[np.isnan(shap_values_use.values)] = 0
                warnings.warn(
                    "NaN values were found when computing interval-specific SHAP values,\
                possibly, S(t_start) is estimated = 0. Setting NaNs to 0."
                )

            ### TODOs:
            # - rethink the probability outputs: rescale them? They are not very intuititve atm
            # - change notation e.g. E(f(X)) and similar

            # Check for NaN or inf in SHAP values before plotting
            if not (
                np.all(np.isfinite(shap_values_use.values))
                and np.all(np.isfinite(shap_values_use.base_values))
            ):
                warnings.warn(
                    f"SHAP values for interval {key} contain NaN or inf. Creating empty plot."
                )
                fig, ax = plt.subplots(figsize=(5, 7))
                ax.set_title(
                    f"Output explanation, interval [{key})   ",
                    fontsize=round(7 * (DPI_RES / 72)),
                )
                ax.text(
                    0.5,
                    0.5,
                    "No valid data",
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="black",
                    transform=ax.transAxes,
                )
                ax.axis("off")

                for fmt in ["png", "pdf"]:
                    fig.savefig(
                        os.path.join(
                            local_interv_figs_folder,
                            f"{local_interv_plt_name}.{fmt}",
                        ),
                        bbox_inches="tight",
                        dpi=DPI_RES,
                    )

                print(f"Saving figure with size: {fig.get_size_inches()} inches")
                plt.close(fig)
            else:  # No NaN values found, saving png file only (no PDF file)
                fig, ax = plt.subplots(figsize=(5, 7))
                plt.sca(ax)  # make this Axes current for SHAP (needed?)
                ax = shap.plots.waterfall(shap_values_use, max_display=10, show=False)
                ax.set_title(
                    f"Output explanation, interval [{key})   ",
                    fontsize=round(7 * (DPI_RES / 72)),
                )
                print(f"Saving figure with size: {fig.get_size_inches()} inches")
                fig.savefig(
                    os.path.join(
                        local_interv_figs_folder, f"{local_interv_plt_name}.png"
                    ),
                    bbox_inches="tight",
                    dpi=DPI_RES,
                )
                plt.close(fig)

            print(f"TIME INTERVAL: [{key})")
            print(
                f"Sample prediction, SHAP based: {shap_values_use.base_values + shap_values_use.values.sum():.4f}"
            )
            print(
                f"Population prediction, test data, SHAP based: {shap_values_use.base_values:.4f}"
            )
            print(
                f"Population prediction, train data, 1-S({t_end:.1f}): {1-y_pred_pop.iloc[index_t_end]:.4f}"
            )

            # interval loop closed, now let's load images and paste them one next to each other

        ######    INTERVAL SHAP PLOT, with local SHAP plot    ######
        # load image folders and interval keys
        interval_keys = list(interval_shap_values.keys())

        # prepare dict with folder paths for combo image
        folders = {
            "survival_curves": survival_figs_folder,
            "local_shap": local_shap_folder,
            "local_interval_shap": local_interv_figs_folder,
        }

        MAX_ADMITTED_PER_ROW = 4

        # Step 1: get all image paths needed to build the first combo image, and load them:
        from paste_combo_image import get_images_from_paths

        image_paths = get_images_from_paths(i, interval_keys, **folders)
        images = [Image.open(p) for p in image_paths if os.path.exists(p)]

        bottom_images = images[2:] if len(images) > 2 else images[1:]
        top_images = images[:2] if len(images) >= 2 else images[:1]

        scale_factor = 1.3  # make top row images bigger
        top_images = [
            im.resize((int(im.width * scale_factor), int(im.height * scale_factor)))
            for im in top_images
        ]

        # Step 2: Calculate layout (size of combo image), position all sub-images
        # accoridingly, treat top row and bottom rows separately
        from paste_combo_image import (
            calculate_layout,
            compute_top_row_x_pos,
            place_top_row,
            place_bottom_rows,
        )

        layout = calculate_layout(
            top_images,
            bottom_images,
            max_admitted_per_row=MAX_ADMITTED_PER_ROW,
        )
        # Now that we know the size, create blank canvas for combo image
        combo_image = Image.new(
            "RGB",
            (layout["combo_width"], layout["combo_height"]),
            color=(255, 255, 255),
        )
        # Compute x-positions for top row
        positions = compute_top_row_x_pos(layout["combo_width"], layout["top_widths"])
        # Place top row and bottom row images:
        place_top_row(combo_image, top_images, positions, layout["y_pad_top"])
        place_bottom_rows(combo_image, bottom_images, layout)

        # Cleanup
        for im in images:
            im.close()

        from paste_combo_image import (
            render_title,
            assemble_image_title,
            display_image,
        )

        # Step 3: rendering title, final assembly, store and display final combo-image:
        title_image = render_title(
            layout["combo_width"], f"Time-SHAP explanation", DPI_RES
        )

        final_image = assemble_image_title(
            title_image,
            combo_image,
            layout["combo_width"],
            layout["combo_height"],
            title_padding=20,
        )
        # Save final combo image
        out_dir = os.path.join(figures_main_folder, "combo-plots")
        os.makedirs(out_dir, exist_ok=True)

        combo_local_name = f"Local-timeSHAP_idx{i}_combined"

        for fmt in ["pdf", "png"]:
            save_combo_file = os.path.join(out_dir, f"{combo_local_name}.{fmt}")
            final_image.save(save_combo_file, dpi=(DPI_RES, DPI_RES))

        display_image(final_image)

        #############       GLOBAL INTERVAL SHAP PLOTS        #############
        # repeat the same procedure as above, but now for global SHAP plots
        #############                                         #############

        global_folders = {
            "global_shap": global_shap_folder,
            "global_interval_shap": global_interv_figs_folder,
        }

        MAX_ADMITTED_PER_ROW = 3

        # Step 1: get all image paths needed to build the second combo image, and load them:

        image_paths_global = get_images_from_paths(i, interval_keys, **global_folders)
        images_global = [Image.open(p) for p in image_paths_global if os.path.exists(p)]

        top_images_global = images_global[:1]
        bottom_images_global = images_global[1:]

        scale_factor = 1.3  # make top row images bigger
        top_images_global = [
            im.resize((int(im.width * scale_factor), int(im.height * scale_factor)))
            for im in top_images_global
        ]

        # Step 2: Calculate layout (size of combo image), position all sub-images
        # accordingly. Treat top row and bottom rows separately

        layout = calculate_layout(
            top_images_global,
            bottom_images_global,
            max_admitted_per_row=MAX_ADMITTED_PER_ROW,
        )
        # Now that we know the size, create blank canvas for combo image
        combo_image_global = Image.new(
            "RGB",
            (layout["combo_width"], layout["combo_height"]),
            color=(255, 255, 255),
        )
        # Compute x-positions for top row
        positions = compute_top_row_x_pos(layout["combo_width"], layout["top_widths"])
        # Place top row and bottom row images:
        place_top_row(
            combo_image_global, top_images_global, positions, layout["y_pad_top"]
        )
        place_bottom_rows(combo_image_global, bottom_images_global, layout)

        # Cleanup
        for im in images_global:
            im.close()

        # Step 3: rendering title, final assembly, store and display final combo-image:
        # we provide more padding for the title for global combo image
        title_image_global = render_title(
            layout["combo_width"],
            f"Time-SHAP explanation",
            DPI_RES,
            font_size=28,
            title_padding=40,
        )

        final_image_global = assemble_image_title(
            title_image_global,
            combo_image_global,
            layout["combo_width"],
            layout["combo_height"],
            title_padding=20,
        )
        # Save final combo image
        combo_image_name = f"Global-timeSHAP_idx{i}_combined"

        out_dir = os.path.join(figures_main_folder, "combo-plots")
        os.makedirs(out_dir, exist_ok=True)

        for fmt in ["pdf", "png"]:
            save_combo_file = os.path.join(out_dir, f"{combo_image_name}.{fmt}")
            final_image_global.save(save_combo_file, dpi=(DPI_RES, DPI_RES))

        display_image(final_image_global)
