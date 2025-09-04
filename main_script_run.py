# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:45:47 2024

@author:       Klest Dedja
@institution:  KU Leuven
"""
import os
import pickle
import warnings
from datetime import datetime
import gc
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont  # stitch images together, write text etc.
from IPython.display import display

from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored as c_index

# from utilities import SDT_to_dict_interval as SDT_to_dict, tree_list_to_dict_model, adjust_tick_label_size
from utilities import SurvivalModelConverter, predict_hazard_function
from utilities import auto_rename_fields
from utilities import format_timedelta, format_SHAP_values

DPI_RES = 180
DRAFT_RUN = False


if __name__ == "__main__":

    root_folder = os.getcwd()

    X = pd.read_csv(os.path.join(root_folder, "FLChain-single-event-imputed", "data.csv"))
    # X['flc_ratio'] = X['kappa']/X['lambda']
    X.rename(columns={"sample_yr": "sample_year"})
    y = pd.read_csv(
        os.path.join(root_folder, "FLChain-single-event-imputed", "targets.csv")
    ).to_records(index=False)
    y = auto_rename_fields(y)
    from utilities import map_time_to_six_bins
    # y = map_time_to_six_bins(y, time_field="time", event_field="event")

    # with open("surv_outcome.pkl", "rb") as f:
    #     y_np = pickle.load(f)
    # print(y_np.dtype)
    # y = np.rec.array(y_np, dtype=[('event', '?'), ('time', '<f8')])
    # print(y)
    # X = np.random.rand(len(y), 3)  # Example input data
    # print(X.shape)

    if DRAFT_RUN:  # faster run with less samples (almost instant)
        X = X[:500]
        y = y[:500]

    # choose whether to store in `draft-figures` folder (in .gitignore)
    # or in the normal `figures` folder
    fig_folder = "draft-figures" if DRAFT_RUN else "figures"  # pylint: disable=invalid-name
    general_figs_folder = os.path.join(root_folder, fig_folder)
    interval_figs_folder = os.path.join(root_folder, fig_folder, "interval-plots")

    clf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, n_jobs=5, random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    test_perf = c_index(y_test["event"], y_test["time"], y_pred)[0]
    print(f"Test performance: {test_perf:.4f}")

    unique_times = clf.unique_times_

    y_train_surv = clf.predict_survival_function(X_train, return_array=True)
    y_train_surv = pd.DataFrame(y_train_surv, columns=unique_times)
    y_pred_surv = clf.predict_survival_function(X_test, return_array=True)
    y_pred_surv = pd.DataFrame(y_pred_surv, columns=unique_times)

    IDX_PLOT = (
        1  # meaningful example is idx = 36 for example data with size = 700, and idx =1 for the full data
    )
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

    plt.figure()
    plt.title("Survival function $S(t)$", fontsize=FONTSIZE + 2)
    plt.plot(unique_times, y_surv_smooth, lw=2)
    plt.xlabel("time $t$", fontsize=FONTSIZE)
    plt.xlim(0, max(unique_times)*1.02)
    plt.ylabel("$S(t)$", fontsize=FONTSIZE)
    plt.ylim(0, 1.05)
    plt.savefig(os.path.join(root_folder, fig_folder, "survival-curve-example.pdf"))
    if DRAFT_RUN:
        plt.show(block=False)
        plt.pause(0.4)
    plt.close()

    plt.figure()
    plt.title(r"Cum. Hazard function $\Lambda(t)$", fontsize=FONTSIZE + 2)
    plt.plot(unique_times, y_hazard_smooth, lw=2)
    plt.xlabel("time $t$", fontsize=FONTSIZE)
    plt.xlim(0, max(unique_times)*1.02)
    plt.ylabel(r"$\Lambda(t)$", fontsize=FONTSIZE)
    plt.savefig(os.path.join(root_folder, fig_folder, "cum-hazard-curve-example.pdf"))
    if DRAFT_RUN:
        plt.show(block=False)
        plt.pause(0.4)
    plt.close()

    plt.figure()
    plt.title(r"Hazard function $\lambda(t)$", fontsize=FONTSIZE + 2)
    plt.plot(unique_times, 100 * dy_hazard_smooth, lw=2)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1, zorder=0)  # thin line at y=0
    plt.xlabel("time $t$", fontsize=FONTSIZE)
    plt.xlim(0, max(unique_times) * 1.02)
    plt.ylabel(r"$100\:\lambda(t)$", fontsize=FONTSIZE)
    plt.savefig(os.path.join(root_folder, fig_folder, "hazard-curve-example.pdf"))
    if DRAFT_RUN:
        plt.show(block=False)
        plt.pause(0.4)
    plt.close()

    convert_all = SurvivalModelConverter(clf_obj=clf, t_start=0, t_end=max(unique_times) * 1.02)

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

    GLOBAL_PLT_NAME = "Global_SHAP.pdf"

    plt.figure()
    plt.title("Global explanation", size=16)
    shap.summary_plot(shap_values, max_display=10, alpha=0.7, show=False)
    plt.xlabel("SHAP value: impact on output", size=14)
    plt.savefig(
        os.path.join(general_figs_folder, GLOBAL_PLT_NAME),
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
    time_intervals = [0, 1250, 2500, 4000, 5200]  # longer version
    # ^ nice plots, but a bit too many. Shorter version below:
    time_intervals = [0, 1825, 3600, 5100]

    # time_intervals = [0, 2, 5, 6]

    # Store interval shap values as dictionary here (intervals as keys):
    interval_shap_values = {}


    for i, t_i in enumerate(range(len(time_intervals) - 1)):

        t_start = time_intervals[t_i]
        t_end = time_intervals[t_i + 1]

        print(f"Computing SHAP values for interval: [{t_start}, {t_end}) ...")

        convert_interv = SurvivalModelConverter(clf_obj=clf, t_start=t_start, t_end=t_end)

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

        shap_values_int = explainer(X_test, check_additivity=False)
        shap_values_int = format_SHAP_values(shap_values_int, clf, X_test)
        interval_shap_values[f"{str(t_start)}-{str(t_end)}"] = shap_values_int


    # examples to explain: 3 for draft run, 12 for full data
    N = 3 if DRAFT_RUN else 8

    for i in range(N):

        t0_local_explains = datetime.now()

        y_pred_pop = y_train_surv.mean(axis=0)  # sample from training data
        y_pred_pop_med = np.percentile(y_train_surv.values, q=50, axis=0)
        y_pred_pop_low = np.percentile(y_train_surv.values, q=25, axis=0)
        y_pred_pop_high = np.percentile(y_train_surv.values, q=75, axis=0)

        y_pred_i = y_pred_surv.iloc[i, :]  # sample from TEST (unseen) data

        """ survival plot here, dump as .mpl file"""

        plt.figure()  # figsize will be overwritten after reloading,
        plt.suptitle("Predicted survival curve", size=round(7 * (DPI_RES / 72)), y=0.98)
        plt.step(
            y_pred_i.index,
            y_pred_i.values,
            where="post",
            label="$S(t)$",
            lw=2.4,
            color="purple",
        )
        plt.step(
            y_pred_pop.index,
            y_pred_pop.values,
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
        plt.xlabel("time", fontsize=round(7 * (DPI_RES / 72)))
        plt.xlim([0, None])
        plt.xticks(fontsize=round(7 * (DPI_RES / 72)))
        plt.ylabel("Survival over time", fontsize=round(7 * (DPI_RES / 72)))
        plt.yticks(np.arange(0.2, 1.15, 0.2), None, fontsize=round(7 * (DPI_RES / 72)))
        plt.yticks(np.arange(0.1, 1, 0.2), None, minor=True)
        plt.legend(fontsize=round(6 * (DPI_RES / 72)))  # loc='auto' or 'upper right'
        with open(f"temp_plot_surv_{i}.mpl", "wb") as file:
            pickle.dump(plt.gcf(), file)
        plt.legend(fontsize=round(6 * (DPI_RES / 72)))  # loc='auto' or 'upper right'
        plt.savefig(
            os.path.join(general_figs_folder, "survival-curves", f"survival_curve_idx{i}.png"),
            bbox_inches="tight",
            dpi=DPI_RES,
        )
        plt.close()

        """ local SHAP plot here: computed over entire interval """
        local_plt_name = f"Local_SHAP_idx{i}.pdf"

        fig, ax = plt.subplots(figsize=(5, 5))
        plt.sca(ax)  # make this Axes current for SHAP
        ax = shap.plots.waterfall(shap_values[i], max_display=10, show=False)
        ax.set_title("Output explanation, full interval", fontsize=round(7 * (DPI_RES / 72)))
        fig.savefig(
            os.path.join(general_figs_folder, "local-SHAP", local_plt_name),
            bbox_inches="tight",
        )
        fig.savefig(f"temp_plot_{i}_full.png", bbox_inches="tight", dpi=DPI_RES)
        plt.close(fig)  # Close the figure to free up memory

        """
        given local instance, iterate through time intervals
        (stored in previous dictionary)
        """

        for key, value in interval_shap_values.items():

            t_start, t_end = [float(s) for s in key.split("-")]
            index_t_end = np.argmax(unique_times > t_end) - 1

            local_interv_plt_name = f"Local_SHAP_idx{i}_T{key}.pdf"
            combo_local_plt_name_pdf = f"Time-SHAP_idx{i}_combined.pdf"
            combo_local_plt_name_png = f"Time-SHAP_idx{i}_combined.png"

            shap_values_use = interval_shap_values[key][i]

            ## conditional SHAP can contain NaN values:
            if np.isnan(shap_values_use.values).sum() > 0:
                shap_values_use.values[np.isnan(shap_values_use.values)] = 0
                warnings.warn(
                    "NaN values were found when computing interval-specific SHAP values,\
                possibly, the S(t_start) is estimated to be = 0"
                )

            ### TODOs:
            # - rethink the probability outputs: rescale them? They are not very intuititve atm
            # - change notation e.g. E(f(X)) and similar

            # Making figures and plots of the correct size
            single_plotwidth = max(3.4, 7 - len(interval_shap_values))

            # Check for NaN or inf in SHAP values before plotting
            if not (np.all(np.isfinite(shap_values_use.values)) and np.all(np.isfinite(shap_values_use.base_values))):
                print(f"Warning: SHAP values for interval {key} contain NaN or inf. Creating empty plot.")
                fig, ax = plt.subplots(figsize=(single_plotwidth, 7))
                ax.set_title(f"Output explanation, interval [{key})   ", fontsize=round(7 * (DPI_RES / 72)))
                ax.text(0.5, 0.5, "No valid data", ha='center', va='center', fontsize=14, color='black', transform=ax.transAxes)
                ax.axis('off')
                fig.savefig(os.path.join(interval_figs_folder, local_interv_plt_name), bbox_inches="tight", dpi=DPI_RES)
                fig.savefig(f"temp_plot_{i}_{key}.png", bbox_inches="tight", dpi=DPI_RES)
                print(f"Saving figure with size: {fig.get_size_inches()} inches")
                plt.close(fig)
            else:
                fig, ax = plt.subplots(figsize=(single_plotwidth, 7))
                plt.sca(ax)  # make this Axes current for SHAP
                ax = shap.plots.waterfall(shap_values_use, max_display=10, show=False)
                ax.set_title(
                    f"Output explanation, interval [{key})   ",
                    fontsize=round(7 * (DPI_RES / 72)),
                )
                print(f"Saving figure with size: {fig.get_size_inches()} inches")
                fig.savefig(
                    os.path.join(interval_figs_folder, local_interv_plt_name),
                    bbox_inches="tight",
                    dpi=DPI_RES,
                )
                fig.savefig(f"temp_plot_{i}_{key}.png", bbox_inches="tight", dpi=DPI_RES)
                # plt.show()
                plt.close(fig)  # Close the figure to free up memory

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

        # Here manipulate and paste the images one next to each other

        with open(f"temp_plot_surv_{i}.mpl", "rb") as file:
            fig = pickle.load(file)

        # Update the figure size
        # fit survival plot and local SHAP plot over the entire interval
        width_pad_prop = 0.03
        tot_width = single_plotwidth * (len(interval_shap_values) - 1)
        fig.set_size_inches((1 -  width_pad_prop) * tot_width, 5.3, forward=True)
        plt.tight_layout()
        plt.savefig(f"temp_plot_surv_{i}.png", bbox_inches="tight", dpi=DPI_RES)
        print(f"Saving figure with size: {fig.get_size_inches()} inches")
        plt.close()

        # Load the saved images and find the total width and height for the combined image
        surv_image = Image.open(f"temp_plot_surv_{i}.png")
        local_image = Image.open(f"temp_plot_{i}_full.png")
        images = [
            Image.open(f"temp_plot_{i}_{key}.png") for key, val in interval_shap_values.items()
        ]
        # images = [img.rotate(270, expand=True) for img in images]

        # collect widths and heights, magnitude is pixel-wise, not in inches as in mpl
        widths, heights = zip(*(i.size for i in images))

        y_pad = 10  # needed not to cut off the survival curve plot title
        y_pad_intrarow = 100  # padding between top row and bottom row
        x_pad_intrarow = -20 if len(images) > 2 else 0

        combo_height = max(heights) + surv_image.size[1] + y_pad + y_pad_intrarow
        combo_width = sum(widths) + x_pad_intrarow * (len(widths) - 1)  # N-1 gaps
        print(f"Creating PIL image with size: {combo_width}x{combo_height} pixels")

        # Create a new image with the appropriate size to contain all the plots
        combo_image = Image.new("RGB", (combo_width, combo_height), color=(255, 255, 255))

        # Paste the matplotlib survival curve on top, add some padding on the x_axis
        # bear in mind, the unit of measure is in pixels so the 2 vars must be integers
        pos_left = (
            (combo_width - surv_image.size[0] - local_image.size[0]) // 2 + x_pad_intrarow * len(widths) - x_pad_intrarow
        )
        pos_right = (
            (combo_width + surv_image.size[0] - local_image.size[0]) // 2 - x_pad_intrarow * len(widths) + x_pad_intrarow
        )

        combo_image.paste(surv_image, (pos_left, y_pad))
        # Paste the overall local SHAP next
        combo_image.paste(local_image, (pos_right, y_pad))

        # Paste each of the other images below the matplotlib image
        x_offset = 0
        y_offset = (
            surv_image.size[1] + y_pad + y_pad_intrarow
        )  # Pasting below the survival curve image
        for img in images:
            combo_image.paste(img, (x_offset, y_offset))
            x_offset += img.size[0]  # Update the x_offset by the width of the current image
            x_offset += x_pad_intrarow  # And add the x padding

        local_image.close()
        surv_image.close()

        # add title image on top of the current collage

        TITLE_SHAP_PLOT = "Time-SHAP explanation"  # for sample instance i={i}
        font_size = round(28 * (DPI_RES / 72))  # Adjust title size. Scale is relative to dpi=72
        font = ImageFont.truetype("arial.ttf", font_size)  # insert correct font path here

        # Determine the size required for the title text
        draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))  # Temp image for calculating text size
        try:  # newer Pillow >=11.3.0
            bbox = draw.textbbox((0, 0), TITLE_SHAP_PLOT, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:  # older Pillow versions, probably Pillow <10.0
            text_width, text_height = draw.textsize(TITLE_SHAP_PLOT, font=font)  # older Pillow

        v_padding = 20  # otherwise bottom part of the title can be cut

        title_image = Image.new(
            "RGB", (combo_width, text_height + v_padding), color=(255, 255, 255)
        )  # Added padding
        # Initialize drawing context
        draw = ImageDraw.Draw(title_image)
        # Calculate text position (centered)
        text_x = (title_image.width - text_width) / 2
        text_y = (title_image.height - text_height) / 2
        # Draw the text
        draw.text((text_x, text_y), TITLE_SHAP_PLOT, fill="black", font=font)

        # Create a new image with a height that includes both the title and the combo images
        final_image = Image.new(
            "RGB",
            (combo_width, title_image.height + combo_height),
            color=(255, 255, 255),
        )
        # Paste the combo image below the title first
        final_image.paste(combo_image, (0, title_image.height - v_padding))
        # Now paste the title above it
        final_image.paste(title_image, (0, 0))
        final_image.save(
            os.path.join(general_figs_folder, "combo-plots", combo_local_plt_name_pdf),
            dpi=(DPI_RES, DPI_RES),
        )  # overwrite combo_image
        final_image.save(
            os.path.join(general_figs_folder, "combo-plots", combo_local_plt_name_png),
            dpi=(DPI_RES, DPI_RES),
        )  # overwrite combo_image

        display(final_image)

        print(
            "Combined plot stored in:",
            os.path.join(general_figs_folder, "combo-plots", combo_local_plt_name_pdf),
        )

        # display(combo_image)

        # Clean up the temporary images. Explicit garbage collection is necessary
        gc.collect()
        import time
        time.sleep(0.1)  # give some time to the OS to release the files
        os.remove(f"temp_plot_surv_{i}.png")
        os.remove(f"temp_plot_surv_{i}.mpl")
        os.remove(f"temp_plot_{i}_full.png")

        # TODO: improve management of intervals and keys (tuples vs strings)

        for key, value in interval_shap_values.items():
            os.remove(f"temp_plot_{i}_{key}.png")

        t1_local_explains = datetime.now()
        time_local_explains = format_timedelta(t1_local_explains - t0_local_explains, "mm:ss:ms")
        print(f"Plotted time-SHAP explanations in: {time_local_explains}")
