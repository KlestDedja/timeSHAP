import os
import numpy as np
import matplotlib.pyplot as plt


def plot_survival_curve(
    unique_times,
    y_surv,
    fontsize=18,
):
    plt.figure()
    plt.title("Survival function $S(t)$", fontsize=fontsize)
    plt.plot(unique_times, y_surv, lw=2)
    plt.xlabel("time $t$", fontsize=fontsize)
    plt.xlim(0, max(unique_times) * 1.02)
    plt.ylabel("$S(t)$", fontsize=fontsize)
    plt.ylim(0, 1.05)
    return plt


def plot_cum_hazard_curve(
    unique_times,
    y_cum_hazard,
    fontsize=18,
):
    plt.figure()
    plt.title(r"Cum. Hazard function $\Lambda(t)$", fontsize=fontsize)
    plt.plot(unique_times, y_cum_hazard, lw=2)
    plt.xlabel("time $t$", fontsize=fontsize)
    plt.xlim(0, max(unique_times) * 1.02)
    plt.ylabel(r"$\Lambda(t)$", fontsize=fontsize)
    return plt


def plot_hazard_curve(
    unique_times,
    dy_hazard,
    fontsize=18,
):
    plt.figure()
    plt.title(r"Hazard function $\lambda(t)$", fontsize=fontsize)
    plt.plot(unique_times, 100 * dy_hazard, lw=2)
    plt.axhline(
        0, color="gray", linestyle="--", linewidth=1, zorder=0
    )  # thin line at y=0
    plt.xlabel("time $t$", fontsize=fontsize)
    plt.xlim(0, max(unique_times) * 1.02)
    plt.ylabel(r"$100\:\lambda(t)$", fontsize=fontsize)
    return plt
