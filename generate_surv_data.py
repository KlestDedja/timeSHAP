# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:37:54 2024

@author:       Klest Dedja
@institution:  KU Leuven

Created on Wed Mar 10 14:20:47 2021

     ------    properties and ideas:    ------

-> TIME DEPENDENT CENSORING

-> SOME SUBKECTS not INDEPENDENT ( family members, or similar disease/accident)

-> latent features Z_i generate and correlate with X_i
-> include different censoring causes C_i
        -----------------------------------"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy


class SyntheticSurvivalDataGenerator:

    import numpy as np
    from numpy.random import RandomState

    def __init__(
        self,
        N,
        p,
        q,
        interaction_pairs=None,
        noise_level=0.1,
        noise_level_censor=0.1,
        params_event=[1.2, 1],
        params_censor=[1, 1],
        random_state=None,
    ):
        self.N = N
        self.p = p
        self.q = q
        self.interaction_pairs = interaction_pairs
        self.noise_level_event = noise_level_event
        self.noise_level_censor = noise_level_censor
        self.shape_event = pars_event[0]
        self.scale_event = pars_event[1]
        self.shape_censor = params_censor[0]
        self.scale_censor = params_censor[1]
        self.random_state = random_state
        # fixed random seed for coefficient generation
        # to be regenrated every time for fixed outputs
        # self.rng = np.random.default_rng(random_state)

        # Check input parameters
        if q > p:
            raise ValueError(
                f"Number of common components for modeling censoring and time-to-event ({q}) "
                f"cannot exceed number of components for modeling time-to-event ({p})."
            )

    def _sigmoid_transform(x, amplitude, stretch):
        assert stretch > 0 and amplitude > 0
        return amplitude / (1 + np.exp(-x / stretch)) - amplitude / 2

    def _my_weibull(shape, scale=None, size=None, random_state=None):
        """
        generating a two-parameter Weibull distribution

        NOTE: given Weibull(a, b) we have:
            - a: shape. a < 1 hazard decreasing over time. a > 1: increasing over time
                        a = 1 becomes exponential distribution
            - b: scale. Simple multiplicative factor.

            EXP. VALUE: b * Gamma(1 + 1/a), minimum when a ~ 2 (?)
            VARIANCE: b^2 * [ Gamma(1 + 2/a) - Gamma(1 + 1/a)**2 ]
        """

        rng = np.random.default_rng(random_state)

        if scale is None:  # default scale is = 1
            scale = 1

        if (isinstance(shape, (float, np.ndarray)) and np.any(shape <= 0)) or (
            isinstance(scale, (float, np.ndarray)) and np.any(scale <= 0)
        ):
            raise ValueError("Shape and scale parameters must be > 0")
        # assert np.shape(scale) == np.shape(size)

        return scale * rng.weibull(a=shape, size=size)

    def generate_covariates(self):
        X = self.rng.normal(loc=0, scale=1, size=(self.N, self.p))
        if self.interaction_pairs is not None:
            for i, j in self.interaction_pairs:
                X[:, i] *= X[:, j]
        return X

    def generate_time_to_event_data(self, X):

        # Create a vector of coefficients with varying importance
        # generate p+1 for reproducibility
        rng = np.random.default_rng(self.random_state)
        beta_event = rng.uniform(-2, 2, p + 1)[:p]

        # Calculate the linear predictor
        eta = np.dot(X, beta_event)
        eta = _sigmoid_transform(eta)  # compress extreme values
        # Add (gaussian) noise to the linear predictor
        eta += rng.normal(scale=self.noise_level * eta.std(), size=self.N)

        # Set the baseline hazard and calculate the hazard for each observation
        h0 = 0.1
        h_events = h0 * np.exp(eta)

        event_times = _my_weibull(
            self.alpha_weibull, scale=1 / h_events, random_state=self.random_state
        )

        return event_times

    def generate_censoring_distribution(self, X, event_times):

        # Fix the random seed for reproducibility
        random.seed(a_fixed_seed)  # Replace a_fixed_seed with your chosen seed value
        # Select 'q' random elements from the list
        selected_elements = random.sample(X.columns, q)

        rng = np.random.default_rng(random_state)
        # generate q elements for beta_censor, q-1 of which depend on the same
        # covariates as the beta_times

        # no repeated seed so no
        beta_censor = rng.uniform(-2, 2, p + 1)[p - q + 1 :]  # q elements
        # beta_censor_indep = rng.uniform(-2, 2, p-q+1)

        index_set = rng.choice(range(len(beta)), size=q, replace=False)
        beta_censor = np.empty(len(beta))
        select_indices = index_set
        remain_indices = list(set(range(len(beta))) - set(select_indices))
        beta_censor[select_indices] = beta[select_indices]

        flip = rng.choice([-1, 1], size=len(select_indices), p=[0.5, 0.5])
        beta[select_indices] = beta[select_indices] * flip
        beta_censor[remain_indices] = beta_censor_indep[remain_indices]

        eta_censor = np.dot(X, beta_censor)
        eta_censor += rng.normal(scale=noise_level, size=N)
        eta_censor = sigmoid_transform(eta_censor, amplitude=5, stretch=2)

        tot_censor = 0.1 * np.exp(eta_censor)
        censoring_times = my_weibull(
            1.1, scale=scaling_censoring / tot_censor, random_state=random_state + 1
        )

        return censoring_times, tot_censor

    def combine_data(self, X, event_times, censoring_times, h_events, h_censor):
        data = pd.DataFrame(X, columns=[f"X{i + 1}" for i in range(self.p)])
        data["time"] = np.minimum(event_times, censoring_times)
        data["event"] = (event_times <= censoring_times).astype(bool)

        df_info = pd.DataFrame(
            {
                "event_time": event_times,
                "censoring_time": censoring_times,
                "h_events": h_events,
                "h_censor": h_censor,
            }
        )

        y = data[["time", "event"]].to_records(index=False)
        return data, y, df_info

    def generate_data(self):
        X = self.generate_covariates()
        event_times, h_events = self.generate_time_to_event_data(X)
        censoring_times, h_censor = self.generate_censoring_distribution(X, event_times)
        return self.combine_data(X, event_times, censoring_times, h_events, h_censor)


def generate_time_to_event_data(
    N, p, interaction_pairs=None, noise_level=0.1, alpha_weibull=2, random_state=None
):
    rng = np.random.default_rng(random_state)
    X = rng.normal(loc=0, scale=1, size=(N, p))

    if interaction_pairs is not None:
        for i, j in interaction_pairs:
            X[:, i] *= X[:, j]

    beta = rng.uniform(-2, 2, p)
    eta = np.dot(X, beta)
    eta = sigmoid_transform(eta, amplitude=5, stretch=2)
    eta += rng.normal(scale=noise_level * eta.std(), size=N)

    h0 = 0.1
    tot_hazard = h0 * np.exp(eta)
    event_times = my_weibull(alpha_weibull, scale=1 / tot_hazard, random_state=random_state)

    return X, event_times, tot_hazard


def generate_censoring_distribution(
    X, event_times, q, p, noise_level=0.1, scaling_censoring=1, random_state=None
):
    rng = np.random.default_rng(random_state)
    beta_censor = rng.uniform(-2, 2, q)
    beta_censor_indep = rng.uniform(-2, 2, p - q)

    index_set = rng.choice(range(len(beta)), size=q, replace=False)
    beta_censor = np.empty(len(beta))
    select_indices = index_set
    remain_indices = list(set(range(len(beta))) - set(select_indices))
    beta_censor[select_indices] = beta[select_indices]

    flip = rng.choice([-1, 1], size=len(select_indices), p=[0.5, 0.5])
    beta[select_indices] = beta[select_indices] * flip
    beta_censor[remain_indices] = beta_censor_indep[remain_indices]

    eta_censor = np.dot(X, beta_censor)
    eta_censor += rng.normal(scale=noise_level, size=N)
    eta_censor = sigmoid_transform(eta_censor, amplitude=5, stretch=2)

    tot_censor = 0.1 * np.exp(eta_censor)
    censoring_times = my_weibull(
        1.1, scale=scaling_censoring / tot_censor, random_state=random_state + 1
    )

    return censoring_times, tot_censor


def create_survival_dataframe(
    N,
    p,
    q,
    interaction_pairs=None,
    noise_level=0.1,
    alpha_weibull=2,
    scaling_censoring=1,
    random_state=None,
):

    # Generate time-to-event data
    X, event_times, h_events = generate_time_to_event_data(
        N, p, interaction_pairs, noise_level, alpha_weibull, random_state
    )

    # Generate censoring distribution
    censoring_times, h_censor = generate_censoring_distribution(
        X, event_times, q, p, noise_level, scaling_censoring, random_state
    )

    # Create the main DataFrame with covariates, time, and event indicator
    data = pd.DataFrame(X, columns=[f"X{i + 1}" for i in range(p)])
    data["time"] = np.minimum(event_times, censoring_times)
    data["event"] = (event_times <= censoring_times).astype(bool)

    # Create an additional DataFrame for detailed event and censoring information
    df_info = pd.DataFrame(
        {
            "event_time": event_times,
            "censoring_time": censoring_times,
            "h_events": h_events,
            "h_censor": h_censor,
        }
    )

    # Create a structured array for survival analysis
    y = data[["time", "event"]].to_records(index=False)

    return data, y, df_info


def generate_synthetic_data(
    N,
    p,
    q=None,
    interaction_pairs=None,
    noise_level=0.1,
    alpha_weibull=2,
    scaling_censoring=1,
    random_state=None,
):

    if q > p:
        raise ValueError(
            f"Number of common components in common for modelling\
censoring and time-to-event ({q})cannot exceed number of components for modelling\
time to event({p})."
        )

    rng = np.random.default_rng(random_state)

    # Generate random covariates, with standard normal distribution
    X = rng.normal(loc=0, scale=1, size=(N, p))

    # Create interaction terms
    if interaction_pairs is not None:
        for i, j in interaction_pairs:
            X[:, i] *= X[:, j]

    # Create a vector of coefficients with varying importance
    beta = rng.uniform(-2, 2, p)

    # Calculate the linear predictor
    eta = np.dot(X, beta)
    # Apply sigmoid transformation to control the range of the linear predictor
    eta = sigmoid_transform(eta, amplitude=5, stretch=2)

    # Add (gaussian) noise to the linear predictor
    eta += rng.normal(scale=noise_level * eta.std(), size=N)

    # Set the baseline hazard and calculate the hazard for each observation
    h0 = 0.1
    tot_hazard = h0 * np.exp(eta)

    # generate event-times arising from given hazard function
    # choose whether the risk increases over time or not

    event_times = my_weibull(alpha_weibull, scale=1 / tot_hazard, random_state=random_state)

    # We have q covariates that are in commmon with the event at hand:
    beta_censor = rng.uniform(-2, 2, q)

    # Geneate p-q new vars for modelling censoring now. List can be empty
    beta_censor_indep = rng.uniform(-2, 2, p - q)

    index_set = rng.choice(range(len(beta)), size=q, replace=False)

    # Create a new array by copying selected entries from beta and the remaining entries from beta_censor_indep
    beta_censor = np.empty(len(beta))
    select_indices = index_set
    remain_indices = list(set(range(len(beta))) - set(select_indices))
    beta_censor[select_indices] = beta[select_indices]

    flip = rng.choice([-1, 1], size=len(select_indices), p=[0.5, 0.5])
    beta[select_indices] = beta[select_indices] * flip

    beta_censor[remain_indices] = beta_censor_indep[remain_indices]

    # Calculate the linear predictor
    eta_censor = np.dot(X, beta_censor)
    # Add (gaussian) noise to the linear predictor
    eta_censor += rng.normal(scale=noise_level, size=N)

    # Apply sigmoid transformation to control the range of the linear predictor
    eta_censor = sigmoid_transform(eta_censor, amplitude=5, stretch=2)
    # is the *eta.std() missing ehre?

    corr_coeff = np.corrcoef(eta, eta_censor)[0, 1]

    plt.scatter(x=eta, y=eta_censor, s=8)
    plt.title(f"Correlation of event and censoring hazards: {corr_coeff:.3f}")
    plt.xlabel("eta event")
    plt.ylabel("eta censor")
    plt.show()

    # Calculate the hazard for each observation
    h_censor = 0.1 * np.exp(eta_censor)

    pct5 = np.percentile(h_censor, 5)
    pct10 = np.percentile(h_censor, 10)
    # Replace elements in the lowest 5th percentile with twice the amount of the 10th percentile
    h_censor[h_censor < pct5] = 2 * pct10

    # Generate censoring times (scale compared to event times??)
    # censoring_times = np.random.exponential(scale=np.maximum((5 - censoring_dependency * event_times),0))
    censoring_times = my_weibull(
        1.1, scale=scaling_censoring / h_censor, random_state=random_state + 1
    )

    # Create a DataFrame to hold the data
    data = pd.DataFrame(X, columns=[f"X{i + 1}" for i in range(p)])
    data["time"] = np.minimum(event_times, censoring_times)
    data["event"] = (event_times <= censoring_times).astype(bool)

    df_info = pd.DataFrame()
    df_info["event_time"] = event_times
    df_info["censoring_time"] = censoring_times
    df_info["h_events"] = h
    df_info["h_censor"] = h_censor

    y = data[["time", "event"]]

    # Return the data as a recarray
    return data, y.to_records(index=False), df_info


def generate_multivariate_data(
    N,
    p,
    q,
    interaction_pairs,
    noise_level,
    scaling_censoring=2000,
    event_names=["event1", "event2", "event3", "death"],
    censoring_event="death",
):

    df_events = pd.DataFrame()
    i = 0
    for event in event_names:
        i += 1
        df_comp, _, all_info = generate_synthetic_data(
            N,
            p,
            q,
            interaction_pairs,
            noise_level,
            scaling_censoring,
            random_state=i - 1,
        )
        all_info.rename(
            columns={
                "event_time": "time_" + str(event),
                "censoring_time": "censor_" + str(event),
            },
            inplace=True,
        )
        df_event = all_info[["time_" + str(event), "censor_" + str(event)]]
        df_events = pd.concat([df_events, df_event], axis=1)

    df_covars = df_comp[[col for col in df_comp.columns if "X" in col]]
    df_tot = pd.concat([df_covars, df_events], axis=1)

    df = copy.copy(df_tot)
    # censor_cols = [col for col in df_tot.columns if 'censor_' in col]

    for event in event_names:
        df["censor_" + str(event)] = df[
            ["censor_" + str(event), "time_" + str(censoring_event)]
        ].min(axis=1)

    return df


if __name__ == "__main__":

    import os

    root_folder = os.getcwd()

    SAVE_DF = False
    idx = 4

    if "augmentation" in root_folder:
        root_folder = os.path.dirname(
            root_folder
        )  # go up one and make sure we are in the Bellatrex folder
        os.chdir(root_folder)

    store_data = os.path.join(root_folder, "datasets", "original-data")

    simul_name = "my_simul_data_" + str(idx) + ".csv"

    simul_full_name = os.path.join(store_data, simul_name)

    # if not os.path.exists(simul_dir):
    #     os.makedirs(simul_dir)

    N = 1000  # Number of samples
    p = 10  # Number of covariates
    q = 4  # Number of covariates in common bertween event time and censoring time
    interaction_pairs = [
        (0, 1),
        (2, 3),
    ]  # Interactions between X1 and X2, and between X3 and X4
    # do not repeast indeces, always i < j
    noise_level = 0.1  # Amount of noise
    # 1: 0.01   # 2: 0.1    # 3: 0.3    # 4: 1
    # flip_p = 0.5

    df_surv, ys, all_info = generate_synthetic_data(N, p, q, interaction_pairs, noise_level)

    df_surv_multi = generate_multivariate_data(2 * N, 3 * p, 3 * q, interaction_pairs, noise_level)

    if SAVE_DF:
        df_surv.to_csv(simul_full_name, index=False)
