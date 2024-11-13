import pymc as pm
import pytensor.tensor as pt
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm

from surv_optimizer.models.MCModels import SurvivalModel


# Hierarchical Log-Normal AFT Model
class HierarchicalLogNormalAFTModel(SurvivalModel):
    def __init__(self, df, preds, coords, censored, event_times):
        super().__init__(df, preds, coords, censored, event_times)
        self.model_name = "HierarchicalLogNormal"

    def build_model(self):
        with pm.Model(coords=self.coords, check_bounds=False) as model:
            # Data input
            X_data = pm.Data(
                "X_data_obs", self.df[self.preds], dims=("obs_id", "preds")
            )
            y = self.event_times
            cens = self.censored

            # Map playlist IDs to indices
            playlist_codes, unique_playlists = pd.factorize(self.df["playlist_id"])
            self.df["playlist_idx"] = playlist_codes
            playlist_idx = self.df["playlist_idx"].values

            # Update coords with playlist indices
            model.add_coord("playlist_id", unique_playlists)
            playlist_idx_shared = pm.Data("playlist_idx", playlist_idx, dims="obs_id")

            # Priors
            beta = pm.Normal("beta", 0.0, 1.0, dims="preds")
            mu = pm.Normal("mu", 0.0, 1.0)
            sigma = pm.HalfNormal("sigma", 5.0)
            tau = pm.HalfNormal("tau", 5.0)
            # Playlist random effects
            u = pm.Normal("u", mu=0.0, sigma=tau, dims="playlist_id")

            # Linear predictor
            eta = pm.Deterministic(
                "eta",
                pm.math.dot(X_data, beta) + mu + u[playlist_idx_shared],
                dims="obs_id",
            )

            # Define the observed and censored likelihoods without specifying extra dimensions
            uncensored_idx = np.where(~cens)[0]
            censored_idx = np.where(cens)[0]

            # Likelihood for uncensored data
            y_obs = pm.Normal(
                "y_obs",
                mu=eta[uncensored_idx],
                sigma=sigma,
                observed=np.log(y[uncensored_idx]),
            )

            # Likelihood for censored data using potential
            z_cens = (np.log(y[censored_idx]) - eta[censored_idx]) / (
                sigma * pt.sqrt(2)
            )
            log_survival_function = pt.log(0.5 * pt.erfc(z_cens))
            y_cens = pm.Potential("y_cens", log_survival_function)

            self.model = model
        return model


# Recurrent Events Hierarchical Log-Normal AFT Model
class RecurrentHierarchicalLogNormalAFTModel(SurvivalModel):
    def __init__(self, df, preds, coords, censored, event_times):
        super().__init__(df, preds, coords, censored, event_times)
        self.model_name = "RecurrentHierarchicalLogNormal"
        # Map IDs to indices
        track_codes, unique_tracks = pd.factorize(df["isrc"])
        playlist_codes, unique_playlists = pd.factorize(df["playlist_id"])
        df["track_idx"] = track_codes
        df["playlist_idx"] = playlist_codes
        self.df = df
        self.n_tracks = len(unique_tracks)
        self.n_playlists = len(unique_playlists)

    def build_model(self):
        with pm.Model(coords=self.coords, check_bounds=False) as model:
            # Data input
            X_data = pm.Data(
                "X_data_obs", self.df[self.preds], dims=("obs_id", "preds")
            )
            y = self.event_times
            cens = self.censored
            track_idx = pm.Data("track_idx", self.df["track_idx"], dims="obs_id")
            playlist_idx = pm.Data(
                "playlist_idx", self.df["playlist_idx"], dims="obs_id"
            )

            # Update coords
            model.add_coord("isrc", range(self.n_tracks))
            model.add_coord("playlist_id", range(self.n_playlists))

            # Priors
            beta = pm.Normal("beta", 0.0, 1.0, dims="preds")
            mu = pm.Normal("mu", 0.0, 1.0)
            sigma = pm.HalfNormal("sigma", 5.0)
            tau_track = pm.HalfNormal("tau_track", 5.0)
            tau_playlist = pm.HalfNormal("tau_playlist", 5.0)
            # Random effects
            u = pm.Normal("u", mu=0.0, sigma=tau_track, dims="isrc")
            v = pm.Normal("v", mu=0.0, sigma=tau_playlist, dims="playlist_id")

            # Linear predictor
            eta = pm.Deterministic(
                "eta",
                pm.math.dot(X_data, beta) + mu + u[track_idx] + v[playlist_idx],
                dims="obs_id",
            )

            # Likelihood for uncensored data
            y_obs = pm.Normal(
                "y_obs",
                mu=eta[~cens],
                sigma=sigma,
                observed=np.log(y[~cens]),
            )

            # Likelihood for censored data using potential
            z_cens = (np.log(y[cens]) - eta[cens]) / (sigma * pt.sqrt(2))
            log_survival_function = pt.log(0.5 * pt.erfc(z_cens))
            y_cens = pm.Potential("y_cens", log_survival_function)

            self.model = model
        return model

    def evaluate_model(self, idata, times):
        """
        Evaluate the model using Concordance Index (C-Index) and Brier score.
        """
        # Predict survival probabilities
        survival_probs = self.predict_survival_function(
            idata, times
        )  # Shape: (n_individuals, n_times)

        # Observed survival times and events
        y_true = self.event_times
        event_observed = ~self.censored

        # Calculate expected survival times instead of median survival times
        expected_survival_times = self.calculate_expected_survival_times(
            survival_probs, times
        )

        # Compute C-index using expected survival times as risk scores
        ci = concordance_index(y_true, expected_survival_times, event_observed)

        # Compute Brier score at each time point
        brier_scores = []
        for j, t in enumerate(times):
            # Binary event occurrence up to time t
            y_binary = (y_true <= t) & event_observed
            # Predicted probability of event occurring by time t
            y_pred = 1 - survival_probs[:, j]
            # Brier score
            bs = brier_score_loss(y_binary, y_pred)
            brier_scores.append(bs)

        return ci, brier_scores  # Return CI and Brier scores for later plotting

    def plot_survival_curves(self, idata, times, track_playlist_pairs=None):
        """
        Plot survival curves for selected track-playlist pairs.
        """
        if track_playlist_pairs is None:
            track_playlist_pairs = [(0, 0)]  # Default to first track and first playlist

        survival_probs = self.predict_survival_function(idata, times)

        for track_idx, playlist_idx in track_playlist_pairs:
            # Identify rows for the given track-playlist pair
            mask = (self.df["track_idx"] == track_idx) & (
                self.df["playlist_idx"] == playlist_idx
            )
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            for idx in indices:
                # Use the same color for both the step and vertical line
                step_plot = plt.step(
                    times,
                    survival_probs[idx, :],
                    where="post",
                    label=f"Track {track_idx}, Playlist {playlist_idx}",
                )
                color = step_plot[0].get_color()  # Get the color of the step plot

                plt.axvline(
                    self.event_times[idx],
                    color=color,  # Use the same color as the step
                    linestyle="--",
                    label="Observed Event Time" if idx == indices[0] else "",
                )
        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
        plt.title("Predicted Survival Curves for Track-Playlist Pairs")
        plt.legend()
        plt.show()

    def calculate_expected_survival_times(self, survival_probs, times):
        """
        Calculate expected survival time for each individual by integrating the survival function.
        """
        expected_survival_times = np.zeros(survival_probs.shape[0])

        for i in range(survival_probs.shape[0]):
            # Survival probabilities for individual i
            surv_probs_i = survival_probs[i, :]
            # Integrate survival function over time to get expected survival time
            expected_survival_times[i] = np.trapz(surv_probs_i, times)

        return expected_survival_times

    def predict_survival_function(self, idata, times):
        """
        Predict the survival function at specified times based on the fitted model.
        """
        # Extract posterior samples from the inference data
        posterior_samples = idata.posterior.stack(samples=("chain", "draw"))

        # Get covariate values for prediction
        X = self.df[self.preds].values
        n_individuals = X.shape[0]
        n_times = len(times)
        n_samples = posterior_samples.dims["samples"]

        # Initialize array to store survival probabilities
        survival_probs = np.zeros((n_individuals, n_times))

        # Extract samples of beta, mu, u, v, and sigma
        beta_samples = posterior_samples["beta"].values
        mu_samples = posterior_samples["mu"].values
        u_samples = posterior_samples["u"].values
        v_samples = posterior_samples["v"].values
        sigma_samples = posterior_samples["sigma"].values

        # Compute linear predictor for each individual and sample
        for i in range(n_individuals):
            track_idx = self.df["track_idx"].iloc[i]
            playlist_idx = self.df["playlist_idx"].iloc[i]

            for j in range(n_times):
                t = times[j]
                eta_samples = (
                    mu_samples
                    + np.dot(beta_samples.T, X[i, :])
                    + u_samples[track_idx]
                    + v_samples[playlist_idx]
                )
                # Ensure eta_samples and sigma_samples do not lead to divide by zero or invalid operations
                eta_samples_safe = np.maximum(eta_samples, 1e-6)
                sigma_samples_safe = np.maximum(sigma_samples, 1e-6)
                # Log-normal survival function: S(t) = 1 - CDF(log(t) - eta / sigma)
                survival_probs[i, j] = np.mean(
                    1
                    - norm.cdf(
                        (np.log(np.maximum(t, 1e-5)) - eta_samples_safe)
                        / sigma_samples_safe
                    )
                )

        return survival_probs


class HawkesProcessModel(SurvivalModel):
    def __init__(self, df, preds, coords, event_times, playlist_ids):
        super().__init__(df, preds, coords, censored=None, event_times=event_times)
        self.model_name = "HawkesProcess"
        self.playlist_ids = playlist_ids  # List of playlist IDs for each event
        self.event_times = event_times  # Times at which events occur
        self.n_playlists = len(np.unique(playlist_ids))
        self.df = df  # DataFrame containing covariates and playlist IDs
        self.risk_matrices, self.event_matrices = get_risk_event_matrices(df)

    def build_model(self):
        with pm.Model(coords=self.coords) as model:
            # Data input
            X_data = pm.Data(
                "X_data_obs", self.df[self.preds], dims=("obs_id", "preds")
            )
            event_times = self.event_times
            playlist_ids = self.playlist_ids

            # Map playlist IDs to indices
            playlist_codes, unique_playlists = pd.factorize(playlist_ids)
            self.df["playlist_idx"] = playlist_codes
            playlist_idx = self.df["playlist_idx"].values

            # Update coords with playlist indices
            model.add_coord("playlist_id", unique_playlists)
            playlist_idx_shared = pm.Data("playlist_idx", playlist_idx, dims="obs_id")

            # Priors
            mu = pm.HalfNormal(
                "mu", sigma=1.0, dims="playlist_id"
            )  # Baseline intensity
            alpha = pm.HalfNormal("alpha", sigma=1.0)  # Excitation coefficient
            omega = pm.HalfNormal("omega", sigma=1.0)  # Decay rate
            beta = pm.Normal("beta", 0.0, 1.0, dims="preds")  # Covariate effects

            # Initialize the intensity function
            lambda_ = mu[playlist_idx_shared] + pm.math.dot(X_data, beta)

            # Add excitation from past events
            def excitation(t, history_times, omega):
                time_diffs = t - history_times
                excitation = alpha * pm.math.exp(-omega * time_diffs)
                excitation = pm.math.sum(excitation * (time_diffs > 0))
                return excitation

            # Build the intensity function incorporating excitation
            lambda_events = []
            for i in range(len(event_times)):
                t_i = event_times[i]
                idx_i = playlist_idx[i]
                history_mask = (event_times < t_i) & (playlist_idx == idx_i)
                history_times = event_times[history_mask]

                # Compute excitation
                excitation_i = excitation(t_i, history_times, omega)
                lambda_i = lambda_[i] + excitation_i

                # Ensure lambda is positive to prevent log(0)
                lambda_i = pm.math.maximum(lambda_i, 1e-5)
                lambda_events.append(lambda_i)

            lambda_events = pm.math.stack(lambda_events)

            # Likelihood
            # Since events occur at the event times, the likelihood is given by the intensity at those times
            pm.Potential("log_likelihood", pm.math.sum(pm.math.log(lambda_events)))

            # Add the compensator (integral of the intensity over the observation period)
            # For simplicity, assume a fixed observation window [0, T_max]
            T_max = event_times.max()
            compensator = pm.math.sum(mu[playlist_idx_shared]) * T_max
            # Add excitation terms to the compensator
            excitation_integral = (alpha / omega) * len(event_times)
            compensator += excitation_integral
            pm.Potential("compensator", -compensator)

            self.model = model
        return model


def get_risk_event_matrices(
    df,
    duration_col="days_on_playlist",
    event_col="event_observed",
    playlist_col="playlist_id",
):
    """
    Get risk and event matrices for multiple playlists.

    Parameters:
    - df: pandas DataFrame containing event data.
    - duration_col: Column indicating time on playlist.
    - event_col: Column indicating if event occurred (True/False).
    - playlist_col: Column representing playlist identifier.

    Returns:
    - risk_matrices: Dictionary of risk matrices for each playlist.
    - event_matrices: Dictionary of event matrices for each playlist.
    """

    # Factorize playlist IDs to create unique numerical index for each playlist
    playlist_codes, unique_playlists = pd.factorize(df[playlist_col])
    df["playlist_idx"] = playlist_codes

    # Initialize dictionaries to store matrices for each playlist
    risk_matrices = {}
    event_matrices = {}

    # Loop over each playlist to create event and risk matrices
    for playlist_idx, playlist_id in enumerate(unique_playlists):
        playlist_data = df[df["playlist_idx"] == playlist_idx]
        n_playlist_tracks = playlist_data.shape[0]
        max_duration = playlist_data[duration_col].max()

        # Initialize risk and event matrices
        event_matrix = np.zeros((n_playlist_tracks, max_duration))
        risk_matrix = np.zeros((n_playlist_tracks, max_duration))

        # Populate matrices
        for i, (_, row) in enumerate(playlist_data.iterrows()):
            duration = int(row[duration_col])  # Duration in days/weeks
            event_occurred = row[event_col]

            # Set risk up to and including the event occurrence time
            risk_matrix[i, :duration] = 1

            # Set the event occurrence
            if event_occurred:
                event_matrix[i, duration - 1] = (
                    1  # Event occurs at the end of the duration
                )

        # Convert to DataFrame for easier inspection
        event_matrices[playlist_id] = pd.DataFrame(
            event_matrix, columns=[f"Time_{i}" for i in range(max_duration)]
        )
        risk_matrices[playlist_id] = pd.DataFrame(
            risk_matrix, columns=[f"Time_{i}" for i in range(max_duration)]
        )

    return risk_matrices, event_matrices


def plot_brier_score_over_time(times, brier_scores):
    """
    Plot Brier score over time.

    Parameters:
    - times: Array of time points.
    - brier_scores: Brier scores calculated at each time point.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(times, brier_scores, marker="o", linestyle="-", color="b")
    plt.xlabel("Time")
    plt.ylabel("Brier Score")
    plt.title("Brier Score over Time")
    plt.grid(True)
    plt.show()


def plot_qq(idata, param_name="mu"):
    """
    Generate a QQ plot for the parameter of interest.

    Parameters:
    - idata: InferenceData object containing posterior samples.
    - param_name: The parameter to create the QQ plot for (e.g., "mu").
    """
    posterior_samples = idata.posterior.stack(samples=("chain", "draw"))
    parameter_samples = posterior_samples[param_name].values.flatten()

    # Generate QQ plot
    stats.probplot(parameter_samples, dist="norm", plot=plt)
    plt.title(f"QQ Plot for Parameter: {param_name}")
    plt.grid(True)
    plt.show()
