import pymc as pm
import pytensor.tensor as pt
import numpy as np
import pandas as pd
from patsy.highlevel import dmatrix, build_design_matrices
from lifelines.utils import concordance_index
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import weibull_min, logistic, lognorm, norm
from scipy.interpolate import BSpline
import warnings
from scipy.special import gammainc


warnings.filterwarnings(
    "ignore",
    message="The effect of Potentials on other parameters is ignored during prior predictive sampling.*",
)
warnings.filterwarnings(
    "ignore",
    message=" The effect of Potentials on other parameters is ignored during prior predictive sampling.*",
)


# Common functions for survival models
def logistic_sf(y, mu, s):
    return 1.0 - pm.math.sigmoid((y - mu) / s)


def weibull_lccdf(x, alpha, beta):
    """Log complementary cdf of Weibull distribution."""
    return -((x / beta) ** alpha)


# Base class for survival models
class SurvivalModel:
    def __init__(self, df, preds, coords, censored, event_times, interval_bounds=None):
        self.df = df
        self.preds = preds
        self.coords = coords
        self.censored = censored
        self.event_times = event_times
        self.model_type = None  # To be set in subclasses
        self.interval_bounds = (
            interval_bounds  # Optional, used in Cox and Piecewise models
        )
        self.model = None  # This will store the PyMC model after it's built
        self.model_name = ""  # To be set in each subclass

    def build_model(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def sample(self, model):
        if self.model is None:
            raise RuntimeError("Build the model first by calling build_model().")
        idata = pm.sample_prior_predictive(model=model)
        idata.extend(
            pm.sample(
                model=model,
                init="jitter+adapt_diag",
                target_accept=0.95,
                random_seed=100,
                idata_kwargs={"log_likelihood": True},
            )
        )
        idata.extend(pm.sample_posterior_predictive(idata, model=model))
        return idata

    def predict_survival_function(self, idata, times):
        """Predict survival function at specified times."""
        # Extract posterior samples
        posterior_samples = idata.posterior.stack(samples=("chain", "draw"))

        # Get covariate values
        X = self.df[self.preds].values  # Shape: (n_individuals, n_covariates)
        n_individuals = X.shape[0]
        n_times = len(times)
        n_samples = posterior_samples.dims["samples"]

        # Initialize array to store survival probabilities
        survival_probs = np.zeros((n_individuals, n_times))

        # Extract samples of beta
        beta_samples = posterior_samples[
            "beta"
        ].values  # Shape: (n_covariates, n_samples)

        # For models that have 'mu', 's', or other parameters, extract them if they exist
        mu_samples = posterior_samples.get("mu", None)
        if mu_samples is not None:
            mu_samples = mu_samples.values  # Shape: (n_samples,)

        sigma_samples = posterior_samples.get("sigma", None)
        if sigma_samples is not None:
            sigma_samples = sigma_samples.values  # Shape: (n_samples,)

        gamma_samples = posterior_samples.get("gamma", None)
        if gamma_samples is not None:
            gamma_samples = gamma_samples.values  # Shape: (n_splines, n_samples)

        lambda0_samples = posterior_samples.get("lambda0", None)
        if lambda0_samples is not None:
            lambda0_samples = lambda0_samples.values  # Shape depends on the model

        kappa_samples = posterior_samples.get("kappa", None)
        if kappa_samples is not None:
            kappa_samples = kappa_samples.values  # Shape: (n_samples,)

        tau_samples = posterior_samples.get("tau", None)
        if tau_samples is not None:
            tau_samples = tau_samples.values  # Shape: (n_samples,)

        u_samples = posterior_samples.get("u", None)
        if u_samples is not None:
            u_samples = u_samples.values  # Shape: (n_playlists, n_samples)

        # Compute linear predictor
        # For models without 'mu', we can assume mu = 0
        if mu_samples is None:
            mu_samples = np.zeros(beta_samples.shape[1])  # Shape: (n_samples,)

        # For each individual, compute the survival probabilities at each time point
        for i in range(n_individuals):
            # Get covariate values for individual i
            x_i = X[i, :]  # Shape: (n_covariates,)

            # Compute eta for all samples
            eta_samples = (
                np.dot(beta_samples.T, x_i) + mu_samples
            )  # Shape: (n_samples,)

            # Initialize survival probabilities for individual i
            surv_probs_i = np.ones(n_times)

            if self.model_name == "Weibull":
                # s_samples required
                if sigma_samples is None:
                    raise ValueError("s_samples is required for Weibull model")

                # Compute reg = exp(-(eta) / s)
                reg_samples = np.exp(
                    -(eta_samples) / sigma_samples
                )  # Shape: (n_samples,)

                # For each time point
                for j, t in enumerate(times):
                    # Compute survival probability at time t
                    surv_samples = np.exp(
                        -((t / reg_samples) ** sigma_samples)
                    )  # Shape: (n_samples,)
                    surv_probs_i[j] = surv_samples.mean()

            elif self.model_name == "GeneralizedGamma":
                # sigma_samples and kappa_samples required
                if sigma_samples is None or kappa_samples is None:
                    raise ValueError(
                        "sigma_samples and kappa_samples are required for GeneralizedGamma model"
                    )

                # For each time point
                for j, t in enumerate(times):
                    # Compute y for all samples
                    y = (np.log(t) - eta_samples) / sigma_samples  # Shape: (n_samples,)

                    a = 1 / (kappa_samples**2)  # Shape: (n_samples,)
                    c = kappa_samples  # Shape: (n_samples,)
                    z = a * np.exp(c * y)  # Shape: (n_samples,)

                    # Compute survival probability
                    s = 1 - gammainc(a, z)  # Shape: (n_samples,)
                    surv_samples = s  # Shape: (n_samples,)

                    surv_probs_i[j] = surv_samples.mean()

            elif self.model_name == "LogLogistic":
                # s_samples required
                if sigma_samples is None:
                    raise ValueError("s_samples is required for LogLogistic model")

                for j, t in enumerate(times):
                    t = max(t, 1e-10)  # Avoid log(0)
                    z = (np.log(t) - eta_samples) / sigma_samples  # Shape: (n_samples,)
                    surv_samples = 1 / (1 + np.exp(z))  # Shape: (n_samples,)
                    surv_probs_i[j] = surv_samples.mean()

            elif self.model_name == "LogNormal":
                # sigma_samples required
                sigma_samples = posterior_samples["sigma"].values  # Shape: (n_samples,)

                for j, t in enumerate(times):
                    z = (np.log(t) - eta_samples) / sigma_samples  # Shape: (n_samples,)
                    surv_samples = 1 - norm.cdf(z)  # Shape: (n_samples,)
                    surv_probs_i[j] = surv_samples.mean()

            elif self.model_name == "HierarchicalLogNormal":
                if sigma_samples is None or u_samples is None:
                    raise ValueError(
                        "sigma_samples and u_samples are required for HierarchicalLogNormal model"
                    )

                for i in range(n_individuals):
                    # Get covariate values for individual i
                    x_i = X[i, :]  # Shape: (n_covariates,)

                    # Playlist index for individual i
                    playlist_idx_i = self.df["playlist_idx"].iloc[i]

                    # Compute eta for all samples
                    eta_samples = (
                        np.dot(beta_samples.T, x_i)
                        + mu_samples
                        + u_samples[playlist_idx_i, :]
                    )  # Shape: (n_samples,)

                    # Initialize survival probabilities for individual i
                    surv_probs_i = np.ones(n_times)

                    # For each time point
                    for j, t in enumerate(times):
                        z = (
                            np.log(t) - eta_samples
                        ) / sigma_samples  # Shape: (n_samples,)
                        surv_samples = 1 - norm.cdf(z)  # Shape: (n_samples,)
                        surv_probs_i[j] = surv_samples.mean()

                    # Store survival probabilities for individual i
                    survival_probs[i, :] = surv_probs_i
            elif self.model_name == "CoxPH":
                # lambda0_samples required
                if lambda0_samples is None:
                    raise ValueError("lambda0_samples is required for CoxPH model")

                # Need to compute the cumulative baseline hazard
                # First, compute the baseline hazard for each interval
                # Then, interpolate to get cumulative hazard at each time

                # For simplicity, let's assume that lambda0_samples is shape (n_intervals, n_samples)
                # We'll need to compute the cumulative hazard for each time point
                # This is a complex task; I'll provide a simplified version

                # For each time point
                for j, t in enumerate(times):
                    # Compute baseline cumulative hazard at time t
                    baseline_cumhaz_samples = self.compute_baseline_cumhaz(
                        t, lambda0_samples
                    )

                    # Compute cumulative hazard for individual i
                    cumhaz_samples = (
                        np.exp(eta_samples) * baseline_cumhaz_samples
                    )  # Shape: (n_samples,)

                    # Compute survival probability
                    surv_samples = np.exp(-cumhaz_samples)  # Shape: (n_samples,)
                    surv_probs_i[j] = surv_samples.mean()

            elif self.model_name == "Piecewise":
                # Similar to CoxPH but the hazard is constant within intervals
                if lambda0_samples is None:
                    raise ValueError("lambda0_samples is required for Piecewise model")

                # For each time point
                for j, t in enumerate(times):
                    # Compute cumulative hazard up to time t
                    cumhaz_samples = self.compute_piecewise_cumhaz(
                        t, lambda0_samples, eta_samples
                    )

                    # Compute survival probability
                    surv_samples = np.exp(-cumhaz_samples)  # Shape: (n_samples,)
                    surv_probs_i[j] = surv_samples.mean()

            elif self.model_name == "Flexible":
                # gamma_samples required
                if gamma_samples is None:
                    raise ValueError("gamma_samples is required for Flexible model")
                if np.any(times <= 0):
                    raise ValueError(
                        "Time values must be positive for the spline basis transformation."
                    )
                # Create spline basis for all prediction times
                df_spline_times = pd.DataFrame({"log_time": np.log(times)})
                # Ensure that interior and boundary knots are passed to dmatrix consistently
                spline_basis_times = dmatrix(
                    f"bs(log_time, knots={self.interior_knots.tolist()}, degree=3, include_intercept=True, "
                    f"lower_bound={self.boundary_knots[0]}, upper_bound={self.boundary_knots[1]})",
                    df_spline_times,
                    return_type="dataframe",
                )
                spline_basis_times = np.asarray(
                    spline_basis_times
                )  # Shape: (n_times, n_splines)
                # For each time point
                # For each time point
                for j in range(n_times):
                    # Get the spline basis for time j
                    spline_basis_t = spline_basis_times[j, :]  # Shape: (n_splines,)

                    # Compute log cumulative hazard
                    log_H_samples = eta_samples + np.dot(
                        gamma_samples.T, spline_basis_t
                    )  # Shape: (n_samples,)

                    # Compute cumulative hazard
                    H_samples = np.exp(log_H_samples)  # Shape: (n_samples,)

                    # Compute survival probability
                    surv_samples = np.exp(-H_samples)  # Shape: (n_samples,)

                    # Average over samples
                    survival_probs[i, j] = surv_samples.mean()

            else:
                raise ValueError(f"Unknown model name: {self.model_name}")

            # Store survival probabilities for individual i
            survival_probs[i, :] = surv_probs_i

        return survival_probs  # Shape: (n_individuals, n_times)

    def compute_baseline_cumhaz(self, t, lambda0_samples):
        """
        Compute the baseline cumulative hazard at time t for CoxPH model.
        """
        # Since the baseline hazard is defined over intervals, we need to sum the hazards up to time t.
        # Assume that lambda0_samples is of shape (n_intervals, n_samples)

        # Get interval boundaries
        t_min = self.interval_bounds[:-1]
        t_max = self.interval_bounds[1:]
        n_intervals = len(t_min)

        # Determine which intervals are up to time t
        intervals_up_to_t = t_max <= t

        # For intervals fully before time t, sum the hazards
        cumhaz_samples = np.sum(
            lambda0_samples[intervals_up_to_t, :], axis=0
        )  # Shape: (n_samples,)

        # For the interval containing time t, add the proportion of hazard
        idx = np.searchsorted(t_max, t, side="right")
        if idx < n_intervals:
            # Time t is within interval idx
            delta_t = t - t_min[idx]
            total_interval = t_max[idx] - t_min[idx]
            prop = delta_t / total_interval
            cumhaz_samples += lambda0_samples[idx, :] * prop

        return cumhaz_samples  # Shape: (n_samples,)

    def compute_piecewise_cumhaz(self, t, lambda0_samples, eta_samples):
        """
        Compute the cumulative hazard at time t for Piecewise model.
        """
        # Similar to compute_baseline_cumhaz but include the linear predictor
        # lambda0_samples: Shape (n_intervals, n_samples)
        # eta_samples: Shape (n_samples,)
        n_samples = lambda0_samples.shape[1]  # Assume shape is (n_intervals, n_samples)

        # Get interval boundaries
        t_min = self.interval_bounds[:-1]
        t_max = self.interval_bounds[1:]
        n_intervals = len(t_min)

        # Determine exposure time in each interval up to time t
        exposure = np.zeros((n_samples,))
        for idx in range(n_intervals):
            start = t_min[idx]
            end = t_max[idx]
            if t > start:
                delta_t = min(t, end) - start
                delta_t = max(delta_t, 0)
                exposure += delta_t * lambda0_samples[idx, :]
            else:
                break

        # Multiply by exp(eta)
        cumhaz_samples = np.exp(eta_samples) * exposure  # Shape: (n_samples,)

        return cumhaz_samples  # Shape: (n_samples,)

    def calculate_expected_survival_times(self, survival_probs, times):
        """Calculate expected survival time for each individual by integrating the survival function."""
        expected_survival_times = np.zeros(survival_probs.shape[0])

        for i in range(survival_probs.shape[0]):
            # Survival probabilities for individual i
            surv_probs_i = survival_probs[i, :]
            # Integrate survival function over time to get expected survival time
            expected_survival_times[i] = np.trapz(surv_probs_i, times)

        return expected_survival_times

    def evaluate_model(self, idata, times):
        """Evaluate the model using Brier score and C-index."""
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
        # print(f"Concordance Index: {ci}")

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

    def plot_survival_curves(self, idata, times, individual_indices=None):
        """Plot survival curves for selected individuals."""
        if individual_indices is None:
            individual_indices = np.arange(len(self.df))

        survival_probs = self.predict_survival_function(idata, times)

        for idx in individual_indices:
            plt.step(
                times, survival_probs[idx, :], where="post", label=f"Individual {idx}"
            )
            plt.axvline(
                self.event_times[idx],
                color="r",
                linestyle="--",
                label="Observed Event Time" if idx == individual_indices[0] else "",
            )
        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
        plt.title("Predicted Survival Curves")
        plt.legend()
        plt.show()


# Weibull AFT Model Assumption: The logarithm of the survival time follows a Weibull distribution.
class WeibullAFTModel(SurvivalModel):
    def __init__(self, df, preds, coords, censored, event_times):
        super().__init__(df, preds, coords, censored, event_times)
        self.model_name = "Weibull"

    def build_model(self):
        with pm.Model(coords=self.coords, check_bounds=False) as model:
            # Data input
            X_data = pm.Data(
                "X_data_obs", self.df[self.preds], dims=("obs_id", "preds")
            )
            y = self.event_times
            cens = self.censored

            # Priors
            beta = pm.Normal("beta", 0.0, 1.0, dims="preds")
            mu = pm.Normal("mu", 0.0, 1.0)
            sigma = pm.HalfNormal("sigma", 5.0)

            # Linear predictor
            eta = pm.Deterministic("eta", pm.math.dot(X_data, beta) + mu, dims="obs_id")
            reg = pm.Deterministic("reg", pm.math.exp(-(eta) / sigma), dims="obs_id")

            # Likelihood for uncensored data
            y_obs = pm.Weibull(
                "y_obs", alpha=sigma, beta=reg, observed=y[~cens], dims="obs_uncens"
            )

            # Likelihood for censored data using potential
            y_cens = pm.Potential(
                "y_cens", weibull_lccdf(y[cens], alpha=sigma, beta=reg[cens])
            )

            self.model = model
        return model


# Log-Logistic Model Assumption: The logarithm of the survival time follows a Logistic distribution.
class LogLogisticAFTModel(SurvivalModel):
    def __init__(self, df, preds, coords, censored, event_times):
        super().__init__(df, preds, coords, censored, event_times)
        self.model_name = "LogLogistic"

    def build_model(self):
        with pm.Model(coords=self.coords, check_bounds=False) as model:
            # Data input
            X_data = pm.Data(
                "X_data_obs", self.df[self.preds], dims=("obs_id", "preds")
            )
            y = self.event_times
            cens = self.censored

            # Priors
            beta = pm.Normal("beta", 0.0, 1.0, dims="preds")
            mu = pm.Normal("mu", 2.0, 1.0)
            sigma = pm.HalfNormal("sigma", 5.0)

            # Linear predictor
            eta = pm.Deterministic("eta", pm.math.dot(X_data, beta) + mu, dims="obs_id")
            # Likelihood for uncensored data
            y_obs = pm.Logistic(
                "y_obs",
                mu=eta[~cens],
                s=sigma,
                observed=np.log(y[~cens]),
                dims="obs_uncens",
            )

            # Likelihood for censored data using potential
            log_survival_function = pm.math.log(
                1 - pm.math.sigmoid((np.log(y[cens]) - eta[cens]) / sigma)
            )
            y_cens = pm.Potential("y_cens", log_survival_function)

            self.model = model
        return model


# Log-Normal AFT Model Log-Normal Model: Assumes the logarithm of survival time is normally distributed.
class LogNormalAFTModel(SurvivalModel):
    def __init__(self, df, preds, coords, censored, event_times):
        super().__init__(df, preds, coords, censored, event_times)
        self.model_name = "LogNormal"

    def build_model(self):
        with pm.Model(coords=self.coords, check_bounds=False) as model:
            # Data input
            X_data = pm.Data(
                "X_data_obs", self.df[self.preds], dims=("obs_id", "preds")
            )
            y = self.event_times
            cens = self.censored

            # Priors
            beta = pm.Normal("beta", 0.0, 1.0, dims="preds")
            mu = pm.Normal("mu", 0.0, 1.0)
            sigma = pm.HalfNormal("sigma", 5.0)

            # Linear predictor
            eta = pm.Deterministic("eta", pm.math.dot(X_data, beta) + mu, dims="obs_id")

            # Likelihood for uncensored data
            y_obs = pm.Normal(
                "y_obs",
                mu=eta[~cens],
                sigma=sigma,
                observed=np.log(y[~cens]),
                dims="obs_uncens",
            )

            # Likelihood for censored data using potential
            z_cens = (np.log(y[cens]) - eta[cens]) / (sigma * pt.sqrt(2))
            log_survival_function = pt.log(0.5 * pt.erfc(z_cens))
            y_cens = pm.Potential("y_cens", log_survival_function)

            self.model = model
        return model


# Cox Proportional Hazards Model (using Piecewise Exponential Approximation)
class CoxPHModel(SurvivalModel):
    def __init__(self, df, preds, coords, censored, event_times, interval_bounds):
        super().__init__(df, preds, coords, censored, event_times)
        self.model_name = "CoxPH"
        self.interval_bounds = interval_bounds  # Include interval bounds

    def build_model(self):
        if self.interval_bounds is None:
            raise ValueError("interval_bounds must be specified for CoxPHModel.")
        with pm.Model(coords=self.coords) as model:
            # Data input
            X_data = pm.Data(
                "X_data_obs", self.df[self.preds], dims=("obs_id", "preds")
            )
            cens = self.censored
            event_times = self.event_times
            intervals = self.coords["intervals"]  # Time intervals

            # Time interval boundaries
            t_min = self.interval_bounds[:-1]
            t_max = self.interval_bounds[1:]
            n_intervals = len(t_min)

            # Determine which interval each event time falls into
            interval_idx = np.searchsorted(t_max, event_times, side="right") - 1
            interval_idx = np.clip(interval_idx, 0, n_intervals - 1)

            # Priors
            beta = pm.Normal("beta", 0.0, 1.0, dims="preds")
            lambda0 = pm.Exponential("lambda0", lam=1.0, dims="intervals")

            # Linear predictor
            linpred = pm.Deterministic(
                "linpred", pm.math.dot(X_data, beta), dims="obs_id"
            )

            # Hazard for each interval
            hazard = pm.Deterministic(
                "hazard",
                pm.math.exp(linpred[:, None]) * lambda0[None, :],
                dims=("obs_id", "intervals"),
            )

            # Compute cumulative hazard
            cum_hazard = hazard.cumsum(axis=1)

            # Likelihood for uncensored data
            loglik_obs = -cum_hazard[
                np.arange(len(event_times)), interval_idx
            ] + pm.math.log(hazard[np.arange(len(event_times)), interval_idx])
            y_obs = pm.Potential("y_obs", loglik_obs[~cens])

            # Likelihood for censored data
            loglik_cens = -cum_hazard[np.arange(len(event_times)), interval_idx]
            y_cens = pm.Potential("y_cens", loglik_cens[cens])

            self.model = model
        return model


# Piecewise Exponential Model assumes a constant hazard within each interval.
class PiecewiseExponentialModel(SurvivalModel):
    def __init__(self, df, preds, coords, censored, event_times, interval_bounds):
        super().__init__(df, preds, coords, censored, event_times)
        self.model_name = "Piecewise"
        self.interval_bounds = interval_bounds  # Include interval bounds

    def build_model(self):
        if self.interval_bounds is None:
            raise ValueError(
                "interval_bounds must be specified for PiecewiseExponentialModel."
            )

        with pm.Model(coords=self.coords) as model:
            # Data input
            X_data = pm.Data(
                "X_data_obs", self.df[self.preds], dims=("obs_id", "preds")
            )
            cens = self.censored
            event_times = self.event_times
            intervals = self.coords["intervals"]  # Time intervals

            # Time interval boundaries
            t_min = self.interval_bounds[:-1]
            t_max = self.interval_bounds[1:]
            n_intervals = len(t_min)

            # Determine exposure time in each interval
            exposure = np.zeros((len(event_times), n_intervals))
            for i in range(n_intervals):
                exposure[:, i] = np.clip(event_times, t_min[i], t_max[i]) - t_min[i]
                exposure[:, i] = np.maximum(exposure[:, i], 0)
            exposure += 1e-6
            # Priors
            beta = pm.Normal("beta", 0.0, 1.0, dims="preds")
            lambda0 = pm.Exponential("lambda0", 1.0, dims="intervals")

            # Linear predictor
            linpred = pm.math.dot(X_data, beta)

            # Expected events
            mu = pm.math.exp(linpred[:, None]) * lambda0[None, :] * exposure

            # Observed events
            observed_events = np.zeros((len(event_times), n_intervals))
            for i, t in enumerate(event_times):
                idx = np.searchsorted(t_max, t, side="right") - 1
                if not cens[i]:
                    observed_events[i, idx] = 1
            # Add a small constant to avoid -inf log probability
            observed_events += 1e-6
            # Likelihood
            obs = pm.Poisson(
                "obs", mu, observed=observed_events, dims=("obs_id", "intervals")
            )

            self.model = model
        return model


# Flexible Parametric Survival Model
class FlexibleParametricModel(SurvivalModel):
    def __init__(self, df, preds, coords, censored, event_times):
        super().__init__(df, preds, coords, censored, event_times)
        self.model_name = "Flexible"

    def build_model(self):
        min_time = 0  # Start time
        max_time = (
            np.max(self.event_times) * 1.1
        )  # Extend beyond the maximum observed time

        with pm.Model(coords=self.coords) as model:
            # Data input
            X_data = pm.Data(
                "X_data_obs", self.df[self.preds], dims=("obs_id", "preds")
            )
            y = self.event_times
            cens = self.censored

            # Define the range of log times
            log_time = np.log(y)
            min_log_time = np.log(min_time + 1e-5)  # Avoid log(0)
            max_log_time = np.log(max_time)

            # Determine knot positions using quantiles
            num_knots = 3  # Adjust as needed
            interior_knots = np.quantile(
                log_time, np.linspace(0, 1, num_knots + 2)[1:-1]
            )

            # Define boundary knots
            boundary_knots = [min_log_time, max_log_time]

            # Create spline basis for log(time)
            df_spline = pd.DataFrame({"log_time": log_time})

            # Build the spline basis with specified knots and save design info
            spline_basis = dmatrix(
                f"bs(log_time, knots={interior_knots.tolist()}, degree=3, include_intercept=False)",
                df_spline,
                return_type="dataframe",
            )

            # Store design info for prediction
            self.spline_design_info = spline_basis.design_info

            # Convert spline_basis to NumPy array if needed
            spline_basis = np.asarray(spline_basis)
            n_splines = spline_basis.shape[1]

            # Store the knots for later use
            self.interior_knots = interior_knots
            self.boundary_knots = boundary_knots

            # Priors
            beta = pm.Normal("beta", 0.0, 1.0, dims="preds")
            gamma = pm.Normal("gamma", 0.0, 1.0, shape=n_splines)

            # Linear predictor for log cumulative hazard
            log_H = pm.math.dot(X_data, beta) + spline_basis @ gamma  # Shape: (n_obs,)

            # Cumulative hazard
            H = pm.Deterministic("H", pm.math.exp(log_H), dims="obs_id")

            # Survival function
            S = pm.Deterministic("S", pm.math.exp(-H), dims="obs_id")

            # Log-likelihood for uncensored data
            # For uncensored data, we need the hazard function h(t)
            # We approximate the hazard function as:
            # h(t) = H'(t) = dH/dt
            # Since H(t) = exp(log_H(t)), we have:
            # h(t) = H(t) * d(log_H)/dt

            # Compute derivative of log_H with respect to log_time
            # Since we use splines in log_time, the derivative with respect to log_time is:
            dlogH_dlogt = spline_basis @ gamma  # Shape: (n_obs,)
            # Since log_time = log(t), d(log_H)/dt = (1 / t) * d(log_H)/d(log_time)
            dlogH_dt = dlogH_dlogt / y

            # Hazard function h(t)
            h = H * dlogH_dt

            # Log-likelihood for uncensored data
            loglik_obs = pm.math.log(h[~cens]) - H[~cens]

            # Log-likelihood for censored data
            loglik_cens = -H[cens]

            # Total log-likelihood
            total_loglik = pm.math.sum(loglik_obs) + pm.math.sum(loglik_cens)

            # Likelihood
            pm.Potential("likelihood", total_loglik)

            # Ensure log_likelihood is stored
            pm.Deterministic("log_likelihood", loglik_obs, dims="obs_uncens")

            self.model = model
            self.n_splines = n_splines
            self.spline_basis = spline_basis  # Store for use in prediction
        return model


class GeneralizedGammaAFTModel(SurvivalModel):
    def build_model(self):
        self.model_name = "GeneralizedGamma"
        with pm.Model(coords=self.coords, check_bounds=False) as model:
            # Data input
            X_data = pm.Data(
                "X_data_obs", self.df[self.preds], dims=("obs_id", "preds")
            )
            y = self.event_times
            cens = self.censored

            # Priors
            beta = pm.Normal("beta", 0.0, 1.0, dims="preds")
            mu = pm.Normal("mu", 0.0, 1.0)
            sigma = pm.HalfNormal("sigma", 1.0)
            kappa = pm.Normal("kappa", 0.0, 1.0)  # Shape parameter

            # Linear predictor
            eta = pm.Deterministic("eta", pm.math.dot(X_data, beta) + mu, dims="obs_id")

            # Log-likelihood function for uncensored data
            def logp_uncens(t, eta, sigma, kappa):
                y = (pt.log(t) - eta) / sigma
                a = 1 / (kappa**2)
                c = kappa
                z = a * pt.exp(c * y)
                logpdf = (
                    pt.log(pt.abs(c))
                    - pt.gammaln(a)
                    - pt.log(sigma)
                    + a * pt.log(a)
                    + a * c * y
                    - z
                )
                return logpdf

            # Log-survival function for censored data
            def logsurv_cens(t, eta, sigma, kappa):
                y = (pt.log(t) - eta) / sigma
                a = 1 / (kappa**2)
                c = kappa
                z = a * pt.exp(c * y)
                s = 1 - pt.gammainc(a, z)
                logsurv = pt.log(s)
                return logsurv

            # Likelihood for uncensored data
            y_obs = pm.Potential(
                "y_obs", logp_uncens(y[~cens], eta[~cens], sigma, kappa)
            )

            # Likelihood for censored data
            y_cens = pm.Potential(
                "y_cens", logsurv_cens(y[cens], eta[cens], sigma, kappa)
            )

            self.model = model
        return model
