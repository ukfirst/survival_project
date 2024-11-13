import statsmodels.formula.api as smf
from lifelines import WeibullAFTFitter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd

from db_utils.multi_dataset_model import MusicService


class SurvivalModelTester:
    def __init__(
        self,
        df,
        duration_col="days_on_playlist",
        event_col="event_observed",
        significance_level=0.07,
        interval_length=7,
    ):
        """
        Initialize the SurvivalModelTester with data and parameters.

        Parameters:
        - df: pandas DataFrame with the data.
        - predictors: List of predictor variables.
        - duration_col: Name of the duration column (time to event).
        - event_col: Name of the event indicator column (event occurred or censored).
        - significance_level: Significance level for determining important predictors.
        """
        self.df = df.copy()
        self.n_tracks = self.df.shape[0]
        self.max_days = df[duration_col].max()
        self.interval_length = interval_length
        self.predictors = None
        self.duration_col = duration_col
        self.event_col = event_col
        self.days_since_release = "days_since_release"
        self.significance_level = significance_level
        self.cox_significant = []
        self.aft_significant = []
        self.combined_significant = []
        self.is_preprocessed = False  # Flag to check if data has been preprocessed
        self.intervals = None

    def preprocess_data(self, minmax=False):
        """Preprocess the dataset by standardizing numeric columns and adjusting interval-based columns."""
        if self.is_preprocessed:
            print("Data has already been preprocessed.")
            return
        """Preprocess the dataset by standardizing numeric columns and adjusting interval-based columns."""
        if self.predictors is None:
            # Identify columns to standardize (all numeric except duration and event columns)
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.predictors = [
                col
                for col in numeric_cols
                if col not in [self.duration_col, self.event_col]
            ]

        # Apply MinMaxScaler to the selected columns
        if not minmax:
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler((0.01, 0.99))
        self.df[self.predictors] = scaler.fit_transform(self.df[self.predictors])

        # Convert 'days_on_playlist' and 'days_since_release' to weekly intervals

        self.df[self.duration_col] = (
            np.floor((self.df[self.duration_col] - 0.01) / self.interval_length).astype(
                int
            )
            + 1
        )
        # if "days_since_release" in self.df.columns:
        #     self.df["days_since_release"] = (
        #         np.floor(
        #             (self.df["days_since_release"] - 0.01) / self.interval_length
        #         ).astype(int)
        #         + 1
        #     )
        self.df[self.event_col] = 1
        # Mark preprocessing as complete
        self.is_preprocessed = True
        print("Data preprocessing complete.")

    def fit_cox_model(self):
        """Fit the Cox proportional hazards model and extract significant predictors."""
        # Define the formula for the Cox model
        exclude_cols = [str(self.event_col), str(self.duration_col)]
        columns_list = list(self.predictors)
        filtered_columns = [col for col in columns_list if col not in exclude_cols]

        formula = f"{self.duration_col} ~ " + " + ".join(filtered_columns)

        # Fit the Cox proportional hazards model
        cox_model = smf.phreg(formula, data=self.df, status=self.df[self.event_col])
        cox_result = cox_model.fit()

        # Extract significant predictors from Cox model
        self.cox_significant = [
            pred
            for pred, p in zip(self.predictors, cox_result.pvalues)
            if p < self.significance_level
        ]

    def fit_weibull_model(self):
        """Fit the Weibull AFT model and extract significant predictors."""
        aft = WeibullAFTFitter()
        try:
            # Attempt to fit the model
            aft.fit(
                self.df[self.predictors + [self.duration_col, self.event_col]],
                duration_col=self.duration_col,
                event_col=self.event_col,
            )
        except ValueError as e:
            # Check if the specific error message matches
            if "non-positive durations" in str(e):
                # Add a small positive value to the duration column
                self.df[self.duration_col] = self.df[self.duration_col].apply(
                    lambda x: x + 1e-6 if x <= 0 else x
                )
                # Retry fitting the model
                aft.fit(
                    self.df[self.predictors + [self.duration_col, self.event_col]],
                    duration_col=self.duration_col,
                    event_col=self.event_col,
                )
            else:
                # Raise the error if it's not the expected one
                raise e

        # Extract significant predictors from Weibull model, removing the lambda_ prefix
        self.aft_significant = aft.summary.index.get_level_values(1)[
            aft.summary["p"] < self.significance_level
        ].tolist()

    def fit_models_and_get_significant_predictors(self):
        if not self.is_preprocessed:
            print("Data should be preprocessed fisrt")
            return
        """Preprocess data, fit both Cox and Weibull models, then get combined significant predictors."""
        # Preprocess the dataset
        # self.preprocess_data()

        # Fit the Cox model and extract significant predictors
        self.fit_cox_model()

        # Fit the Weibull model and extract significant predictors
        self.fit_weibull_model()

        # Combine lists of significant predictors, keeping only unique values
        self.combined_significant = list(
            set(self.cox_significant + self.aft_significant)
        )

    def get_significance(self):
        """Return the lists of significant predictors for each model and combined."""
        print(f"cox_significant: {self.cox_significant}")
        print(f"weibull_significant: {self.aft_significant}")
        print(f"combined_significant: {self.combined_significant}")

        return {
            "cox_significant": self.cox_significant,
            "weibull_significant": self.aft_significant,
            "combined_significant": self.combined_significant,
        }

    def get_preprocessed_data(self):
        if not self.is_preprocessed:
            print("Data should be preprocessed fisrt")
            return
        """Return the preprocessed DataFrame."""
        return self.df


if __name__ == "__main__":
    import pymc as pm
    import pytensor.tensor as pt

    music_data = MusicService().get_music_data()
    service = MusicService()
    service.get_merged_data(music_data)
    data = service.get_featured_dataset()
    fitting_data = SurvivalModelTester(data)
    fitting_data.preprocess_data()
    df = fitting_data.get_preprocessed_data()
    event, risk = fitting_data.get_risk_event_matrices()
    preds = [
        "variance_streams",
        "artist_genre_performance",
        "artist_label_performance",
        "position_changes",
    ]

    def make_coxph(preds, intervals, risk, event, df):
        # Define coords with factorized indices and intervals
        coords = {
            "intervals": intervals,  # Time intervals for survival analysis
            "preds": preds,
            # "playlist": unique_playlists,  # Unique playlist identifiers
            "isrcs": range(len(df)),  # Observations count
        }
        with pm.Model(coords=coords) as base_model:
            # Setting up MutableData for predictor matrix (individual-level covariates)
            X_data = pm.Data("X_data_obs", df[preds], dims=("isrcs", "preds"))

            # Priors for baseline hazard rate and covariate coefficients
            lambda0 = pm.Gamma("lambda0", alpha=2.0, beta=2.0, dims="intervals")

            beta = pm.Normal("beta", 0, sigma=0.5, dims="preds")

            # Hazard rate calculation: lambda = exp(beta * X) * lambda0
            lambda_ = pm.Deterministic(
                "lambda_",
                pt.outer(pt.exp(pm.math.dot(beta, X_data.T)), lambda0),
                dims=("isrcs", "intervals"),
            )

            # Expected number of events using exposure matrix and lambda
            mu = pm.Deterministic("mu", risk * lambda_, dims=("isrcs", "intervals"))

            # Observed data likelihood
            # obs = pm.Poisson("obs", mu, observed=event, dims=("isrcs", "intervals"))
            alpha = pm.Exponential("alpha", 1.0)
            obs = pm.NegativeBinomial(
                "obs", mu=mu, alpha=alpha, observed=event, dims=("isrcs", "intervals")
            )

            # Sample posterior
            base_idata = pm.sample(
                chains=2,
                target_accept=0.95,
                random_seed=100,
                tune=500,
                idata_kwargs={"log_likelihood": True},
                return_inferencedata=True,
            )

        return base_idata, base_model

    base_idata, base_model = make_coxph(preds, fitting_data.intervals, risk, event, df)
    # ext_base_idata, ext_base_model = make_coxph(preds, intervals, risk, event, df)
