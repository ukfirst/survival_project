import matplotlib.pyplot as plt
import numpy as np
import arviz as az
from scipy.stats import probplot
import warnings

warnings.filterwarnings(
    "ignore",
    message="The effect of Potentials on other parameters is ignored during prior predictive sampling.*",
)
warnings.filterwarnings(
    "ignore",
    message=" The effect of Potentials on other parameters is ignored during prior predictive sampling.*",
)


def evaluate_and_plot_all_models(models, idata_dict, times, max_individuals=5):

    # Initialize lists to store evaluation metrics
    ci_list = []
    brier_scores_list = []
    model_names = []

    # Evaluate each model and store results
    for name, model_obj in models.items():
        print(f"Evaluating {name} model...")
        idata = idata_dict[name]
        ci, brier_scores = model_obj.evaluate_model(idata, times)
        ci_list.append(ci)
        brier_scores_list.append(brier_scores)
        model_names.append(name)

    # Plot Brier scores, Survival Curves, k_hat, Rank/Energy, PPC/Q-Q, and Autocorrelation for each model in rows
    num_models = len(models)
    fig, axes = plt.subplots(num_models, 5, figsize=(30, 6 * num_models))
    if num_models == 1:
        axes = np.array(
            [axes]
        )  # Make axes iterable and consistent with multi-model setup

    for idx, (ci, brier_scores, name) in enumerate(
        zip(ci_list, brier_scores_list, model_names)
    ):
        # Plot Brier scores in the first column
        ax_brier = axes[idx, 0]
        ax_brier.plot(times, brier_scores, label=f"{name} Model")
        ax_brier.set_xlabel("Time")
        ax_brier.set_ylabel("Brier Score")
        ax_brier.set_title(f"{name} Model - Brier Score\nConcordance Index: {ci:.2f}")
        ax_brier.legend()

        # Plot survival curves in the second column
        idata = idata_dict[name]
        survival_probs = models[name].predict_survival_function(idata, times)
        ax_surv = axes[idx, 1]
        for i in range(min(len(survival_probs), max_individuals)):
            ax_surv.step(
                times, survival_probs[i, :], where="post", label=f"Individual {i}"
            )
            ax_surv.axvline(
                models[name].event_times[i], color="r", linestyle="--", alpha=0.5
            )
        ax_surv.set_xlabel("Time")
        ax_surv.set_ylabel("Survival Probability")
        ax_surv.set_title(f"{name} Model - Survival Curves")
        ax_surv.legend()

        # Plot k_hat values in the third column, with error handling
        ax_khat = axes[idx, 2]
        try:
            loo_result = az.loo(idata)
            az.plot_khat(loo_result, show_bins=True, ax=ax_khat)
            elpd_loo = loo_result.elpd_loo
            ax_khat.set_title(f"{name} Model - k_hat values\nELPD LOO: {elpd_loo:.2f}")
        except Exception as e:
            ax_khat.set_title(f"{name} Model - k_hat values (log_likelihood missing)")
            ax_khat.axis("off")

        # Plot PPC or Q-Q plot in the fifth column
        ax_qq = axes[idx, 3]
        try:
            ppc = az.sample_posterior_predictive(idata)
            az.plot_ppc(ppc, ax=ax_qq)
            ax_qq.set_title(f"{name} Model - Posterior Predictive Check")
        except Exception:
            y_obs = models[name].event_times
            predicted = survival_probs.mean(axis=1)
            probplot(predicted, dist="norm", plot=ax_qq)
            ax_qq.set_title(f"{name} Model - Q-Q Plot for Survival Predictions")

        # Plot autocorrelation in the sixth column
        ax_autocorr = axes[idx, 4]
        try:
            param_names = list(idata.posterior.data_vars)
            # Autocorrelation plot for key parameters if available
            az.plot_autocorr(
                idata, var_names=param_names[:4], ax=ax_autocorr
            )  # Limit to 4 parameters
            ax_autocorr.set_title(f"{name} Model - Autocorrelation")
        except Exception as e:
            ax_autocorr.set_title(f"{name} Model - Autocorrelation (Unavailable)")
            ax_autocorr.axis("off")

    plt.tight_layout()
    plt.show()


def stepwise_selection(model_class, df, possible_preds, coords, censored, event_times):
    selected_preds = []
    base_idata = None  # Set initial base_idata to None for the first iteration
    base_model_obj = None

    for pred in possible_preds:
        temp_preds = selected_preds + [pred]
        print(f"Testing predictor(s): {temp_preds}")

        # Update the coordinates for the current set of predictors
        temp_coords = coords.copy()
        temp_coords["preds"] = temp_preds  # Update preds in coords

        # Instantiate and fit the model with the current set of predictors
        temp_model_obj = model_class(df, temp_preds, temp_coords, censored, event_times)
        temp_model_obj.build_model()
        temp_idata = temp_model_obj.sample(temp_model_obj.model)

        improvement = False  # Initialize improvement as False
        updated_elpd_loo = None  # Initialize updated_elpd_loo for tracking improvements

        if base_idata is not None:
            try:
                # Perform LOO comparison
                comparison = az.compare(
                    {"previous": base_idata, "current": temp_idata}, ic="loo"
                )
                # Check for model improvement based on elpd_loo
                if "current" in comparison.index and "previous" in comparison.index:
                    improvement = (
                        comparison.loc["current", "elpd_loo"]
                        > comparison.loc["previous", "elpd_loo"]
                    )  # Indicates improvement if elpd_diff is positive
                    if improvement:
                        updated_elpd_loo = comparison.loc["current", "elpd_loo"]
                else:
                    print(f"Comparison failed for predictor(s) {temp_preds}. Skipping.")
            except Exception as e:
                print(
                    f"LOO comparison encountered an error for predictor(s) {temp_preds}: {e}"
                )

        else:
            # First iteration, automatically accept the initial predictor(s)
            improvement = True
        # If the model improves, retain this predictor
        if improvement:
            selected_preds.append(pred)
            base_idata = temp_idata  # Update the baseline idata
            base_model_obj = temp_model_obj  # Update the baseline model object
            print(f"Selected predictor: {pred}")
            if updated_elpd_loo is not None:
                print(f"Updated elpd_loo: {updated_elpd_loo}")
        else:
            print(f"Predictor {pred} did not improve the model and was not selected.")

    print("Final selected predictors:", selected_preds)
    return selected_preds, base_model_obj, base_idata


# # Initialize an empty list of selected predictors
# selected_preds = []
# base_idata = None  # Set initial base_idata to None for the first iteration
#
# for pred in preds:
#     # Temporary list of predictors for this iteration
#     temp_preds = selected_preds + [pred]
#
#     # Fit the model with the current set of predictors
#     temp_idata, temp_model = make_coxph(
#         temp_preds, fitting_data.intervals, risk, event, df
#     )
#
#     if base_idata is not None:
#         # Compare models using LOO-CV for stability if there is a previous model
#         try:
#             comparison = az.compare(
#                 {"previous": base_idata, "current": temp_idata}, ic="loo"
#             )
#             improvement = (
#                 comparison.loc["current", "loo"] < comparison.loc["previous", "loo"]
#             )
#         except KeyError:
#             print(f"LOO comparison failed for predictor {pred}. Skipping.")
#             improvement = False
#     else:
#         # First iteration, automatically accept the initial predictor
#         improvement = True
#
#     # If the model improves, retain this predictor
#     if improvement:
#         selected_preds.append(pred)
#         base_idata, base_model = temp_idata, temp_model  # Update the baseline model
#         print(f"Selected predictor: {pred}")
#     else:
#         print(f"Predictor {pred} did not improve the model and was not selected.")
#
# print("Final selected predictors:", selected_preds)
