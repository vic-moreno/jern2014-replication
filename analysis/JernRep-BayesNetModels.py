# %% [markdown]
# # Bayes-Net Models: Jern et al. (2014) Replication
#
# This notebook defines the replication's belief-updating models as explicit
# Bayesian networks (pgmpy), fits them to participants' ratings, and compares
# them. It is organized in five parts:
#
# 1. **Modeling** — network definition and construction, exact inference,
#    and the fitting and plotting infrastructure shared by everything below.
# 2. **Point-mass reliability models** — Model A (chart prior) and Model C
#    (participant-level empirical prior): one inferred quantity, the
#    reliability `theta`. Fits, diagnostics, figures, comparison.
# 3. **Beta reliability models** — Model B (chart prior) and Model D
#    (empirical prior): a Beta(a, b) prior over `theta`. Fits, figures, the
#    identification ridge, an exact equivalence result, what the ridge says
#    about expected reliability, and what design would identify it.
# 4. **Response drift** — all four models with a condition-independent
#    additive response drift, estimated either from Control alone or jointly,
#    giving a 3 regimes x 4 models comparison.
# 5. **Interpretation**.
#
# ## The network
#
# Nodes:
#
# - `Theta`: reliability of the tests, discretized over a grid of candidate
#   values. Any prior shape can be supplied as a weight vector over the slots;
#   a point mass is a single-slot grid.
# - `D`: the patient's true disease, one of L1/L2 (Allozedic) or Y1/Y2
#   (Hypozedic).
# - `C`: the disease class (A = Allozedic, H = Hypozedic), a deterministic
#   function of `D`.
# - `T1`, `T2`: the two test results shown to participants.
#
# The graph differs by condition in where the tests attach:
#
# - Polarization & Moderation (tests name a *disease*):
#   `Theta -> T1 <- D -> C`, `Theta -> T2 <- D`, with
#   P(T = t | D, theta) = theta if t = D, else (1 - theta)/3.
# - Control (tests name a *class*): `Theta -> T1 <- C <- D`,
#   `Theta -> T2 <- C`, with P(T = t | C, theta) = theta_C if t = C, else
#   1 - theta_C, where theta_C = theta + (1 - theta)/3 is the probability
#   that a test with disease-level accuracy theta names a disease in the
#   true class (the true disease, or the other disease of the same class).
#   The class screens off the disease (T ⊥ D | C).
#
# **Lower bound on theta.** A disease-level test names one of four diseases,
# so a test answering at random is right a quarter of the time: theta = 0.25
# is the uninformative point (beliefs do not move) and theta < 0.25 would
# make the tests anti-informative, which the instructions ("fairly accurate
# but sometimes give false results") do not support. All fitting restricts
# theta to [0.25, 1). Under the derived class accuracy, theta in [0.25, 1]
# maps onto theta_C in [0.5, 1], so the class-level test's own chance floor
# is respected automatically by the same parameter.
#
# **No instructed reliability.** Participants were never shown a numerical
# reliability (`materials/stimuli/JernRep-ExperimentScreens-Pilot.pdf`); the
# 0.7 used by the R pipeline is Jern et al.'s modeling assumption. Here theta
# is always a fitted quantity. The R pipeline's models are reproduced exactly
# by these networks in `JernRep-BayesNetModels-Full.py`; that validation is
# omitted here.

# %% [markdown]
# # Part 1: Modeling

# %%
## Setup
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from scipy import stats
from scipy.optimize import brentq, minimize, minimize_scalar

## Outcome spaces
DISEASES = ["L1", "L2", "Y1", "Y2"]
CLASSES = ["A", "H"]  ## A = Allozedic, H = Hypozedic
CLASS_OF = {"L1": "A", "L2": "A", "Y1": "H", "Y2": "H"}

## Prior over the four diseases implied by the baseline frequency chart
CHART_PRIOR = {"L1": 0.01, "L2": 0.35, "Y1": 0.40, "Y2": 0.24}

## Reliability support. THETA_GRID (the fitting support) starts at the chance
## level of a four-outcome test; THETA_GRID_FULL is the unconstrained (0, 1)
## support, used only to show what the constraint changes.
THETA_FLOOR = 0.25
THETA_GRID = np.linspace(THETA_FLOOR, 0.999, 2000)
THETA_GRID_FULL = np.linspace(0.001, 0.999, 2000)

## Test results observed in each condition (T1 then T2)
POLARIZATION_TESTS = {"T1": "Y1", "T2": "L1"}
MODERATION_TESTS = {"T1": "Y2", "T2": "L2"}
CONTROL_TESTS = {"T1": "H", "T2": "A"}

DATA_DIR = next(
    d / "data" / "processed"
    for d in [Path.cwd(), Path.cwd().parent]
    if (d / "data" / "processed").is_dir()
)

# %% [markdown]
# ## Scale conversion
#
# Participants report beliefs on a -100..+100 slider; the networks operate
# purely on probability scale. These helpers convert between the two and are
# applied only when reading inputs or reporting outputs.

# %%
def to_slider(p):
    """Convert a probability in [0, 1] to the -100..+100 slider scale."""
    return 200 * np.asarray(p) - 100


def from_slider(rating):
    """Convert a -100..+100 slider rating to probability scale."""
    return (np.asarray(rating) + 100) / 200

# %% [markdown]
# ## Priors as inputs
#
# A theta prior is a `(grid, weights)` pair: candidate reliability values and
# a probability weight for each slot. Any shape over the slots is valid; a
# point mass is a one-slot grid. Disease priors are dicts over L1/L2/Y1/Y2:
# the chart itself, or the chart rescaled so the class totals match a
# participant's initial rating while the within-class ratios stay at the
# chart's values.

# %%
def point_mass_prior(value):
    """Theta prior with all mass on a single reliability value."""
    return np.array([value]), np.array([1.0])


def beta_prior(a, b, grid=THETA_GRID):
    """Beta(a, b) theta prior, discretized (and thereby truncated) to grid."""
    with np.errstate(divide="ignore", invalid="ignore"):  ## degenerate (a, b) -> nan, handled by callers
        weights = stats.beta.pdf(grid, a, b)
        return grid, weights / weights.sum()


def empirical_prior(p_hypo, base=CHART_PRIOR):
    """Disease prior rescaled so the class totals match an observed class-level
    belief, keeping the within-class ratios fixed at the base (chart) values."""
    p_base = sum(base[d] for d in DISEASES if CLASS_OF[d] == "H")
    scale = {"A": (1 - p_hypo) / (1 - p_base), "H": p_hypo / p_base}
    return {d: base[d] * scale[CLASS_OF[d]] for d in DISEASES}

# %% [markdown]
# ## Network construction
#
# One public function per condition. Polarization and Moderation share the
# disease-level graph (they differ only in which test results are observed);
# Control attaches the tests to the class node with the derived class-level
# accuracy theta_C = theta + (1 - theta)/3.

# %%
def class_level_accuracy(theta):
    """Probability that a test with disease-level accuracy theta names a
    disease in the true class: the true disease (theta) or the other disease
    of the same class ((1 - theta) / 3). Maps [0.25, 1] onto [0.5, 1]."""
    theta = np.asarray(theta, dtype=float)
    return theta + (1 - theta) / 3


def _theta_cpd(grid, weights):
    weights = np.asarray(weights, dtype=float)
    return TabularCPD(
        "Theta", len(grid), weights.reshape(-1, 1) / weights.sum(),
        state_names={"Theta": [float(v) for v in grid]},
    )


def _disease_cpds(prior_D):
    p = np.array([prior_D[d] for d in DISEASES], dtype=float)
    cpd_D = TabularCPD("D", 4, (p / p.sum()).reshape(-1, 1), state_names={"D": DISEASES})
    ## C is a deterministic readout of D's class
    cpd_C = TabularCPD(
        "C", 2, [[1.0 if CLASS_OF[d] == c else 0.0 for d in DISEASES] for c in CLASSES],
        evidence=["D"], evidence_card=[4], state_names={"C": CLASSES, "D": DISEASES},
    )
    return cpd_D, cpd_C


def _test_cpd(test, parent, parent_states, grid, accuracy=None):
    """CPD for one test given its parent (D or C) and Theta: the accuracy when
    the result matches the parent's state, the remainder split evenly across
    the other outcomes. Accuracy defaults to theta (disease-level tests)."""
    grid = np.asarray(grid, dtype=float)
    acc = grid if accuracy is None else np.asarray(accuracy, dtype=float)
    k = len(parent_states)
    blocks = []
    for i in range(k):  ## Columns: parent-major blocks of theta slots
        block = np.tile((1 - acc) / (k - 1), (k, 1))
        block[i] = acc
        blocks.append(block)
    return TabularCPD(
        test, k, np.hstack(blocks),
        evidence=[parent, "Theta"], evidence_card=[k, len(grid)],
        state_names={test: parent_states, parent: parent_states,
                     "Theta": [float(v) for v in grid]},
    )


def _make_disease_level_network(prior_D, theta_grid, theta_weights):
    net = DiscreteBayesianNetwork(
        [("D", "C"), ("D", "T1"), ("D", "T2"), ("Theta", "T1"), ("Theta", "T2")]
    )
    net.add_cpds(
        *_disease_cpds(prior_D), _theta_cpd(theta_grid, theta_weights),
        _test_cpd("T1", "D", DISEASES, theta_grid),
        _test_cpd("T2", "D", DISEASES, theta_grid),
    )
    net.check_model()
    return net


def make_polarization_network(prior_D, theta_grid, theta_weights):
    """Polarization condition: tests name diseases (observed: Y1, then L1)."""
    return _make_disease_level_network(prior_D, theta_grid, theta_weights)


def make_moderation_network(prior_D, theta_grid, theta_weights):
    """Moderation condition: tests name diseases (observed: Y2, then L2)."""
    return _make_disease_level_network(prior_D, theta_grid, theta_weights)


def make_control_network(prior_D, theta_grid, theta_weights):
    """Control condition: tests name classes, so they attach to C, not D,
    with the derived class-level accuracy."""
    net = DiscreteBayesianNetwork(
        [("D", "C"), ("C", "T1"), ("C", "T2"), ("Theta", "T1"), ("Theta", "T2")]
    )
    net.add_cpds(
        *_disease_cpds(prior_D), _theta_cpd(theta_grid, theta_weights),
        _test_cpd("T1", "C", CLASSES, theta_grid, class_level_accuracy(theta_grid)),
        _test_cpd("T2", "C", CLASSES, theta_grid, class_level_accuracy(theta_grid)),
    )
    net.check_model()
    return net


NETWORKS = {
    "Polarization": (make_polarization_network, POLARIZATION_TESTS),
    "Moderation": (make_moderation_network, MODERATION_TESTS),
    "Control": (make_control_network, CONTROL_TESTS),
}

# %% [markdown]
# ## Inference
#
# Exact inference by variable elimination. `predict` returns the class-level
# belief before and after the tests plus the disease and theta posteriors;
# `final_belief` is the single query that fitting needs.

# %%
def _query(infer, variables, evidence=None):
    try:
        return infer.query(variables=variables, evidence=evidence, show_progress=False)
    except TypeError:  ## pgmpy versions without show_progress
        return infer.query(variables=variables, evidence=evidence)


def predict(network, evidence):
    """P(C = Hypozedic) before/after the tests, with D and Theta posteriors."""
    infer = VariableElimination(network)
    prior_C = _query(infer, ["C"])
    post_C = _query(infer, ["C"], evidence)
    post_D = _query(infer, ["D"], evidence)
    post_theta = _query(infer, ["Theta"], evidence)
    return {
        "p_initial": float(prior_C.get_value(C="H")),
        "p_final": float(post_C.get_value(C="H")),
        "p_D": {d: float(post_D.get_value(D=d)) for d in DISEASES},
        "theta_grid": np.array(post_theta.state_names["Theta"], dtype=float),
        "theta_posterior": post_theta.values,
    }


def final_belief(condition, prior_D, theta_prior):
    """P(C = Hypozedic | tests) for one condition's network."""
    make_network, tests = NETWORKS[condition]
    grid, weights = theta_prior
    infer = VariableElimination(make_network(prior_D, grid, weights))
    return float(_query(infer, ["C"], tests).get_value(C="H"))

# %% [markdown]
# ## Data

# %%
obs = pd.read_csv(DATA_DIR / "JernRep-Data-Processed-Full.csv")
emmeans = pd.read_csv(DATA_DIR / "JernRep-Results-Full-MixedModel-EstMeans.csv")
n_obs = len(obs)
cond_order = ["Moderation", "Control", "Polarization"]
chart_initial = float(to_slider(sum(CHART_PRIOR[d] for d in ["Y1", "Y2"])))
mean_initial = obs.groupby("Condition")["Initial"].mean()
mean_final = obs.groupby("Condition")["Final"].mean()

print(f"n = {n_obs} participants (initial ratings below 0 were excluded in data processing)")
print(obs.groupby("Condition")[["Initial", "Final"]].agg(["mean", "std", "count"]).round(2))

# %% [markdown]
# ## Fitting infrastructure
#
# **Predictions.** For each participant, a network for their condition with
# a disease prior that is either the chart (`"chart"`) or the chart rescaled
# to their own initial rating (`"empirical"`), queried for the class belief
# after the tests and converted to the slider scale. Predictions depend only
# on (condition, initial rating), so they are cached per unique pair.
#
# **Likelihood.** Each participant's Final rating is Normal(fitted, sigma),
# with one noise SD shared across conditions. Optionally a
# condition-independent *response drift* delta is added to the network's
# prediction P: fitted = P + delta. The drift is either absent, fixed at a
# value supplied by the caller (Part 4 anchors it to Control), or profiled:
# given the predictions, the ML delta is the mean residual Final - P. Sigma
# is always profiled (sigma^2 = SSE / n). Maximizing the resulting profile
# log-likelihood over the reliability parameters is the joint MLE. Sigma and
# delta count as parameters in AIC/BIC whenever they are estimated from the
# data.
#
# **Reliability.** Point-mass models optimize theta in [0.25, 1) by bounded
# scalar search; Beta models optimize (log a, log b) by Nelder-Mead from
# several starts, on the truncated grid.

# %%
def prior_D_for(prior_type, initial):
    if prior_type == "chart":
        return CHART_PRIOR
    return empirical_prior(float(from_slider(initial)))


def participant_predictions(theta_prior, prior_type, data):
    """Network-predicted final rating (slider scale) for each row of data."""
    conds = data["Condition"].to_numpy()
    initials = data["Initial"].to_numpy()
    keys = list(conds) if prior_type == "chart" else list(zip(conds, initials))
    cache = {}
    for key in set(keys):
        cond, initial = (key, None) if prior_type == "chart" else key
        cache[key] = float(to_slider(final_belief(cond, prior_D_for(prior_type, initial), theta_prior)))
    return np.array([cache[k] for k in keys])


def evaluate(preds, data, delta=None):
    """Profile log-likelihood of the final ratings given network predictions.

    delta: None (rating = prediction), "joint" (delta profiled as the mean
    residual Final - P), or a fixed number.
    """
    finals = data["Final"].to_numpy()
    if delta is None:
        shift = None
    elif isinstance(delta, str):
        shift = float(np.mean(finals - preds))
    else:
        shift = float(delta)
    fitted = preds if shift is None else preds + shift
    resid = finals - fitted
    sigma = float(np.sqrt(np.mean(resid ** 2)))
    return {"logLik": float(np.sum(stats.norm.logpdf(resid, scale=sigma))),
            "sigma": sigma, "delta": shift, "fitted": fitted}


def _extra_params(delta):
    return 0 if delta is None else 1  ## delta estimated from the data


def fit_point_mass(prior_type, delta=None, data=None):
    """Point-mass reliability by maximum likelihood."""
    data = obs if data is None else data
    def nll(theta):
        return -evaluate(participant_predictions(point_mass_prior(theta), prior_type, data),
                         data, delta)["logLik"]
    result = minimize_scalar(nll, bounds=(THETA_FLOOR, 0.999), method="bounded",
                             options={"xatol": 1e-6})
    preds = participant_predictions(point_mass_prior(result.x), prior_type, data)
    return {"family": "point", "prior": prior_type, "theta": float(result.x), "preds": preds,
            "n_params": 2 + _extra_params(delta), **evaluate(preds, data, delta)}


def fit_beta(prior_type, starts, delta=None, data=None):
    """Beta(a, b) reliability prior by maximum likelihood, multi-start."""
    data = obs if data is None else data
    def nll(log_ab):
        a, b = np.exp(log_ab)
        try:
            value = -evaluate(participant_predictions(beta_prior(a, b), prior_type, data),
                              data, delta)["logLik"]
            return value if np.isfinite(value) else 1e12
        except Exception:
            return 1e12
    runs = []
    for start in starts:
        result = minimize(nll, np.log(start), method="Nelder-Mead",
                          options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 400})
        a, b = np.exp(result.x)
        runs.append({"start": start, "a": float(a), "b": float(b), "logLik": -result.fun,
                     "r": evidence_ratio(beta_prior(a, b))})
    best = max(runs, key=lambda run: run["logLik"])
    preds = participant_predictions(beta_prior(best["a"], best["b"]), prior_type, data)
    return {"family": "beta", "prior": prior_type, "a": best["a"], "b": best["b"], "r": best["r"],
            "runs": runs, "preds": preds, "n_params": 3 + _extra_params(delta),
            **evaluate(preds, data, delta)}


def evidence_ratio(theta_prior):
    """r = E[theta (1 - theta)] / E[(1 - theta)^2]: the only feature of a
    theta prior that a conflicting test pair can see (Part 3)."""
    grid, weights = theta_prior
    w = np.asarray(weights, dtype=float) / np.sum(weights)
    return float(np.sum(w * grid * (1 - grid)) / np.sum(w * (1 - grid) ** 2))


def prior_moments(theta_prior):
    grid, w = theta_prior[0], theta_prior[1] / theta_prior[1].sum()
    mean = float(np.sum(w * grid))
    return mean, float(np.sqrt(np.sum(w * (grid - mean) ** 2)))


def fit_stats(loglik, n_params):
    return {"logLik": loglik, "AIC": 2 * n_params - 2 * loglik,
            "BIC": n_params * np.log(n_obs) - 2 * loglik}


def condition_means(values):
    return obs.assign(v=values).groupby("Condition")["v"].mean()


def describe(fit):
    if fit["family"] == "point":
        est = f"theta = {fit['theta']:.4f}"
    else:
        est = f"a = {fit['a']:.3f}, b = {fit['b']:.3f}, r = {fit['r']:.4f}"
    if fit["delta"] is not None:
        est += f", delta = {fit['delta']:.2f}"
    return est


def report(fit):
    print(f"{describe(fit)}, sigma = {fit['sigma']:.2f}, logLik = {fit['logLik']:.3f}")
    print("  mean fitted final by condition:",
          {c: round(v, 2) for c, v in condition_means(fit["fitted"]).items()})


def report_runs(fit):
    for run in fit["runs"]:
        mean, sd = prior_moments(beta_prior(run["a"], run["b"]))
        print(f"  start {run['start']}: a = {run['a']:7.3f}, b = {run['b']:7.3f}, "
              f"prior mean = {mean:.3f}, sd = {sd:.3f}, r = {run['r']:.4f}, "
              f"logLik = {run['logLik']:.3f}")


def summary_table(entries):
    """entries: list of (name, fit_like, estimate string); fit_like needs
    logLik, sigma, fitted, n_params."""
    rows = [{"Model": name, "Estimate": est, "k": f["n_params"], "sigma": f["sigma"],
             **fit_stats(f["logLik"], f["n_params"])} for name, f, est in entries]
    table = pd.DataFrame(rows).set_index("Model")
    table["dAIC"] = table["AIC"] - table["AIC"].min()
    return table


## Reference model: no update at all (final = initial)
fit_null = {"n_params": 1, **evaluate(obs["Initial"].to_numpy(float), obs)}

# %% [markdown]
# ## Plotting infrastructure
#
# Two figure types in the style of the repo's result figures
# (`JernRep-Figures.qmd`). **Pre/post trajectories**: thin gray lines are
# individual participants, the teal line the mixed-model estimated marginal
# mean with its 95% CI ribbon, and model trajectories run from the model's
# initial belief to the mean of its fitted finals. **Final against initial**:
# each participant's final plotted against their initial per condition, with
# model curves, an OLS fit to the participants, and marginal distributions
# (initials above, finals to the right, with the model-implied finals as a
# dark outline).

# %%
phases = [0, 1]
initial_grid = np.arange(0, 101, 2)
curve_data = pd.DataFrame({
    "Condition": np.repeat(cond_order, len(initial_grid)),
    "Initial": np.tile(initial_grid, len(cond_order)),
})
SCALE_CAPTION = ("Certainty ratings range from -100 (Less Common Disease Class) to +100 "
                 "(More Common Disease Class); initial ratings below 0 were excluded.")


def trajectory(fit):
    """{condition: (model initial, mean fitted final)} for a fitted model."""
    start = mean_initial if fit["prior"] == "empirical" else pd.Series(chart_initial, index=cond_order)
    end = condition_means(fit["fitted"])
    return {c: (start[c], end[c]) for c in cond_order}


def model_curve(theta_prior, prior_type, delta=None):
    """{condition: fitted final over initial_grid} for a model."""
    preds = participant_predictions(theta_prior, prior_type, curve_data)
    if delta is not None:
        preds = preds + delta
    return {c: preds[(curve_data["Condition"] == c).to_numpy()] for c in cond_order}


def curve_of(fit):
    prior = point_mass_prior(fit["theta"]) if fit["family"] == "point" else beta_prior(fit["a"], fit["b"])
    return model_curve(prior, fit["prior"], fit["delta"])


def plot_prepost(trajectories):
    """trajectories: list of (label, {condition: (initial, final)}, color, linestyle)."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.6), sharey=True)
    for ax, cond in zip(axes, cond_order):
        sub = obs[obs["Condition"] == cond]
        ax.plot(phases, sub[["Initial", "Final"]].T.to_numpy(), color="gray", alpha=0.05, lw=0.8)
        em = emmeans[emmeans["Condition"] == cond].set_index("Phase")
        ax.fill_between(phases, em.loc[["Initial", "Final"], "lower.CL"],
                        em.loc[["Initial", "Final"], "upper.CL"], color="#3E7C91", alpha=0.15)
        ax.plot(phases, em.loc[["Initial", "Final"], "emmean"], color="#3E7C91", lw=2.4,
                label="Estimated marginal means")
        for label, traj, color, ls in trajectories:
            ax.plot(phases, traj[cond], color=color, lw=1.6, ls=ls, label=label)
        ax.axhline(0, ls=":", color="gray", lw=0.8)
        ax.set_title(cond, fontweight="bold", fontsize=11)
        ax.set_xticks(phases, ["Initial", "Final"])
        ax.set_xlim(-0.25, 1.25)
    axes[0].set_ylim(-100, 100)
    axes[0].set_yticks(range(-100, 101, 50))
    axes[0].set_ylabel("Certainty Rating")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.03))
    fig.text(0.01, 0.005, SCALE_CAPTION, fontsize=7.5, color="gray")
    fig.subplots_adjust(bottom=0.27, left=0.07, right=0.98, top=0.9)
    return fig


def plot_scatter(curves, margin_fitted, margin_label):
    """curves: list of (label, {condition: y over initial_grid}, color, linestyle);
    margin_fitted: fitted finals per row of obs, drawn as an outline on the right margin."""
    bins_initial = np.arange(0, 101, 10)
    bins_final = np.arange(-100, 101, 10)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.4), sharey=True)
    for ax, cond in zip(axes, cond_order):
        sub = obs[obs["Condition"] == cond]
        in_cond = (obs["Condition"] == cond).to_numpy()
        ax.scatter(sub["Initial"], sub["Final"], s=14, color="#3E7C91", alpha=0.45,
                   linewidths=0, label="Participants")
        ols = stats.linregress(sub["Initial"], sub["Final"])
        ax.plot([0, 100], [ols.intercept, ols.intercept + 100 * ols.slope], color="#3E7C91",
                lw=1.2, ls="--", label="OLS fit to participants")
        ax.plot([0, 100], [0, 100], ls=":", color="gray", lw=1, label="No change")
        for label, curve, color, ls in curves:
            ax.plot(initial_grid, curve[cond], color=color, lw=1.7, ls=ls, label=label)
        ax.axhline(0, ls=":", color="gray", lw=0.6)
        ax.set_xlim(-3, 103)
        ax.set_xlabel("Initial rating")
        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("top", size="22%", pad=0.05, sharex=ax)
        ax_right = divider.append_axes("right", size="22%", pad=0.05, sharey=ax)
        ax_top.hist(sub["Initial"], bins=bins_initial, color="#3E7C91", alpha=0.55)
        ax_right.hist(sub["Final"], bins=bins_final, color="#3E7C91", alpha=0.55,
                      orientation="horizontal")
        ax_right.hist(margin_fitted[in_cond], bins=bins_final, histtype="step", color="#303030",
                      lw=1.2, orientation="horizontal")
        for marginal in (ax_top, ax_right):
            marginal.axis("off")
        ax_top.set_title(cond, fontweight="bold", fontsize=11)
    axes[0].set_ylim(-100, 100)
    axes[0].set_yticks(range(-100, 101, 50))
    axes[0].set_ylabel("Final rating")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color="#303030", lw=1.2))
    labels.append(margin_label)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.03))
    fig.text(0.01, 0.005, SCALE_CAPTION, fontsize=7.5, color="gray")
    fig.subplots_adjust(bottom=0.25, left=0.07, right=0.98, top=0.95)
    return fig

# %% [markdown]
# ## Diagnostics infrastructure
#
# **Per-condition theta.** The substantive models use one theta for
# everyone; refitting theta separately to the Polarization and Moderation
# participants (Control is uninformative about theta) checks whether the two
# conditions pull it apart. Any drift is held fixed at the shared fit's
# value so the comparison isolates theta.
#
# **Reachable envelope.** The prediction is monotone in theta, so each
# participant's reachable final ratings run from their no-update value
# (theta = 0.25) to the theta -> 1 limit, in which only the two named
# diseases survive in proportion to their priors. A final outside that
# interval cannot be produced by any reliability — either the participant
# moved the wrong way, or past the limit. Under drift the whole envelope
# shifts by delta.

# %%
def per_condition_thetas(shared_fit):
    """Refit theta per informative condition with the shared fit's drift fixed."""
    delta = shared_fit["delta"]
    out, fitted = {}, shared_fit["fitted"].copy()
    for cond in ["Polarization", "Moderation"]:
        mask = (obs["Condition"] == cond).to_numpy()
        out[cond] = fit_point_mass(shared_fit["prior"], delta=delta, data=obs[mask])
        fitted[mask] = out[cond]["fitted"]
    combined = {"n_params": shared_fit["n_params"] + 1, **evaluate(fitted, obs)}
    combined["delta"] = delta
    combined["logLik"] = float(np.sum(stats.norm.logpdf(obs["Final"].to_numpy() - fitted,
                                                        scale=combined["sigma"])))
    lr = 2 * (combined["logLik"] - shared_fit["logLik"])
    print(f"per-condition theta: " + ", ".join(
        f"{c} = {f['theta']:.4f}" for c, f in out.items()) +
        f" | shared = {shared_fit['theta']:.4f}; logLik {shared_fit['logLik']:.2f} -> "
        f"{combined['logLik']:.2f}, LR = {lr:.2f} on 1 df, p = {stats.chi2.sf(lr, 1):.4f}")
    return out, combined


def envelope_counts(delta=None):
    rows = []
    for cond in ["Polarization", "Moderation"]:
        sub = obs[obs["Condition"] == cond]
        below = above = 0
        for initial, final in zip(sub["Initial"], sub["Final"]):
            prior_D = empirical_prior(float(from_slider(initial)))
            reach = np.array([to_slider(final_belief(cond, prior_D, point_mass_prior(t)))
                              for t in (THETA_FLOOR, 1 - 1e-9)])
            if delta is not None:
                reach = reach + delta
            below += final < reach.min() - 1e-9
            above += final > reach.max() + 1e-9
        rows.append({"Condition": cond, "n": len(sub), "outside": below + above,
                     "share": (below + above) / len(sub), "below": below, "above": above})
    return pd.DataFrame(rows).set_index("Condition")

# %% [markdown]
# # Part 2: Point-mass reliability models
#
# | Model | Disease prior | Reliability | Free parameters |
# |---|---|---|---|
# | A | chart | point mass at theta | theta, sigma |
# | C | participant-level empirical | point mass at theta | theta, sigma |
#
# These models infer one variable, the reliability. Under the chart prior
# every participant starts at `to_slider(0.64) = 28`, so only final ratings
# inform the fit and each condition gets one predicted final. Under the
# empirical prior each participant starts from their own rating and the model
# is a curve from initial to final. Bayesian updating is nonlinear in the
# prior, so this is not the same as using a condition-mean initial rating.
#
# Two structural facts: theta is one population-level parameter (a
# per-participant theta with one final rating each would be saturated), and
# under any reliability the Control prediction equals the prior exactly (the
# conflicting class-level pair gives every disease the same likelihood), so
# Polarization and Moderation carry all the information about theta.

# %% [markdown]
# ## Model A: chart prior

# %%
fit_A = fit_point_mass("chart")
report(fit_A)

# %% [markdown]
# ## Model C: participant-level empirical prior

# %%
fit_C = fit_point_mass("empirical")
report(fit_C)
print(obs.assign(Fitted=fit_C["fitted"])
         .groupby("Condition")[["Initial", "Final", "Fitted"]].mean().round(2))

# %% [markdown]
# ## Diagnostics

# %%
print("Model C, theta by condition")
per_cond_C, fit_C_percond = per_condition_thetas(fit_C)

control = obs[obs["Condition"] == "Control"]
lr_control = stats.linregress(control["Initial"], control["Final"])
print(f"\nControl: Final = {lr_control.intercept:.2f} + {lr_control.slope:.3f} * Initial "
      f"(SE {lr_control.stderr:.3f}, r = {lr_control.rvalue:.2f}, n = {len(control)}); "
      f"mean change = {(control['Final'] - control['Initial']).mean():.2f}")

print("\nReachable envelope (empirical prior, no drift):")
envelope_none = envelope_counts()
print(envelope_none.round(2).to_string())

# %% [markdown]
# **Reading the diagnostics.** Fitted separately, the two informative
# conditions want very different reliabilities, and the Control regression
# says why that should not be read as a difference in updating: the slope of
# Final on Initial is well below 1 with a negative mean change, so finals
# drift down and toward the middle with no evidence at all. That drift
# opposes the upward update in Polarization and assists the downward update
# in Moderation. The envelope makes the misfit concrete: a large share of
# participants in both conditions sit outside anything a single-reliability
# model can produce, in Polarization mostly by moving the wrong way, in
# Moderation mostly by moving past the perfect-reliability limit. Part 4
# takes up the drift; the Moderation overshoot is a different problem (a
# cluster that discarded the prior entirely).

# %% [markdown]
# ## Figures

# %%
fig = plot_prepost([
    ("A: chart prior", trajectory(fit_A), "#303030", "--"),
    ("C: empirical prior", trajectory(fit_C), "#8B3A3A", "--"),
])

# %%
fig = plot_scatter([
    (f"A: chart prior, theta = {fit_A['theta']:.2f}", curve_of(fit_A), "#303030", ":"),
    (f"C: empirical prior, theta = {fit_C['theta']:.2f}", curve_of(fit_C), "#303030", "-"),
    ("C: per-condition theta", {c: (curve_of(per_cond_C[c])[c] if c in per_cond_C
                                    else curve_of(fit_C)[c]) for c in cond_order}, "#8B3A3A", "--"),
], fit_C["fitted"], "Model C implied finals (right margin)")

# %% [markdown]
# ## Comparison

# %%
table_point = summary_table([
    ("Null: no update (final = initial)", fit_null, "-"),
    ("A: chart prior, point theta", fit_A, describe(fit_A)),
    ("C: empirical prior, point theta", fit_C, describe(fit_C)),
    ("C, per-condition theta (diagnostic)", fit_C_percond,
     ", ".join(f"{c[:3]} {f['theta']:.3f}" for c, f in per_cond_C.items())),
])
print(table_point.round(2).to_string())

# %% [markdown]
# **Reading the comparison.** Any updating model beats no-update decisively:
# the tests moved beliefs in the Bayesian direction. Starting participants
# from their own ratings (C) removes the glaring misfit of the chart prior —
# everyone starting at +28 while they actually started near +50 — yet gains
# under one log-likelihood unit, because a single theta is then a compromise
# between a Polarization that rose too little and a Moderation that fell too
# much.

# %% [markdown]
# # Part 3: Beta reliability models
#
# | Model | Disease prior | Reliability | Free parameters |
# |---|---|---|---|
# | B | chart | Beta(a, b) over [0.25, 1) | a, b, sigma |
# | D | participant-level empirical | Beta(a, b) over [0.25, 1) | a, b, sigma |
#
# The observer is now uncertain about the tests' reliability and infers it
# jointly with the diagnosis. Each fit is run from several starts; the
# multi-start tables are the first sign of the identification problem
# analyzed below.

# %% [markdown]
# ## Model B: chart prior

# %%
fit_B = fit_beta("chart", starts=[(3.5, 1.5), (1.0, 1.0), (7.0, 3.0), (0.5, 0.5)])
report_runs(fit_B)
report(fit_B)

# %% [markdown]
# ## Model D: participant-level empirical prior

# %%
fit_D = fit_beta("empirical", starts=[(2.0, 2.0), (0.7, 0.4)])
report_runs(fit_D)
report(fit_D)

# %% [markdown]
# ## Figures
#
# B lies exactly on A and D on C in both figures.

# %%
fig = plot_prepost([
    ("A: chart, point theta", trajectory(fit_A), "#303030", "--"),
    ("B: chart, Beta theta", trajectory(fit_B), "#303030", ":"),
    ("C: empirical, point theta", trajectory(fit_C), "#8B3A3A", "--"),
    ("D: empirical, Beta theta", trajectory(fit_D), "#8B3A3A", ":"),
])

# %%
fig = plot_scatter([
    ("A: chart, point theta", curve_of(fit_A), "#303030", "-"),
    ("B: chart, Beta theta", curve_of(fit_B), "#8B3A3A", ":"),
    ("C: empirical, point theta", curve_of(fit_C), "#303030", "-"),
    ("D: empirical, Beta theta", curve_of(fit_D), "#8B3A3A", ":"),
], fit_D["fitted"], "Model D implied finals (right margin)")

# %% [markdown]
# ## The ridge: (a, b) are not identified
#
# Every start converges to a different (a, b) with the same log-likelihood,
# the same predictions, and the same value of one number, r. The ridge is
# exact, not an optimizer artifact.
#
# With two *conflicting* disease-level tests, the likelihood of the pair
# given disease D is theta (1 - theta) / 3 when D is one of the two named
# diseases and ((1 - theta) / 3)^2 otherwise. Averaging over any theta
# prior, the posterior over diseases depends on the prior only through
#
#     r = E[theta (1 - theta)] / E[(1 - theta)^2],
#
# and the Control prediction does not depend on theta at all. This holds
# for every disease prior. A point mass at theta0 has r = theta0/(1-theta0),
# which sweeps every r in [1/3, inf) as theta0 runs over [0.25, 1). So
# every theta prior, however spread out, makes exactly the same predictions
# as some point mass: B's ridge is the level set r = theta_A/(1-theta_A) and
# D's is r = theta_C/(1-theta_C). The theta >= 0.25 bound leaves this
# untouched — r is a ratio of expectations of positive functions, bounded by
# the extremes of theta/(1-theta) over the support, so restricting the
# support restricts r to [1/3, inf) and every attainable r is still attained
# by a point mass inside the bound.

# %%
for label, point, beta_fit in [("A vs B", fit_A, fit_B), ("C vs D", fit_C, fit_D)]:
    r_point = point["theta"] / (1 - point["theta"])
    print(f"{label}: theta/(1-theta) = {r_point:.4f} | r across Beta starts = "
          f"{[round(run['r'], 4) for run in beta_fit['runs']]} | logLik point = "
          f"{point['logLik']:.3f}, Beta = {beta_fit['logLik']:.3f}")
    assert np.isclose(point["logLik"], beta_fit["logLik"], atol=0.5)

# %% [markdown]
# ## Comparison

# %%
table_beta = summary_table([
    ("Null: no update (final = initial)", fit_null, "-"),
    ("A: chart prior, point theta", fit_A, describe(fit_A)),
    ("C: empirical prior, point theta", fit_C, describe(fit_C)),
    ("B: chart prior, Beta theta", fit_B, f"r = {fit_B['r']:.3f} (a, b on a ridge)"),
    ("D: empirical prior, Beta theta", fit_D, f"r = {fit_D['r']:.3f} (a, b on a ridge)"),
])
print(table_beta.round(2).to_string())

# %% [markdown]
# B reaches exactly A's likelihood and D exactly C's, so AIC and BIC prefer
# the point-mass versions purely on parameter count. This is not evidence
# that participants held a sharp belief about reliability; it is the design
# being unable to distinguish the two.

# %% [markdown]
# ## What the ridge says about expected reliability
#
# Equivalence does not mean the Beta and the point mass agree about
# reliability. Along the ridge the Beta's *mean* is always higher than the
# equivalent point mass. For a Beta on all of (0, 1) with a = r (b + 1),
#
#     E[theta] = r (b + 1) / (r (b + 1) + b)  >  r / (1 + r) = theta0,
#
# with equality only as b -> inf and the mean climbing toward 1 as b -> 0.
# The mechanism is the weighting inside r: a conflicting pair is evidence
# against high reliability, and its (1 - theta)^2 factor discounts
# high-theta mass, so a spread prior needs a higher mean to deliver the same
# effective evidence weight. Uncertainty about reliability discounts a
# conflicting pair by itself — near-certain reliability is incompatible with
# a conflict — so the fitted point mass is the *floor* of the mean
# reliability beliefs consistent with the data, not an estimate of that mean.
#
# The theta >= 0.25 constraint changes what the ridge admits. On (0, 1) the
# ridge runs to near-flat priors, because anti-informative mass below 0.25
# can offset near-certain mass close to 1. On [0.25, 1) that trade is
# unavailable: below a minimum b the widest priors have r above the target
# for *every* a and cannot lie on the ridge at all, so the admissible means
# compress toward theta0. The trace below finds `b_min` by solving
# r(a -> 0, b) = r on each support and the ridge member at each b by solving
# for a.

# %%
LOG_A_BRACKET = (np.log(1e-4), np.log(1e4))


def ridge_member(r_target, b, grid):
    """Beta(a, b) on grid with evidence ratio r_target, or None if no a works."""
    f = lambda log_a: evidence_ratio(beta_prior(np.exp(log_a), b, grid)) - r_target
    if f(LOG_A_BRACKET[0]) > 0:
        return None
    return float(np.exp(brentq(f, *LOG_A_BRACKET)))


def trace_ridge(point_fit, label, b_values=(0.05, 0.2, 0.5, 1, 2, 5, 10, 30, 100)):
    r_target = point_fit["theta"] / (1 - point_fit["theta"])
    rows = []
    for name, grid in [("(0, 1)", THETA_GRID_FULL), ("[0.25, 1)", THETA_GRID)]:
        g = lambda log_b: (evidence_ratio(beta_prior(np.exp(LOG_A_BRACKET[0]), np.exp(log_b), grid))
                           - r_target)
        b_min = float(np.exp(brentq(g, np.log(1e-3), np.log(1e3)))) if g(np.log(1e-3)) > 0 else None
        print(f"{label}, support {name}: ridge exists for "
              f"{'b >= %.3f' % b_min if b_min else 'every b'}")
        for b in ([b_min * 1.001] if b_min else []) + list(b_values):
            a = ridge_member(r_target, b, grid)
            mean, sd = prior_moments(beta_prior(a, b, grid)) if a is not None else (np.nan, np.nan)
            rows.append({"support": name, "b": round(b, 3), "a": a, "mean": mean, "sd": sd})
    table = pd.DataFrame(rows)
    print(f"theta0 = {point_fit['theta']:.4f}, r = {r_target:.4f}")
    print(table.pivot(index="b", columns="support", values=["a", "mean", "sd"]).round(3).to_string())
    assert (table.dropna()["mean"] > point_fit["theta"]).all()
    return table


ridge_B = trace_ridge(fit_A, "Model B ridge")
print()
ridge_D = trace_ridge(fit_C, "Model D ridge")

# %% [markdown]
# On (0, 1) the ridge admits means from theta0 up to about 0.8. On
# [0.25, 1) it stops at b of about 1.4 (B) or 1.1 (D), where the widest
# admissible member has a mean near 0.5 with sd near 0.2. Under the
# constraint the data are therefore consistent with mean reliability beliefs
# anywhere from the point mass up to roughly 0.5 — much tighter than the
# unconstrained range, but still strictly above the point mass.

# %% [markdown]
# ## What would identify uncertainty about theta
#
# The equivalence is specific to conflicting test pairs. A pair of
# *agreeing* tests brings E[theta^2] into the predictions, and by
# Cauchy-Schwarz E[theta^2] E[(1 - theta)^2] >= E[theta (1 - theta)]^2 with
# equality only for a point mass — so any genuinely uncertain theta prior
# predicts strictly stronger updating from two agreeing tests than its
# observationally equivalent point mass. Models A and B agree on the
# conflicting pair and diverge on an agreeing pair (both tests naming Y1).

# %%
agreeing_tests = {"T1": "Y1", "T2": "Y1"}
for label, theta_prior in [("Model A (point mass)", point_mass_prior(fit_A["theta"])),
                           ("Model B (ridge Beta)", beta_prior(fit_B["a"], fit_B["b"]))]:
    net = make_polarization_network(CHART_PRIOR, *theta_prior)
    conflicting = predict(net, POLARIZATION_TESTS)
    agreeing = predict(net, agreeing_tests)
    print(f"{label:<21}: after conflicting pair (Y1, L1) = "
          f"{float(to_slider(conflicting['p_final'])):6.2f}, "
          f"after agreeing pair (Y1, Y1) = {float(to_slider(agreeing['p_final'])):6.2f}")

# %% [markdown]
# # Part 4: Response drift
#
# Every model so far says that all systematic change from initial to final
# rating is produced by the diagnostic evidence and theta:
#
#     Final_i = f(Initial_i, Condition_i, theta) + noise_i
#
# with f the network's prediction. The drift model adds a common term,
#
#     Final_i = f(Initial_i, Condition_i, theta) + delta + noise_i,
#
# where delta is a *condition-independent response drift*: after whatever
# Bayesian updating occurs, participants give somewhat higher or lower
# final H ratings for reasons not represented by the evidence. It separates
# evidence-driven belief change from a general shift in how participants
# respond between the two judgments:
#
#     observed change = evidence-driven Bayesian change + response drift + noise.
#
# Three clues point to a downward component. Control should show no
# Bayesian change — the symmetric class-level pair leaves P(H | evidence) =
# P(H) for every theta, point mass or Beta — yet its mean rating fell from
# 50.2 to 43.8. Model C over-predicts final H ratings in all three
# conditions, with the same negative sign (Control -6.4, Moderation -16.8,
# Polarization -11.9). And a downward drift would produce exactly the
# per-condition theta split seen in Part 2: in Polarization the Bayesian
# update pushes up while drift pushes down, so the observed polarization
# looks weaker and theta_P comes out low; in Moderation both push down, so
# moderation looks stronger and theta_M comes out high. Without drift,
# theta has to absorb everything.
#
# **The key comparison.** Two explanations of the Part 2 asymmetry each add
# one parameter to Model C: *condition-specific reliability*
# (theta_P, theta_M; evidence is treated as much stronger in Moderation than
# Polarization) versus *shared reliability plus drift* (theta, delta; the
# same reliability in both conditions, plus a general response shift). Their
# AIC comparison is therefore clean.
#
# **Estimation.** The main model estimates theta and delta jointly across all
# three conditions by maximum likelihood (given the predictions, the ML delta
# is the mean residual). As a validation, delta is instead fixed at the value
# Control alone implies — Control predicts zero Bayesian movement, so
# delta_Control = mean(Final - Initial) in Control — and only theta is fitted.
# If a drift estimated independently of Polarization and Moderation improves
# those conditions, the effect is not merely a parameter tuned to their
# outcomes. Both regimes are run for all four models; drift acts after the
# network, so the equivalence B = A and D = C survives it exactly.
#
# delta is a response drift, not a belief-update parameter: it could be
# response-scale drift, anchoring or compression, an order effect, or another
# systematic process between the two measurements. Control is what gives
# leverage to detect it.

# %%
CONTROL_DELTA = float((control["Final"] - control["Initial"]).mean())
print(f"Control-anchored drift: delta_Control = mean(Final - Initial) in Control = {CONTROL_DELTA:.2f}")

fits = {("none", "A"): fit_A, ("none", "B"): fit_B, ("none", "C"): fit_C, ("none", "D"): fit_D}
for regime, delta in [("control", CONTROL_DELTA), ("joint", "joint")]:
    print(f"\n=== drift regime: {regime} ===")
    fits[(regime, "A")] = fit_point_mass("chart", delta=delta)
    fits[(regime, "C")] = fit_point_mass("empirical", delta=delta)
    fits[(regime, "B")] = fit_beta("chart", starts=[(3.5, 1.5), (1.0, 1.0)], delta=delta)
    fits[(regime, "D")] = fit_beta("empirical", starts=[(2.0, 2.0), (0.7, 0.4)], delta=delta)
    for model in ["A", "C", "B", "D"]:
        print(f"Model {model}: ", end="")
        report(fits[(regime, model)])
    for point, beta_model in [("A", "B"), ("C", "D")]:
        assert np.isclose(fits[(regime, point)]["logLik"], fits[(regime, beta_model)]["logLik"],
                          atol=0.5), f"{point} vs {beta_model} not equivalent under {regime} drift"
print("\nEquivalence B = A and D = C holds under both drift regimes.")

# %% [markdown]
# ## Comparison: 3 regimes x 4 models

# %%
REGIME_LABEL = {"none": "no drift", "control": "Control-anchored drift", "joint": "joint drift"}
MODEL_LABEL = {"A": "A: chart, point", "B": "B: chart, Beta", "C": "C: empirical, point",
               "D": "D: empirical, Beta"}
entries = [("Null: no update (final = initial)", fit_null, "-")]
for regime in ["none", "control", "joint"]:
    for model in ["A", "B", "C", "D"]:
        f = fits[(regime, model)]
        est = describe(f) if f["family"] == "point" else \
            f"r = {f['r']:.3f} (ridge)" + (f", delta = {f['delta']:.2f}" if f["delta"] is not None else "")
        entries.append((f"{MODEL_LABEL[model]} | {REGIME_LABEL[regime]}", f, est))
    if regime == "none":  ## the competing one-extra-parameter explanation
        entries.append(("C: empirical, point | per-condition theta, no drift", fit_C_percond,
                        ", ".join(f"{c[:3]} {f['theta']:.3f}" for c, f in per_cond_C.items())))
table_drift = summary_table(entries)
print(table_drift.round(2).to_string())

# %% [markdown]
# ## Diagnostics under drift
#
# The question the drift models exist to answer: once drift is accounted
# for, do Polarization and Moderation still want different reliabilities,
# and how many participants remain out of reach?

# %%
for regime in ["none", "control", "joint"]:
    f = fits[(regime, "C")]
    print(f"Model C, {REGIME_LABEL[regime]} (delta = "
          f"{'none' if f['delta'] is None else '%.2f' % f['delta']}):")
    per_cond, _ = per_condition_thetas(f)
    print("  envelope:", ", ".join(
        f"{c} {row['outside']}/{row['n']} outside ({100 * row['share']:.0f}%; below {row['below']}, above {row['above']})"
        for c, row in envelope_counts(f["delta"]).iterrows()))

# %% [markdown]
# **Reading the envelope under drift.** A negative delta shifts every
# participant's reachable interval down. The lower side in Polarization
# (finals below the drift-adjusted no-update value) is the wrong-way count
# net of drift; the upper side grows because participants who did not drift
# now sit above a lowered envelope. The envelope is a statement about the
# mean structure only; the noise term absorbs the rest.

# %% [markdown]
# ## Figures
#
# Pre/post trajectories for the four models under joint drift (B on A, D on
# C), and Model C's initial-to-final curve under each regime.

# %%
fig = plot_prepost([
    ("A: chart, point | joint drift", trajectory(fits[("joint", "A")]), "#303030", "--"),
    ("B: chart, Beta | joint drift", trajectory(fits[("joint", "B")]), "#303030", ":"),
    ("C: empirical, point | joint drift", trajectory(fits[("joint", "C")]), "#8B3A3A", "--"),
    ("D: empirical, Beta | joint drift", trajectory(fits[("joint", "D")]), "#8B3A3A", ":"),
])

# %%
fig = plot_scatter([
    (f"C, no drift: theta = {fit_C['theta']:.2f}", curve_of(fit_C), "#303030", ":"),
    (f"C, Control-anchored drift: theta = {fits[('control', 'C')]['theta']:.2f}",
     curve_of(fits[("control", "C")]), "#8B3A3A", "--"),
    (f"C, joint drift: theta = {fits[('joint', 'C')]['theta']:.2f}",
     curve_of(fits[("joint", "C")]), "#303030", "-"),
], fits[("joint", "C")]["fitted"], "Model C, joint drift: implied finals (right margin)")

# %% [markdown]
# # Part 5: Interpretation
#
# **Participants updated in the Bayesian direction.** Every fitted model
# beats the no-update baseline by about 17 or more log-likelihood units:
# the conflicting pair moved beliefs up in Polarization, down in
# Moderation, and left Control flat, as the networks predict.
#
# **A common response drift, not condition-specific reliability, explains
# the Polarization/Moderation asymmetry.** This is the key comparison of
# Part 4, and it is decisive. The two explanations each add one parameter
# to Model C:
#
# - condition-specific reliability (theta_P = 0.32, theta_M = 0.68):
#   logLik -1251.2, AIC 2508.4;
# - shared reliability plus drift (theta = 0.45, delta = -11.8):
#   logLik -1245.6, AIC 2497.2.
#
# Shared theta plus drift wins by 11 AIC points at equal parameter count,
# and once delta is in the model the per-condition reliabilities coincide
# (0.450 vs 0.440, LR = 0.02, p = 0.90). The validation is the more
# compelling part: fixing delta at the value Control alone implies (-6.35, a
# number that never saw a Polarization or Moderation rating) still beats the
# condition-specific model (AIC 2503.2 vs 2508.4) and already removes most
# of the split (LR 16.2 -> 3.4, p = 0.07). The asymmetry was an artifact of
# an omitted downward process, exactly as the common sign of the residuals
# suggested.
#
# **Drift does not explain the conservatism.** With delta in the model the
# shared reliability barely moves (0.438 -> 0.446; r from 0.78 to 0.80).
# Participants' effective evidence weight is what it was; drift removes the
# condition asymmetry, not the weakness of the updating. Those are two
# separate findings.
#
# **The jointly estimated drift is larger than Control's.** delta = -11.8
# jointly against -6.4 in Control, so the Control-anchored validation is
# conservative, and the drift model's residuals still tilt the same way
# (Control over-shifted, Moderation under-shifted). Whether the extra shift
# in the test conditions is a larger response drift there or something the
# single-theta network misses cannot be decided from these data; the
# per-condition test says it is not a reliability difference.
#
# **Under the chart prior, "drift" is not drift.** With everyone started at
# +28, the jointly estimated delta is *positive* (+11) — it repairs the
# prior rather than modeling a downward shift — and the Control-anchored
# delta leaves the chart models no better than no update at all (Control
# predicted at 22 against an observed 44). The drift term is interpretable
# only once the model starts each participant from their own rating. That
# is the empirical prior's real contribution: without drift A and C were
# within one log-likelihood unit of each other; with drift the prior choice
# is what makes delta mean something.
#
# **The reliability-prior axis stays empty, by design.** B equals A and D
# equals C in every regime: with only conflicting test pairs, every theta
# prior is observationally equivalent to some point mass through the single
# ratio r, and drift, acting after the network, cannot break that. The
# ridge does tell us two things: the point-mass estimate is a floor on mean
# reliability beliefs, not an estimate of them — every equivalent Beta has
# a higher mean, confined by the theta >= 0.25 constraint to a band of about
# 0.1 above the point mass — and an agreeing-tests condition would separate
# point mass from uncertainty, which nothing in the present design can.
#
# **What delta is, and is not.** It is a condition-independent response
# shift between the two judgments: response-scale drift, anchoring or
# compression, an order effect, or another systematic process. It is not a
# belief-update parameter and should not yet be read as a genuine change in
# disease belief. Control is what makes it detectable at all.
#
# **What remains unexplained.** The envelope still leaves a large share of
# participants outside anything the mean structure can produce, and
# Moderation is still bimodal: one cluster barely moved, another flipped
# below zero past the perfect-reliability limit, discarding the prior rather
# than updating on it. A two-group mixture on test endorsement (the `Choice`
# column) is the natural next model; a participant-level drift is the other.
