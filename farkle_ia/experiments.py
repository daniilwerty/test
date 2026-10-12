"""
experiments.py - Step 4 and the Step 5 checks: all the experiments.

Each experiment function saves its results as CSV files and its plots as
PNG files in an output folder, and returns the main numbers so run_all.py
can print a summary.

    experiment_A_coarse  win rate of adaptive(k) on the coarse grid
    experiment_B         quadratic fit to the whole coarse grid and the
                         decision rule for k* (choose_k_star)
    experiment_A_fine    finer grid around k*, only if k* is inside [0, 3]
    experiment_C  chi-squared test on brand new games
    experiment_D  robustness against fixed-threshold opponents
    experiment_E  game length (mirror matches and a full matrix)
    experiment_markov  exact final-turn curves and simulation check
"""

import math
import os

import matplotlib
matplotlib.use("Agg")            # draw plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from game import DiceRoller, GameSettings, make_rng, play_game, play_turn
from markov import win_probability
from probabilities import bonus_table, probability_table
from strategies import Adaptive, BreakEven, FixedThreshold

# The two bonus settings that every experiment is run with.
# "on" means the default bonus mode, "chain".
BONUS_SETTINGS = {"off": "off", "on": "chain"}

COARSE_K_GRID = [round(0.25 * i, 2) for i in range(13)]   # 0, 0.25, ..., 3
FIXED_T_VALUES = [300, 500, 1000, 2000]
K_MIN, K_MAX = 0.0, 3.0          # the range of k that is searched
E_ADAPTIVE_K = 0.5               # k of the adaptive player in Experiment E (both bonus settings)
Z_95 = 1.959964   # z value for a 95% confidence interval


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def save_csv(table, folder, filename):
    os.makedirs(folder, exist_ok=True)
    table.to_csv(os.path.join(folder, filename), index=False)


def save_plot(figure, folder, filename):
    os.makedirs(folder, exist_ok=True)
    figure.savefig(os.path.join(folder, filename), dpi=150, bbox_inches="tight")
    plt.close(figure)


def make_settings(bonus_label):
    """Game settings for bonus 'off' or 'on' (on = chain)."""
    return GameSettings(bonus_mode=BONUS_SETTINGS[bonus_label])


def make_strategy(kind, value, settings):
    """
    Build a strategy from a short description:
        ("break-even", None), ("adaptive", k) or ("fixed", T).
    The settings tell the strategy which S*_n values to use.
    """
    if kind == "break-even":
        return BreakEven(settings.bonus_mode, settings.bonus_aware)
    if kind == "adaptive":
        return Adaptive(value, settings.bonus_mode, settings.bonus_aware)
    if kind == "fixed":
        return FixedThreshold(value)
    raise ValueError(kind)


def play_match(strategy_a, strategy_b, n_games, settings, label):
    """
    Play n_games between player A and player B, alternating who starts
    (A starts the even-numbered games, B the odd-numbered ones).

    Returns three tables: one row per game, one row per bonus-rule trigger,
    and one row per very large rho value.
    """
    roller = DiceRoller(make_rng(label))
    bonus_log = []
    large_rho_log = []
    games = []
    for game_id in range(n_games):
        a_starts = (game_id % 2 == 0)
        games.append(play_game(strategy_a, strategy_b, a_starts, settings, roller,
                               bonus_log, large_rho_log, game_id))
    return pd.DataFrame(games), pd.DataFrame(bonus_log), pd.DataFrame(large_rho_log)


def win_rate_summary(games):
    """
    Player A's win rate p with its standard error and 95% confidence interval.
        SE = sqrt(p (1 - p) / N)
        CI = p +/- 1.96 x SE
    """
    n = len(games)
    wins = int((games["winner"] == "A").sum())
    p = wins / n
    se = math.sqrt(p * (1 - p) / n)
    return {"wins": wins, "N": n, "p": p, "SE": se,
            "CI_low": p - Z_95 * se, "CI_high": p + Z_95 * se}


def k_sweep(k_values, opponent_kind, opponent_value, n_games, bonus_label, tag):
    """
    Play adaptive(k) (player A) against one opponent (player B) for every k.
    Returns one row per k with the win-rate summary.
    """
    settings = make_settings(bonus_label)
    opponent = make_strategy(opponent_kind, opponent_value, settings)
    rows = []
    for k in k_values:
        player = make_strategy("adaptive", k, settings)
        label = f"{tag} bonus={bonus_label} opponent={opponent.name} k={k:g}"
        games, bonus_log, large_rho_log = play_match(player, opponent, n_games,
                                                     settings, label)
        row = {"k": k, "bonus": bonus_label, "opponent": opponent.name}
        row.update(win_rate_summary(games))
        row["mean_turns"] = games["turns"].mean()
        row["bonus_triggers"] = len(bonus_log)
        row["large_rho_turns"] = len(large_rho_log)
        row["max_rho"] = large_rho_log["rho"].max() if len(large_rho_log) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def fit_quadratic(k_values, win_rates):
    """
    Least-squares fit of p(k) = a k^2 + b k + c.

    Returns a, b, c, R^2 and the vertex -b / (2a) (the turning point of the
    parabola). The vertex is NaN if a = 0.

    R^2 = 1 - (sum of squared residuals) / (total sum of squares).
    """
    k = np.asarray(k_values, dtype=float)
    p = np.asarray(win_rates, dtype=float)
    a, b, c = np.polyfit(k, p, deg=2)          # least squares
    predicted = a * k ** 2 + b * k + c
    ss_residual = float(((p - predicted) ** 2).sum())
    ss_total = float(((p - p.mean()) ** 2).sum())
    r_squared = 1 - ss_residual / ss_total if ss_total > 0 else np.nan
    vertex = -b / (2 * a) if a != 0 else np.nan
    return {"a": a, "b": b, "c": c, "R_squared": r_squared, "vertex": vertex}


def choose_k_star(k_values, win_rates):
    """
    The decision rule for k*, fixed before the full run:

    1. Fit p(k) = a k^2 + b k + c to the whole coarse grid (k = 0 to 3).
    2. If a < 0 and the vertex -b/(2a) lies in [0, 3], then k* = -b/(2a).
       This is "case 2" (a maximum inside the range).
    3. Otherwise the maximum of the fitted curve on [0, 3] is at an end
       point, so k* is 0 or 3, whichever has the larger FITTED value
       p(0) = c or p(3) = 9a + 3b + c. This is "case 3". If the two are
       exactly equal, k* = 0 is chosen (the simpler strategy).
    """
    fit = fit_quadratic(k_values, win_rates)
    a, b, c, vertex = fit["a"], fit["b"], fit["c"], fit["vertex"]

    def fitted_p(k):
        return a * k ** 2 + b * k + c

    if a < 0 and K_MIN <= vertex <= K_MAX:
        case = 2
        k_star = vertex
        reason = "a < 0 and vertex inside [0, 3]"
    else:
        case = 3
        k_star = K_MIN if fitted_p(K_MIN) >= fitted_p(K_MAX) else K_MAX
        if a >= 0:
            reason = "a >= 0, so the curve has no maximum"
        else:
            reason = f"vertex {vertex:.3f} is outside [0, 3]"
        reason += f"; fitted p(0) = {fitted_p(K_MIN):.4f}, p(3) = {fitted_p(K_MAX):.4f}"

    fit.update({
        "case": case,
        "case_description": f"case {case}: {reason}",
        "k_star": k_star,
        "fitted_p_at_k_star": fitted_p(k_star),
        "fitted_p_at_0": fitted_p(K_MIN),
        "fitted_p_at_3": fitted_p(K_MAX),
    })
    return fit


# ---------------------------------------------------------------------------
# Step 1: probability tables
# ---------------------------------------------------------------------------

def step1_probabilities(folder):
    table = probability_table()
    bonus = bonus_table()
    save_csv(table, folder, "step1_probability_table.csv")
    save_csv(bonus, folder, "step1_one_die_bonus.csv")
    return table, bonus


# ---------------------------------------------------------------------------
# Experiment A (coarse grid) and B (curve fit and decision rule for k*)
# ---------------------------------------------------------------------------

def experiment_A_coarse(folder, n_games):
    """Adaptive(k) vs break-even for k = 0, 0.25, ..., 3, bonus off and on."""
    coarse_all = pd.concat(
        [k_sweep(COARSE_K_GRID, "break-even", None, n_games, bonus_label, "A coarse")
         for bonus_label in BONUS_SETTINGS],
        ignore_index=True)
    save_csv(coarse_all, folder, "A_coarse_grid.csv")
    return coarse_all


def experiment_B(folder, coarse_all):
    """
    Apply the decision rule (choose_k_star) to the whole coarse grid, for
    bonus off and on. Returns one row per bonus setting with a, b, c, R^2,
    the vertex, which case happened, and k*.
    """
    fits = []
    for bonus_label in BONUS_SETTINGS:
        coarse = coarse_all[coarse_all["bonus"] == bonus_label]
        fit = choose_k_star(coarse["k"], coarse["p"])
        fit["bonus"] = bonus_label
        fit["points_used"] = len(coarse)
        fits.append(fit)
    fits = pd.DataFrame(fits)
    save_csv(fits, folder, "B_quadratic_fit_and_k_star.csv")
    return fits


def experiment_A_fine(folder, n_games, fits):
    """
    Fine grid (steps of 0.05), only for a bonus setting where case 2
    happened. It is centred on k* rounded to the nearest 0.05, runs from
    centre - 0.25 to centre + 0.25, and is clipped to [0, 3]. New games.
    """
    tables = []
    for _, fit in fits.iterrows():
        if fit["case"] != 2:
            continue
        centre = round(fit["k_star"] / 0.05) * 0.05
        fine_grid = [round(centre + 0.05 * step, 2) for step in range(-5, 6)]
        fine_grid = [abs(k) for k in fine_grid if K_MIN - 1e-9 <= k <= K_MAX + 1e-9]
        fine = k_sweep(fine_grid, "break-even", None, n_games, fit["bonus"], "A fine")
        fine["centre"] = centre
        tables.append(fine)

    if tables:
        fine_all = pd.concat(tables, ignore_index=True)
    else:
        fine_all = pd.DataFrame(columns=["k", "bonus", "p", "SE"])
    save_csv(fine_all, folder, "A_fine_grid.csv")
    return fine_all


def plot_B(folder, coarse_all, fine_all, fits):
    """Win rate against k with 95% error bars and the fitted curve on [0, 3]."""
    for _, fit in fits.iterrows():
        bonus_label = fit["bonus"]
        coarse = coarse_all[coarse_all["bonus"] == bonus_label]
        fine = fine_all[fine_all["bonus"] == bonus_label]

        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.errorbar(coarse["k"], coarse["p"], yerr=Z_95 * coarse["SE"], fmt="o",
                      color="tab:blue", capsize=3, label="coarse grid (95% CI)")
        if len(fine):
            axis.errorbar(fine["k"], fine["p"], yerr=Z_95 * fine["SE"], fmt="s",
                          color="tab:orange", markersize=4, capsize=2,
                          label="fine grid (95% CI)")
        k_line = np.linspace(K_MIN, K_MAX, 300)
        axis.plot(k_line, fit["a"] * k_line ** 2 + fit["b"] * k_line + fit["c"],
                  color="tab:red",
                  label=f"fit: {fit['a']:.4f}k² {fit['b']:+.4f}k {fit['c']:+.4f}"
                        f"  (R² = {fit['R_squared']:.3f})")
        axis.axvline(fit["k_star"], color="tab:red", linestyle=":",
                     label=f"k* = {fit['k_star']:.3f} (case {fit['case']})")
        axis.axhline(0.5, color="black", linewidth=0.8)
        axis.set_xlabel("k")
        axis.set_ylabel("win rate of adaptive(k) vs break-even")
        axis.set_title(f"Win rate against k (bonus rule {bonus_label})")
        axis.legend(fontsize=8)
        save_plot(figure, folder, f"B_win_rate_vs_k_bonus_{bonus_label}.png")


def k_to_test(fit, coarse):
    """
    Which k Experiment C uses for the adaptive player.

    If k* > 0, it is k*. If k* = 0, adaptive(k*) would be the break-even
    player itself, so the best observed coarse k is used instead, to test
    "does adapting help?". If the best observed coarse k is also 0, the best
    observed coarse k above 0 is used.
    """
    k_star = float(fit["k_star"])
    if k_star > 0:
        return round(k_star, 3), f"k* = {k_star:.3f} from the decision rule ({fit['case_description']})"
    best_k = float(coarse.loc[coarse["p"].idxmax(), "k"])
    if best_k > 0:
        which = f"the best observed coarse k ({best_k:g})"
    else:
        above_zero = coarse[coarse["k"] > 0]
        best_k = float(above_zero.loc[above_zero["p"].idxmax(), "k"])
        which = (f"the best observed coarse k above 0 ({best_k:g}), because the "
                 f"best observed coarse k was 0 itself,")
    return best_k, (f"k* = 0 ({fit['case_description']}). So {which} "
                    f"is tested against k = 0")


# ---------------------------------------------------------------------------
# Experiment C: chi-squared test of independence on new games
# ---------------------------------------------------------------------------

def experiment_C(folder, n_games, test_k_by_bonus, description_by_bonus):
    """
    New games: adaptive(k) vs break-even, and break-even vs break-even,
    where k is chosen by k_to_test().
    2x2 table:            player A wins   player A loses
        adaptive vs BE         ...             ...
        BE vs BE               ...             ...
    Chi-squared test of independence (no Yates continuity correction, the
    same as a GDC). Degrees of freedom = (2 - 1)(2 - 1) = 1.
    """
    rows = []
    tables = []
    for bonus_label in BONUS_SETTINGS:
        settings = make_settings(bonus_label)
        k_test = test_k_by_bonus[bonus_label]
        adaptive = make_strategy("adaptive", k_test, settings)
        break_even = make_strategy("break-even", None, settings)

        games_1, _, _ = play_match(adaptive, break_even, n_games, settings,
                                   f"C bonus={bonus_label} adaptive vs break-even")
        games_2, _, _ = play_match(break_even, break_even, n_games, settings,
                                   f"C bonus={bonus_label} break-even vs break-even")

        wins_1 = int((games_1["winner"] == "A").sum())
        wins_2 = int((games_2["winner"] == "A").sum())
        observed = np.array([[wins_1, n_games - wins_1],
                             [wins_2, n_games - wins_2]])
        chi2, p_value, dof, expected = stats.chi2_contingency(observed, correction=False)

        tables.append(pd.DataFrame({
            "bonus": bonus_label,
            "matchup": [f"{adaptive.name} vs break-even (k=0)", "break-even vs break-even"],
            "A_wins": observed[:, 0], "A_losses": observed[:, 1],
            "expected_A_wins": expected[:, 0], "expected_A_losses": expected[:, 1],
        }))
        rows.append({"bonus": bonus_label, "k_tested": k_test,
                     "what_is_tested": description_by_bonus[bonus_label],
                     "p_adaptive": wins_1 / n_games, "p_break_even_mirror": wins_2 / n_games,
                     "chi2": chi2, "dof": dof, "p_value": p_value,
                     "significant_at_5_percent": p_value < 0.05})

    results = pd.DataFrame(rows)
    save_csv(pd.concat(tables, ignore_index=True), folder, "C_contingency_tables.csv")
    save_csv(results, folder, "C_chi_squared.csv")
    return results


# ---------------------------------------------------------------------------
# Experiment D: robustness against fixed-threshold opponents
# ---------------------------------------------------------------------------

def experiment_D(folder, n_games):
    """
    Coarse k grid against fixed(T) for T = 300, 500, 1000, 2000, bonus off
    and on. k* for each opponent uses the same decision rule as Experiment B.

    The summary has two rows per opponent:
        "k = 0 (break-even rule)": observed win rate of break-even vs fixed(T).
            How far this is above 0.5 is the advantage of the break-even
            rule itself.
        "k = k* (adapting)": the decision-rule k* and its fitted win rate.
            gain_from_adapting = fitted p(k*) - fitted p(0) is the extra
            advantage of adapting to rho.
    """
    sweeps = []
    summary = []
    for bonus_label in BONUS_SETTINGS:
        for T in FIXED_T_VALUES:
            sweep = k_sweep(COARSE_K_GRID, "fixed", T, n_games, bonus_label, "D")
            sweeps.append(sweep)
            fit = choose_k_star(sweep["k"], sweep["p"])
            at_zero = sweep[sweep["k"] == 0].iloc[0]
            best = sweep.loc[sweep["p"].idxmax()]
            common = {"bonus": bonus_label, "opponent": f"fixed(T={T})"}

            summary.append({**common, "row": "k = 0 (break-even rule)", "k": 0.0,
                            "p_observed": at_zero["p"], "CI_low": at_zero["CI_low"],
                            "CI_high": at_zero["CI_high"],
                            "p_fitted": fit["fitted_p_at_0"],
                            "break_even_edge_over_0.5": at_zero["p"] - 0.5,
                            "gain_from_adapting": np.nan, "case": ""})

            # Observed values for k* only exist if k* is a grid point.
            on_grid = sweep[np.isclose(sweep["k"], fit["k_star"])]
            summary.append({**common, "row": "k = k* (adapting)", "k": fit["k_star"],
                            "p_observed": on_grid["p"].iloc[0] if len(on_grid) else np.nan,
                            "CI_low": on_grid["CI_low"].iloc[0] if len(on_grid) else np.nan,
                            "CI_high": on_grid["CI_high"].iloc[0] if len(on_grid) else np.nan,
                            "p_fitted": fit["fitted_p_at_k_star"],
                            "break_even_edge_over_0.5": np.nan,
                            "gain_from_adapting": fit["fitted_p_at_k_star"] - fit["fitted_p_at_0"],
                            "case": fit["case_description"]})
            summary[-1]["best_observed_k"] = best["k"]
            summary[-1]["best_observed_p"] = best["p"]

    sweeps = pd.concat(sweeps, ignore_index=True)
    summary = pd.DataFrame(summary)
    save_csv(sweeps, folder, "D_grid_vs_fixed.csv")
    save_csv(summary, folder, "D_k_star_summary.csv")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, bonus_label in zip(axes, BONUS_SETTINGS):
        for T in FIXED_T_VALUES:
            part = sweeps[(sweeps["bonus"] == bonus_label) &
                          (sweeps["opponent"] == f"fixed(T={T})")]
            axis.errorbar(part["k"], part["p"], yerr=Z_95 * part["SE"], marker="o",
                          capsize=2, label=f"vs fixed T = {T}")
        axis.set_title(f"Bonus rule {bonus_label}")
        axis.set_xlabel("k")
        axis.legend(fontsize=8)
    axes[0].set_ylabel("win rate of adaptive(k)")
    save_plot(figure, folder, "D_win_rate_vs_k_fixed_opponents.png")
    return summary


# ---------------------------------------------------------------------------
# Experiment E: game length
# ---------------------------------------------------------------------------

def strategy_list(adaptive_k):
    """
    The six strategies compared in Experiment E. adaptive_k is E_ADAPTIVE_K,
    the same k for bonus off and on, so the "on - off" change for the
    adaptive row compares the same strategy.
    """
    return ([("break-even", None)] + [("fixed", T) for T in FIXED_T_VALUES]
            + [("adaptive", adaptive_k)])


def length_summary(games):
    return {"N": len(games),
            "mean_turns": games["turns"].mean(), "sd_turns": games["turns"].std(),
            "mean_rounds": games["rounds"].mean(), "sd_rounds": games["rounds"].std(),
            "mean_trigger_round": games["trigger_round"].mean()}


def experiment_E(folder, n_mirror, n_matrix, adaptive_k=E_ADAPTIVE_K):
    """
    Main design: mirror matches (both players use the same strategy).
    Secondary design: every strategy against every other one.
    Both with the bonus rule off and on. Also collects every bonus-rule
    trigger from the mirror matches for the rho histogram.
    """
    mirror_rows = []
    matrix_rows = []
    all_triggers = []
    all_large_rho = []

    for bonus_label in BONUS_SETTINGS:
        settings = make_settings(bonus_label)
        specs = strategy_list(adaptive_k)
        players = [make_strategy(kind, value, settings) for kind, value in specs]

        # Main design: mirror matches.
        for player in players:
            label = f"E mirror bonus={bonus_label} {player.name}"
            games, triggers, large_rho = play_match(player, player, n_mirror,
                                                    settings, label)
            row = {"bonus": bonus_label, "strategy": player.name}
            row.update(length_summary(games))
            row["bonus_triggers_per_game"] = len(triggers) / len(games)
            mirror_rows.append(row)
            if len(triggers):
                triggers["bonus"] = bonus_label
                triggers["match"] = player.name
                all_triggers.append(triggers)
            if len(large_rho):
                large_rho["bonus"] = bonus_label
                all_large_rho.append(large_rho)

        # Secondary design: every pairing (each unordered pair once; with
        # alternating starts, X vs Y and Y vs X are the same experiment).
        for i, player_1 in enumerate(players):
            for player_2 in players[i:]:
                label = f"E matrix bonus={bonus_label} {player_1.name} vs {player_2.name}"
                games, _, _ = play_match(player_1, player_2, n_matrix, settings, label)
                row = {"bonus": bonus_label, "strategy_A": player_1.name,
                       "strategy_B": player_2.name}
                row.update(length_summary(games))
                row["win_rate_A"] = (games["winner"] == "A").mean()
                matrix_rows.append(row)

    mirror = pd.DataFrame(mirror_rows)
    matrix = pd.DataFrame(matrix_rows)
    triggers = pd.concat(all_triggers, ignore_index=True) if all_triggers else pd.DataFrame()
    large_rho = pd.concat(all_large_rho, ignore_index=True) if all_large_rho else pd.DataFrame()

    save_csv(mirror, folder, "E_mirror_game_length.csv")
    save_csv(matrix, folder, "E_matrix_game_length_long.csv")
    for bonus_label in BONUS_SETTINGS:
        part = matrix[matrix["bonus"] == bonus_label]
        # Fill both halves of the square table (the matrix is symmetric).
        flipped = part.rename(columns={"strategy_A": "strategy_B", "strategy_B": "strategy_A"})
        square = (pd.concat([part, flipped]).drop_duplicates(["strategy_A", "strategy_B"])
                  .pivot(index="strategy_A", columns="strategy_B", values="mean_turns"))
        order = list(mirror[mirror["bonus"] == bonus_label]["strategy"])
        square = square.loc[order, order].reset_index()
        save_csv(square, folder, f"E_matrix_mean_turns_bonus_{bonus_label}.csv")
    save_csv(triggers, folder, "E_bonus_triggers.csv")
    save_csv(large_rho, folder, "E_large_rho_log.csv")

    _plot_game_length(folder, mirror)
    _plot_rho_histogram(folder, triggers)
    return mirror, matrix, triggers


def _plot_game_length(folder, mirror):
    """Mean game length per strategy (off vs on), and the change on - off."""
    off = mirror[mirror["bonus"] == "off"].reset_index(drop=True)
    on = mirror[mirror["bonus"] == "on"].reset_index(drop=True)
    # Names line up because both lists are built in the same order.
    names = list(off["strategy"])
    x = np.arange(len(names))

    se_off = off["sd_turns"] / np.sqrt(off["N"])
    se_on = on["sd_turns"] / np.sqrt(on["N"])
    change = on["mean_turns"] - off["mean_turns"]
    se_change = np.sqrt(se_off ** 2 + se_on ** 2)

    figure, (axis_1, axis_2) = plt.subplots(1, 2, figsize=(12, 4.5))
    axis_1.bar(x - 0.2, off["mean_turns"], 0.4, yerr=Z_95 * se_off, capsize=3,
               label="bonus off")
    axis_1.bar(x + 0.2, on["mean_turns"], 0.4, yerr=Z_95 * se_on, capsize=3,
               label="bonus on (chain)")
    axis_1.set_xticks(x, names, rotation=30, ha="right")
    axis_1.set_ylabel("mean game length (turns)")
    axis_1.set_title("Mean game length, mirror matches (95% CI)")
    axis_1.legend()

    axis_2.bar(x, change, yerr=Z_95 * se_change, capsize=3, color="tab:green")
    axis_2.axhline(0, color="black", linewidth=0.8)
    axis_2.set_xticks(x, names, rotation=30, ha="right")
    axis_2.set_ylabel("change in mean length (turns), on - off")
    axis_2.set_title("Effect of the bonus rule on game length (95% CI)")
    save_plot(figure, folder, "E_game_length_by_strategy.png")


def _plot_rho_histogram(folder, triggers):
    """
    Histogram of rho at each bonus-rule trigger.

    On a final turn the opponent has already reached G, so rho = (G-B)/(G-L)
    has a zero or negative denominator and is undefined (NaN). Final-turn
    triggers are therefore shown in a second panel, using the deficit D
    (points the chaser needs) instead of rho.
    """
    if triggers.empty:
        return
    normal = triggers[~triggers["final_turn"]]
    final = triggers[triggers["final_turn"]]

    figure, (axis_1, axis_2) = plt.subplots(1, 2, figsize=(12, 4.5))
    rho = normal["rho"].dropna()
    rho_shown = rho[rho <= 5]
    axis_1.hist(rho_shown, bins=50, color="tab:blue")
    axis_1.set_xlabel("rho at trigger")
    axis_1.set_ylabel("number of triggers")
    axis_1.set_title(f"Normal turns: {len(rho)} triggers "
                     f"({len(rho) - len(rho_shown)} with rho > 5 not shown)")

    axis_2.hist(final["deficit"], bins=40, color="tab:orange")
    axis_2.set_xlabel("deficit D at trigger (rho undefined on final turns)")
    axis_2.set_title(f"Final turns: {len(final)} triggers")
    save_plot(figure, folder, "E_rho_histogram_bonus_triggers.png")


# ---------------------------------------------------------------------------
# Step 5: exact final-turn model and its check
# ---------------------------------------------------------------------------

def experiment_markov(folder, n_validation):
    """
    Exact P(win final turn) for D = 50 to 5000 (step 50), bonus off and on,
    then compare with simulated final turns at D = 500, 1000 and 2000.
    """
    deficits = list(range(50, 5001, 50))
    curve = pd.DataFrame({"D": deficits})
    for bonus_label, mode in BONUS_SETTINGS.items():
        # Going up in D means each value reuses the smaller ones already stored.
        curve[f"P_win_bonus_{bonus_label}"] = [win_probability(D, 6, mode) for D in deficits]
    save_csv(curve, folder, "markov_final_turn_curve.csv")

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for bonus_label in BONUS_SETTINGS:
        axis.plot(curve["D"], curve[f"P_win_bonus_{bonus_label}"],
                  label=f"bonus {bonus_label}")
    axis.set_xlabel("deficit D (points the chaser must beat)")
    axis.set_ylabel("P(chaser wins final turn)")
    axis.set_title("Exact final-turn winning probability")
    axis.legend()
    save_plot(figure, folder, "markov_final_turn_curve.png")

    rows = []
    for bonus_label in BONUS_SETTINGS:
        settings = make_settings(bonus_label)
        chaser = BreakEven(settings.bonus_mode)   # the strategy does not matter on a final turn
        for D in (500, 1000, 2000):
            roller = DiceRoller(make_rng(f"markov check bonus={bonus_label} D={D}"))
            wins = 0
            for _ in range(n_validation):
                # Own banked score 0, opponent on D: the chaser needs S > D.
                if play_turn(chaser, 0, D, True, settings, roller) > 0:
                    wins += 1
            p_sim = wins / n_validation
            se = math.sqrt(p_sim * (1 - p_sim) / n_validation)
            exact = win_probability(D, 6, settings.bonus_mode)
            z = (p_sim - exact) / se if se > 0 else np.nan
            rows.append({"bonus": bonus_label, "D": D, "exact": exact,
                         "simulated": p_sim, "N": n_validation, "SE": se,
                         "difference": p_sim - exact, "z_score": z,
                         "within_95_CI": abs(z) < Z_95})
    check = pd.DataFrame(rows)
    save_csv(check, folder, "markov_validation.csv")
    return curve, check
