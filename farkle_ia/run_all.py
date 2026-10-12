"""
run_all.py - run every step of the project and print a summary.

Usage:
    python run_all.py            full run  (N = 20,000 etc.), output in results/
    python run_all.py --quick    small run (N = 1,000 everywhere), output in results_quick/

All random numbers come from the fixed master seed in game.py, so running
this twice gives exactly the same results.
"""

import argparse
import time

import pandas as pd

import experiments as ex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="small test run with N = 1,000 games everywhere")
    args = parser.parse_args()

    if args.quick:
        folder = "results_quick"
        n_A = n_C = n_D = n_mirror = n_matrix = n_markov = 1_000
    else:
        folder = "results"
        n_A, n_C, n_D = 20_000, 20_000, 10_000
        n_mirror, n_matrix, n_markov = 20_000, 5_000, 100_000

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    start = time.time()

    def step_done(name):
        print(f"[{time.time() - start:7.1f} s] {name} done\n")

    # Step 1
    table, bonus = ex.step1_probabilities(folder)
    print("STEP 1: probability tables (bonus rule off)")
    print(table[["n", "P_farkle", "P_score", "E0", "S_star", "S_star_hand",
                 "percent_difference", "flag_over_1_percent"]].round(4).to_string(index=False))
    print("\nOne die with the bonus rule:")
    print(bonus.round(4).to_string(index=False))
    step_done("Step 1")

    # Experiment A
    coarse, fine = ex.experiment_A(folder, n_A)
    columns = ["bonus", "k", "wins", "N", "p", "SE", "CI_low", "CI_high"]
    print("EXPERIMENT A: adaptive(k) vs break-even, coarse grid")
    print(coarse[columns].round(4).to_string(index=False))
    print("\nFine grid around the best coarse k")
    print(fine[columns].round(4).to_string(index=False))
    step_done("Experiment A")

    # Experiment B
    fits = ex.experiment_B(folder, coarse, fine)
    print("EXPERIMENT B: quadratic fit p(k) = a k^2 + b k + c")
    print(fits[["bonus", "a", "b", "c", "R_squared", "k_star", "k_star_used",
                "k_range", "note"]].round(5).to_string(index=False))
    k_star = {row["bonus"]: round(float(row["k_star_used"]), 3)
              for _, row in fits.iterrows()}
    step_done("Experiment B")

    # Experiment C
    chi = ex.experiment_C(folder, n_C, k_star)
    print("EXPERIMENT C: chi-squared test of independence (new games)")
    print(chi.round(5).to_string(index=False))
    step_done("Experiment C")

    # Experiment D
    robust = ex.experiment_D(folder, n_D)
    print("EXPERIMENT D: k* against fixed-threshold opponents")
    print(robust.round(4).to_string(index=False))
    step_done("Experiment D")

    # Experiment E
    mirror, matrix, triggers = ex.experiment_E(folder, n_mirror, n_matrix, k_star)
    print("EXPERIMENT E: game length, mirror matches")
    print(mirror.round(3).to_string(index=False))
    print(f"\nBonus-rule triggers logged in mirror matches: {len(triggers)} "
          f"({int(triggers['final_turn'].sum()) if len(triggers) else 0} on final turns)")
    step_done("Experiment E")

    # Step 5
    curve, check = ex.experiment_markov(folder, n_markov)
    print("STEP 5: exact final-turn model (selected D)")
    print(curve[curve["D"].isin([50, 500, 1000, 2000, 3000, 5000])].round(5).to_string(index=False))
    print("\nValidation against simulated final turns")
    print(check.round(5).to_string(index=False))
    step_done("Step 5")

    print(f"Total run time: {time.time() - start:.1f} s. Results saved in {folder}/")


if __name__ == "__main__":
    main()
