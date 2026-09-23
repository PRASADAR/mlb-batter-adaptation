#!/usr/bin/env python3
"""Build and execute the focused submission notebook."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks/adaptability_submission.ipynb"


def build():
    cells = []
    md = lambda text: cells.append(nbf.v4.new_markdown_cell(text.strip()))
    code = lambda text: cells.append(nbf.v4.new_code_cell(text.strip()))
    md(r'''# Deja Swing: Adaptation Speed and Future Contact Authority

This notebook presents the focused empirical result. A hitter's adaptation rate measures how quickly context-adjusted swing deviation declines as he accumulates same-type looks across pitchers within a game. The central external-validity test asks whether a score estimated in 2025 is associated with 2026 player qualities after controlling for the same quality in 2025.

The main finding is narrow: faster estimated adaptation has a small positive association with later batted-ball authority. The result is exploratory because the outcome family was tested after earlier contact results were inspected. Player rankings retain posterior uncertainty and should not be read as exact talent grades.''')
    code('''from pathlib import Path
import json, sys, tempfile, os
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "deja-swing-matplotlib"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image, clear_output

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "config/player_quality_outcomes.json").exists()), None)
if ROOT is None:
    raise FileNotFoundError("Open this notebook inside the mlb-batter-adaptation repository.")
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
print(f"Project: {ROOT.name} | focused offline replay")''')
    md(r'''## Measurement

The expected-swing model estimates the mechanics expected for each pitch context. The realized swing's standardized multivariate distance from that expectation is the deviation outcome. Within each hitter, repeated same-type exposure includes pitches from every pitcher already faced in the game. A positive adaptation score means deviation declines faster as exposure accumulates.

The hierarchy partially pools noisy hitter slopes. Rankings order posterior means, while posterior rate and rank intervals show how uncertain that ordering remains.''')
    code('''population = pd.read_csv(ROOT / "results/abstract_claim_audit/population_changes.csv")
display(population.loc[population.model.eq("type_fe_plus_general_exposure"),
    ["season", "outcome", "tracked_swings", "beta_deviation_per_log_dose",
     "ci95_low", "ci95_high"]])
display(Image(filename=str(ROOT / "results/abstract_claim_audit/claim_audit.png")))''')
    md(r'''The population association appears in both observed seasons and remains after accounting for repetition against the same pitcher. This establishes a repeated-exposure pattern in swing mechanics at the population level. It does not by itself identify a causal learning effect.''')
    md(r'''## Future player qualities

The validation cohort contains 261 hitters with at least 200 plate appearances and common opportunity thresholds in both periods. Twelve future qualities cover swing decisions, bat-to-ball skill, discipline, contact authority, overall production, and bat speed. Each analysis controls for the player's 2025 value of the same quality.

The five-component authority composite averages standardized hard-hit rate, barrel rate, mean exit velocity, 90th-percentile exit velocity, and expected wOBA on contact. It was defined as a contact-authority domain, without choosing the subset that produced the largest observed correlation.''')
    code('''outcomes = pd.read_csv(ROOT / "results/player_quality_outcomes/outcome_associations.csv")
composites = pd.read_csv(ROOT / "results/player_quality_outcomes/composite_associations.csv")
temporal = pd.read_csv(ROOT / "results/player_quality_outcomes/temporal_checks.csv")
display(composites)
display(outcomes[["label", "partial_rank_rho", "partial_rank_ci95_low",
    "partial_rank_ci95_high", "partial_rank_p", "bh_q", "maxT_p",
    "standardized_huber_beta", "cv_mse_gain_pct"]])
display(temporal)
display(Image(filename=str(ROOT / "results/player_quality_outcomes/quality_outcomes.png")))''')
    md(r'''The authority composite partial rank correlation is 0.131, with a 95% bootstrap interval from 0.009 to 0.249 and a permutation p-value of 0.035. OLS, Huber regression, and quantile regression point in the same general direction. Adding the score improves repeated cross-validation mean squared error by 0.53%.

The association is specific to contact authority. Swing decisions, whiff rate, strikeout rate, walk rate, and overall wOBA are null. No individual outcome survives correction across all 12 comparisons, and half-season score replications are inconclusive.''')
    md(r'''## Player rankings and curves

The table contains every estimable hitter. Point ranks order posterior means. The rate interval and posterior rank interval are the more informative summaries because many hitters have similar rates. Open the [standalone interactive explorer](adaptability_rankings.html) for a searchable and sortable view.''')
    code('''rankings = pd.read_csv(ROOT / "results/adaptability_rankings/player_rankings.csv")
curves = pd.read_csv(ROOT / "results/cross_pitcher/player_curves.csv")
curves = curves.loc[curves.model.eq("pooled_geometry")]
display(rankings.loc[rankings.season.eq(2025),
    ["point_rank", "player_name", "posterior_mean", "cri90_low", "cri90_high",
     "rank_cri90_low", "rank_cri90_high", "prob_positive", "n_exposures"]].head(25))
display(Image(filename=str(ROOT / "results/adaptability_rankings/adaptability_rankings.png")))
display(Image(filename=str(ROOT / "results/adaptability_rankings/adaptability_authority.png")))''')
    code('''try:
    import ipywidgets as widgets
    eligible = rankings.loc[rankings.season.eq(2025)].sort_values("player_name")
    hitter = widgets.Dropdown(
        options=[(r.player_name, int(r.player_id)) for r in eligible.itertuples()],
        description="Hitter:", layout=widgets.Layout(width="360px"))
    season = widgets.ToggleButtons(options=[2025, 2026], description="Season:")
    output = widgets.Output()

    def show_player(*_):
        with output:
            clear_output(wait=True)
            row = rankings.loc[rankings.player_id.eq(hitter.value) &
                                rankings.season.eq(season.value)]
            if row.empty:
                print("This hitter did not meet the estimation rules in that season.")
                return
            row = row.iloc[0]
            curve = curves.loc[curves.player_id.eq(hitter.value) &
                               curves.season.eq(season.value)].sort_values("prior_same_type_pitches")
            display(row[["player_name", "point_rank", "posterior_mean", "cri90_low",
                "cri90_high", "rank_median", "rank_cri90_low", "rank_cri90_high",
                "prob_positive", "n_exposures", "future_authority_residual"]].to_frame("value"))
            fig, ax = plt.subplots(figsize=(8, 4.2))
            ax.plot(curve.prior_same_type_pitches, curve.change_median,
                    color="#19757b", lw=2.5)
            ax.fill_between(curve.prior_same_type_pitches, curve.change_ci90_low,
                            curve.change_ci90_high, color="#19757b", alpha=.18)
            ax.axhline(0, color="#52636b", lw=1)
            ax.set(xlabel="Earlier same-type pitches in game",
                   ylabel="Change in swing deviation",
                   title=f"{row.player_name}: {season.value} fitted adaptation curve")
            ax.grid(alpha=.18)
            display(fig)
            plt.close(fig)

    hitter.observe(show_player, names="value")
    season.observe(show_player, names="value")
    display(widgets.VBox([widgets.HBox([hitter, season]), output]))
    show_player()
except ImportError:
    print("Install ipywidgets for the notebook selector. The standalone explorer remains available.")''')
    md(r'''## Interpretation

The evidence supports a candidate mechanical interpretation: hitters whose swings converge toward their context-specific expectation more quickly also tend to maintain or develop stronger contact authority in the next observed season. The association is small and is not a broad measure of offensive quality.

A new untouched season is required to establish persistence. Until then, the rankings are estimates for research and scouting follow-up, not definitive player grades.''')
    md(r'''## Reproduction

Run `make submission` from the repository root to regenerate the claim audit, player-quality analysis, rankings, notebook, and tests. The full exploratory record remains in the repository so the focused result can be audited against null and failed specifications.''')
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {"display_name": "Python (Deja Swing)", "language": "python", "name": "deja-swing"}
    nb.metadata.language_info = {"name": "python", "version": "3.12"}
    nbf.write(nb, PATH)


def execute():
    for key, rel in [("MPLCONFIGDIR", ".mplconfig"), ("JUPYTER_RUNTIME_DIR", ".jupyter/runtime"),
                     ("IPYTHONDIR", ".jupyter/ipython")]:
        os.environ[key] = str(ROOT / rel)
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    os.environ["JUPYTER_PATH"] = str(ROOT / ".venv/share/jupyter")
    notebook = nbf.read(PATH, as_version=4)
    started = time.monotonic()
    NotebookClient(notebook, timeout=600, kernel_name="deja-swing",
        resources={"metadata": {"path": str(ROOT)}}, allow_errors=False).execute()
    nbf.write(notebook, PATH)
    exporter = HTMLExporter()
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_notebook_node(notebook)
    PATH.with_suffix(".html").write_text(body)
    errors = [o for c in notebook.cells if c.cell_type == "code"
              for o in c.get("outputs", []) if o.output_type == "error"]
    if errors:
        raise AssertionError(errors)
    log = {"code_cells": sum(c.cell_type == "code" for c in notebook.cells),
        "executed_code_cells": sum(c.cell_type == "code" and c.execution_count is not None for c in notebook.cells),
        "errors": 0, "elapsed_seconds": round(time.monotonic() - started, 2),
        "notebook": str(PATH.relative_to(ROOT)),
        "notebook_sha256": hashlib.sha256(PATH.read_bytes()).hexdigest()}
    (ROOT / "results/logs/submission_notebook_execution.json").write_text(
        json.dumps(log, indent=2) + "\n")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    build()
    execute()
