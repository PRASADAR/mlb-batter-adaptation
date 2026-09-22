# Related work and limits of the contribution

Reviewed 2026-09-21. This is a targeted review, not a systematic review or proof of priority. Links below are to papers, authors' repositories, the data publisher, or conference records. Source dates refer to publication dates where available; undated documentation was inspected on the review date. The feasibility audit, not a documentation page, determines which fields this project can use.

## Closest predecessors

### Powers and Yurko: context and measurement

Scott Powers and Ronald Yurko, **“Swinging, Fast and Slow: Interpreting Variation in Baseball Swing Tracking Metrics,”** *The American Statistician*, published online 2026-04-15; [published paper](https://doi.org/10.1080/00031305.2026.2633338), [2025-07-01 preprint](https://arxiv.org/abs/2507.01238), [full preprint](https://arxiv.org/html/2507.01238v1).

They model intended bat speed and swing length conditional on count and location with a hierarchical skew-normal model containing batter intercepts and slopes. Their instrumental-variable analysis and plate-appearance Markov model examine the contact/power tradeoff. Observed speed and length are measured at contact or nearest approach, so their variation can reflect timing and measurement position rather than changed intended effort. Their preprint also flags contact depth and miss distance as potentially valuable future measurements.

**Overlap:** expected swing mechanics conditional on the current pitch and hitter, partial pooling, and careful treatment of what the measurement represents. **Distinction:** this project's exposure-conditioned deviations, negative controls, uncertain player-specific slopes and temporal persistence address a different estimand. A residual alone is not a newly identified mechanical error. We must not describe ourselves as first to contextualize bat tracking or first to estimate latent hitter characteristics.

[Author code](https://github.com/saberpowers/swinging-fast-and-slow) is an R package plus reproduction scripts and an `renv` lockfile; the repository carries GPL-3.0. [Archived data and models](https://repository.rice.edu/items/8130a55c-5be6-4195-8d23-2784a7acbb05), dated 2025, are a 1.93 GB ZIP under CC BY 4.0, DOI [10.25611/7QXV-8612](https://doi.org/10.25611/7QXV-8612). Code and data have different licenses. Referencing the method does not imply its code has been copied.

### Steinhardt and Borowiak: directly overlapping hitter-adaptation claims

Charles Steinhardt and Zach Borowiak, **“Hitter and Catcher Adaptation in Major League Baseball,”** *Baseball Research Journal*, Fall 2025, web publication 2025-12-17. [Original article](https://sabr.org/journal/article/hitter-and-catcher-adaptation-in-major-league-baseball/).

This is a particularly close predecessor. The authors use an XGBoost pitch-quality baseline and compare actual/expected run-value outcomes. They examine changes across times through the order, repeated pitch types within plate appearances, immediate repeats versus repeats after intervening pitches, and catcher effects. They interpret stronger later performance near the top of the lineup as heterogeneous adaptation, and distinguish repeated-pitch familiarity from guessing.

**Overlap:** current-pitch-adjusted performance, heterogeneous hitter adjustment, within-PA repetition, within-game familiarity, and dynamic pitcher response. **Distinction:** public bat-mechanics deviations and calibrated probabilistic comparisons are potential extensions. Their causal interpretations do not identify our estimand. We therefore cannot claim to discover hitter adaptation, adjustment heterogeneity, or the usefulness of repeated exposure. Our own findings must withstand selection, pitch changes, count, and temporal checks independently.

### Gray: sequence-dependent swing timing is established experimentally

Rob Gray, **“‘Markov at the Bat’: A Model of Cognitive Processing in Baseball Batters,”** *Psychological Science* 13(6), 542–547, 2002. [Paper record and abstract](https://pubmed.ncbi.nlm.nih.gov/12430839/), [DOI](https://doi.org/10.1111/1467-9280.00495).

Gray reports experimental evidence that prior expectations influence swing timing and models effects of pitch sequence and count with a two-state Markov model. **Overlap:** pitch history changing swing timing. **Difference:** observational MLB estimation with partial pooling and persistence testing is a different measurement task; the present work is not the first sequence-to-swing cognitive model.

## Sequencing, tunneling, and strategic response

| Primary source | What it establishes and how it constrains this project |
|---|---|
| Arnav Prasad (SSAC 2021), [“Decoding MLB Pitch Sequencing Strategies via Directed Graph Embeddings”](https://www.sloansportsconference.com/research-papers/decoding-mlb-pitch-sequencing-strategies-via-directed-graph-embeddings) | Directed sequence embeddings represent short and longer dependencies, cluster pitch-type/location patterns, and investigate setup/knockout sequences and in-game adjustment. This is direct prior work for sequence representations and dynamic pitcher behavior; it does not by itself estimate mechanical error-correction posteriors. |
| William Melville, Jesse Melville, Theo Dawson, Delma Nieves-Rivera, Christopher Archibald, David Grimsman (SSAC 2023), [“A Game Theoretical Approach to Optimal Pitch Sequencing”](https://www.sloansportsconference.com/research-papers/a-game-theoretical-approach-to-optimal-pitch-sequencing) | Models pitcher–batter interaction as a zero-sum game and studies equilibrium strategies. Our empirical exposure model is not an optimal pitching policy. Any recommendation to change a sequence requires accounting for the hitter's and pitcher's response. |
| [“Computing an Optimal Pitching Strategy in a Baseball At-Bat”](https://arxiv.org/abs/2110.04321) (preprint 2021; 2023 publication version) | Uses a zero-sum stochastic game with pitch-location uncertainty, swing outcomes, and batter patience. This supplies another strategic predecessor and highlights that observed repeated pitches are selected actions. |
| Declan Kneita (SSAC 2025), [“Transformer-Based Baseball Modeling for Pitch Outcome Prediction and Strategy Optimization”](https://www.sloansportsconference.com/research-papers/transformer-based-baseball-modeling-for-pitch-outcome-prediction-and-strategy-optimization) | The conference abstract describes pitch-outcome/hit-location prediction informed by recent batter performance and context, plus pitch selection. We inspected the conference abstract, not the full implementation. Sequence-aware prediction and adaptation to recent form are already explored; a transformer is not by itself a new latent learning-rate estimator. |
| Jeff Long, Harry Pavlidis, Martin Alonso (2018-01-31), [“Updating Pitch Tunnels”](https://www.baseballprospectus.com/news/article/37436/prospectus-feature-updating-pitch-tunnels/) | Original method description quantifies separation at release, batter decision point, and plate, including perceived separation. A simple Euclidean trajectory difference in this repository must be called a proxy, not an exact reproduction of their perceptual metric. |
| Jonathan Judge, Jeff Long, Harry Pavlidis (2017-01-25), [“Two Ways to Tunnel”](https://www.baseballprospectus.com/news/article/31040/prospectus-feature-two-ways-to-tunnel/) | Descriptive tunneling can have multiple successful manifestations; no universal scalar should automatically be treated as deception. |
| [“Contours in Batter Comparisons: A Trick Up the Sleeve”](https://www.baseballprospectus.com/news/article/32453/contours-in-batter-comparisons-a-trick-up-the-sleeve/) (2017-08-04) | Models whiff probability using second-pitch location and pitch-path differences. Outcome-linked tunneling is therefore also prior art. Incremental predictive information should be assessed after current-pitch controls and separately from causal interpretation. |

## Learning, fatigue, perception, and mechanics

**Ryan S. Brill, Sameer K. Deshpande, Abraham J. Wyner (2023), “A Bayesian analysis of the time through the order penalty in baseball.”** [Author-hosted published paper](https://ryansbrill.com/pdf/statistics_in_sports_papers/Brill_TTO_JQAS.pdf); [preprint](https://arxiv.org/abs/2210.06724). Their Bayesian multinomial model distinguishes continuous within-game change from discontinuity at a new time through the order and finds little strong discontinuity after adjustment. **Implication:** pitch number/fatigue and hitter order must be distinguished from exposure. Absence of a discontinuity is not proof that nobody learns; aggregate improvement is not proof of learning either.

**Jay Wigley (2021-04-14), “Did Batters of Long Ago Learn During a Game?”** [Original SABR study](https://sabr.org/journal/article/did-batters-of-long-ago-learn-during-a-game/). Uses historical Retrosheet outcomes to examine within-game patterns across eras and discusses David Smith's earlier work. This provides earlier baseball-learning context, but has neither modern swing measurements nor the same identification problem as mechanical residuals.

**Hiroki Nakamoto, Kazunobu Fukuhara, Taiga Torii, Ryota Takamido, David L. Mann (2022-11-29), “Optimal integration of kinematic and ball-flight information when perceiving the speed of a moving ball.”** [Original paper](https://doi.org/10.3389/fspor.2022.930295); [author institutional record](https://research.vu.nl/en/publications/optimal-integration-of-kinematic-and-ball-flight-information-when/). A virtual-environment study of university baseball batters examines integration of pitcher-motion and ball-flight cues. **Implication:** public ball trajectories omit potentially important perceptual information. This experiment is mechanistic motivation, not evidence that our observational slope measures the same cognitive process.

**Hirotaka Nakashima, Gen Horiuchi, Arata Kimura, Shinji Sakurai (2025), “Acceptable range of timing error at bat-ball impact in baseball depends on the bat swing path.”** [Original paper](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2025.1557145/full). The biomechanical link between timing tolerance and swing path means a common residual distance need not have equal performance consequences across hitters or pitches. Outcome association and component-level interpretability are required before renaming a deviation “error.”

**Maurice A. Smith, Ali Ghazizadeh, Reza Shadmehr (2006-05-23), “Interacting Adaptive Processes with Different Timescales Underlie Short-Term Motor Learning.”** [Original open-access paper](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.0040179). Reaching experiments and a two-state model distinguish strong learning/weak retention from weak learning/strong retention. **Implication:** exposure effects and forgetting can differ, but multiple latent time constants require identifying information. The baseball data do not contain the controlled perturbations and error-clamp trials used here.

**Scott T. Albert and Reza Shadmehr (2018; online 2017-11-29), “Estimating properties of the fast and slow adaptive processes during sensorimotor adaptation.”** [Author-hosted paper](https://reprints.shadmehrlab.org/Albert_Shadmehr_JNP_2018.pdf); [DOI](https://doi.org/10.1152/jn.00197.2017). Separates state and measurement noise and studies recovery of hidden learning processes; better fit to observed measurements need not imply better latent-parameter recovery. **Implication:** model selection needs recovery simulations and negative controls, in addition to predictive fit.

## Bayesian skill, rankings, and measurement uncertainty

**Andrés F. Barrientos, Deborshee Sen, Garritt L. Page, David B. Dunson (2023), “Bayesian Inferences on Uncertain Ranks and Orderings: Application to Ranking Players and Lineups.”** [Author preprint](https://arxiv.org/abs/1907.04842); [published DOI](https://doi.org/10.1214/22-BA1324). The authors emphasize that similar latent player abilities can make deterministic ordering misleading and develop uncertainty-aware summaries of order relations. **Implication:** full posterior rank and pairwise probability displays are established methodology; their application to a defensible batter-adaptation estimand is the potential contribution.

**Ralf Herbrich and Thore Graepel (2006), “TrueSkill: A Bayesian Skill Rating System.”** [Microsoft Research technical report](https://www.microsoft.com/en-us/research/?p=154591); [conference paper](https://proceedings.neurips.cc/paper/2006/file/f44ee263952e65b3610b8ba51229d1f9-Paper.pdf). Probabilistic skill ratings explicitly track uncertainty and update from competitive outcomes. This supports the general latent-skill framing but does not supply a likelihood for swing deviations or identify causal adaptation.

Our interpretation rules are consequences of the present design, not claims copied from a source:

- A random exposure slope is a conditional exposure association. It is not a fraction of remaining error corrected unless the fitted state-transition equation gives it that meaning.
- A distribution of player estimates is not evidence of persistent skill. Persistence requires a separate time split and uncertainty-aware validation.
- Simulation coverage and pairwise calibration validate the implementation under the simulation's assumptions. They do not establish empirical MLB comparison calibration.
- Posterior rank intervals and pairwise probabilities depend on the model, prior, reference-player set, and selected sample. Always state those choices.
- Plug-in expected-swing predictions and empirical-Bayes hyperparameters omit uncertainty unless explicitly propagated. Report this limitation instead of calling such intervals exact full-model Bayesian uncertainty.
- Reduced residual magnitude is not intrinsically better contact, and a signed residual can cross zero. Show the underlying mechanical components and outcome links.

## Public fields and reproducible data access

MLB's [Statcast CSV documentation](https://baseballsavant.mlb.com/csv-docs), inspected 2026-09-21, documents attack angle, attack direction, swing-path tilt, and intercept coordinates. The first two describe bat direction at impact; tilt summarizes the pre-contact swing plane; intercept coordinates are relative to the batter's center of mass. These definitions do not guarantee coverage on whiffs. The inspected page did not contain a `miss_distance` entry. Its absence from documentation does not prove absence from CSV exports. In 2026 `plate_x`/`plate_z` change from front-of-plate to middle-of-plate measurement, and `sz_top`/`sz_bot` change to ABS-defined bounds. Raw cross-year location comparisons therefore mix measurement regimes.

For this project's design, evaluate pitch trajectories at a consistent physical plane when the required kinematics are available, and treat 2026 as a separate measurement era in sensitivity analyses. A historical zone-normalization transform is not automatically comparable with ABS normalization. Document the actual implementation and missing-data fallback; never silently relabel raw 2026 coordinates as historical coordinates.

The [2025 MLB swing-metrics release](https://www.mlb.com/news/new-statcast-swing-metrics-2025) announces additional public swing-path measurements. Coverage must still be audited empirically.

| Access route | Verified scope and caveat |
|---|---|
| [Baseball Savant Statcast Search](https://baseballsavant.mlb.com/statcast_search) | Primary publisher. Download small dated slices, preserve original identifiers, record retrieval time, and inspect returned fields before scaling. |
| [Powers/Yurko Rice archive](https://repository.rice.edu/items/8130a55c-5be6-4195-8d23-2784a7acbb05) | DOI-backed frozen data/model archive. Prefer it when reproducing their historical sample. It cannot establish coverage through 2026. Archive metadata verified; download success must be logged by the pipeline. |
| [Official CSAS 2025 data repository](https://github.com/CSAS-Data-Challenge/2025) and [Kaggle challenge](https://www.kaggle.com/competitions/csas-2025-data-challenge) | Organizer repository supplies Arrow readers and directs users to the Kaggle data. This is a historical bat-tracking sample, not the requested multiseason archive. Kaggle may require an account or terms acceptance. |
| [Olubayode's CSAS/SSAC research repository](https://github.com/Olubayode/SSAC_Research_Paper) | Contains `statcast_pitch_swing_data_20240402_20240630.arrow`. Useful feasibility fallback, but its repository states CC BY-NC-ND, so do not assume permissive redistribution. Prefer original organizer/publisher provenance. |
| [Yasunori's 2024–2025 Statcast bat-tracking dataset](https://www.kaggle.com/datasets/yasunorim/mlb-statcast-bat-tracking-2024-2025) and [author's generation notebook](https://github.com/yasumorishima/kaggle-datasets/blob/main/dataset4_statcast_bat_tracking/generate.ipynb) | Dataset author documents a full 2024–2025 regular-season export with newer bat fields. This is an independently maintained mirror; verify file content, dates, licensing, and checksums before use. Claimed coverage is not substituted for our audit. |
| [sabRmetrics](https://github.com/saberpowers/sabRmetrics/tree/v1.0.4) | Author-maintained retrieval/trajectory tooling, referenced by Powers/Yurko; useful method reference. It does not guarantee the upstream endpoint is available. |
| [Retrosheet event files](https://www.retrosheet.org/game.htm) | Public event-level history can support conventional-outcome comparisons but does not replace pitch-level bat mechanics. |

## Scope of the contribution

This study provides an **audited and reproducible test of whether pitch-conditioned mechanical deviation contains repeatable player-specific exposure information**, accompanied by partial pooling, posterior comparison uncertainty, temporal validation, and falsification tests. Prior work already establishes pitch sequencing, within-game learning hypotheses, contextual swing estimation, and Bayesian skill ranking separately. The primary results do not establish persistent adaptation skill.

No source reviewed establishes that every desired new bat-tracking field is populated on whiffs, that observational exposure is exogenous, or that MLB player-specific adaptation rates are already identifiable from the proposed public sample. Those are questions for this repository's actual data and diagnostics.
