# Public data and frozen snapshot

Source: MLB Baseball Savant [pitch-level Statcast CSV](https://baseballsavant.mlb.com/statcast_search), [field documentation](https://baseballsavant.mlb.com/csv-docs). Requests include **all recorded regular-season pitches**, including takes, so exposure counts do not condition on swinging. Each four-day request has its original URL, UTC retrieval time, response-byte SHA256, local Parquet SHA256, row count, schema, status, and any retrieval errors in `raw/statcast_START_END.json`. Exact request records are consolidated in `results/logs/data_retrieval_manifest.json`.

The target is 2023-07-14 through available 2026-09-21 data. **Requested dates are not evidence of complete coverage.** The included `raw/snapshot.json` and `results/tables/data_sample.csv` state the actual retrieved dates and sizes. Empty/failed requests are explicit in the manifest. Seasons may contain calendar days without games; the final execution date may be incomplete.

The frozen 2026-09-21 snapshot contains **2,433,934 pitch-level records across 8,271 games**, with actual dates **2023-07-14 through 2026-09-20**. This includes 7,802 automatic balls and 395 automatic strikes, which are administrative events rather than physical pitches; exposure definitions must account for that distinction. The September 21 request returned no recorded pitches. All 161 nonempty partition hashes were verified; three windows were empty and none failed. There are zero duplicate pitch identifiers after concatenation, so deduplication removes no observations.

| Season window | Pitch-level records | Played games |
|---|---:|---:|
| 2023-07-14 – 2023-10-01 | 319,018 | 1,073 |
| 2024-03-20 – 2024-09-30 | 711,899 | 2,429 |
| 2025-03-18 – 2025-09-28 | 712,528 | 2,430 |
| 2026-03-25 – 2026-09-20 | 690,489 | 2,339 |

The completed-game presence audit finds no missing played games. One 2024 scheduled game was canceled. MLB can mark canceled or postponed games as abstract state `Final`; the audit uses detailed states and retains the played occurrence of rescheduled game IDs. Hitter names are resolved for all 978 distinct batter IDs from the public MLB people API.

`raw/schedule_games.csv` independently records MLB's official game schedule and completion states at retrieval. `results/tables/data_schedule_completeness.csv` compares downloaded game IDs with completed games in each requested season window. This is a game-presence audit, not proof that every physical pitch was tracked. The analysis should use completed dates or completed games when a same-day source snapshot contains live play.

The dated partition Parquets retain every exported column and constitute the included frozen source dataset. `raw/statcast.parquet`, when created, is an optional deduplicated, date-sorted combined cache excluded from Git. Duplicate keys are resolved on `(game_pk, at_bat_number, pitch_number)`. Official corrections between requests could create conflicting duplicate rows; the first retrieved partition is retained. The frozen snapshot, rather than later mutable API output, is canonical for reproduction. These public MLB records are attributed to MLB; the repository code license does not purport to relicense MLB data.

## Reproduce acquisition and feasibility audit

```bash
python scripts/02_download_statcast.py --full-target
python scripts/01_audit_data.py
python scripts/02_download_statcast.py --players-only
python scripts/02_download_statcast.py --audit-schedule
```

The downloader caches successful and empty windows; failed windows are retried. `--force` explicitly refreshes the source. Requests use at most three concurrent workers by default and are bounded to four days to avoid the service row ceiling. An HTTP failure is never silently replaced with simulated observations.

Acquisition requires `curl` on `PATH`, outbound HTTPS access, and the project's pinned Python environment. Offline analysis uses the included frozen Parquet partitions and does not require the remote endpoint. Large combined `raw/statcast.parquet` is a convenience cache; the smaller dated partitions plus manifests provide the same full snapshot without a single oversized GitHub file.

`results/tables/data_coverage.csv` contains counts and populated percentages among all pitches, swings, whiffs, and contact swings, overall and by season, pitch type, and hitter. Swing labels include foul bunts and missed bunts in the descriptive coverage audit; the primary model excludes bunts. Contact here includes fouls and foul tips, not just balls in play. A populated bat measurement on a take does not redefine that pitch as a swing. Miss distance is expected on misses and should not be treated as zero on contact or when unavailable.

For the frozen sample, bat speed, swing length, attack angle, and attack direction are populated on 95.848% of recorded swings and 96.416% of whiffs; miss distance is populated on 96.147% of whiffs. Coverage varies by season and hitter, so the detailed audit, rather than this overall rate, governs analysis eligibility. A small number of contact-labeled pitches also contain miss distance; do not infer swing outcomes from that field's presence. Range checks are in `data_coverage_ranges.csv`; verified partition checksums are in `data_coverage_integrity.csv`.

`player_name` in this pitcher-oriented Statcast download is the pitcher name, **not the batter name**. Hitter names must be resolved separately from public MLB player identifiers.

The public [Powers/Yurko research repository](https://github.com/saberpowers/swinging-fast-and-slow) and [Rice archive](https://doi.org/10.25611/7QXV-8612) were inspected as fallback sources. The present dataset is downloaded directly from MLB; no fallback data are mixed into it.

## Final analysis files

`processed/swings_YEAR.parquet` retains the physical-pitch-derived histories on observed non-bunt swings. `processed/residuals_YEAR_MONTH.parquet` contains current/context and sequence model residuals, covariance-calibrated deviation and calibration rows. `processed/analytic_YEAR_MONTH.parquet` is the exact inference dataset, including cross-fitted nuisance residuals and exposure transforms. Monthly partitions preserve all original values and avoid GitHub's per-file size limit. `data/checksums.json` fingerprints source partitions/manifests, public name/schedule maps and every frozen processed file.

The pipeline excludes automatic-ball and automatic-strike events **before** constructing physical pitch lags and exposure/memory histories, while retaining them in the source audit. The current count state on each actual pitch is preserved, not recomputed from pitch numbers. Final counts distinguish source records from delivered-pitch records.
