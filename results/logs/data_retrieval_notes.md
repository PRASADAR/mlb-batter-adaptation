# Retrieval notes — 2026-09-21

- First attempted an ordinary Python `urllib` GET for 2024-06-01 from the workspace sandbox. DNS resolution failed because network access is restricted in that sandbox.
- Retried with approved network access and urllib's default user agent; the public server returned HTTP 403.
- The same public Statcast CSV endpoint succeeded with `curl -L -A 'Mozilla/5.0'` and approved network access. The implemented downloader uses this request style; no authentication, cookies, or nonpublic endpoint are involved.
- Representative 2024-06-01 response: 4,399 pitches, 119 columns. The schema included all eight requested bat fields. This preliminary response was used for feasibility only; its date is also included in the full frozen download.
- Representative 2026-09-20 response: 4,265 pitches, 119 columns. This verified actual 2026 availability instead of assuming that requested dates would return data. This preliminary response was used for feasibility only; its date is also included in the full frozen download.
- The Powers/Yurko GitHub repository and Rice dataset DOI were inspected as fallback sources. Fallback acquisition was unnecessary once direct MLB retrieval succeeded.
- Per-window outcomes, retry errors, exact source URLs, hashes, dates, and sizes are recorded separately in JSON manifests. Read those records and `data/raw/snapshot.json` for the actual final sample; these initial notes are not a completeness claim.
- Final acquisition completed with 2,433,934 rows across 8,271 games, 161 successful nonempty windows, three empty windows, and no failed windows. The September 21 export was empty; latest observed date is September 20. Every completed played game in the requested windows was present. All 161 partition hashes matched, and no duplicate pitch keys were removed. Schedule handling excludes cancellations and resolves duplicate postponed/rescheduled entries before counting completed games.
