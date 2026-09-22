#!/usr/bin/env python3
"""Download all regular-season pitches, preserving immutable request manifests.

Examples:
  python scripts/02_download_statcast.py --start 2024-03-20 --end 2024-09-30
  python scripts/02_download_statcast.py --full-target
  python scripts/02_download_statcast.py --combine-only
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.statcast import date_windows, fetch_window, load_pitches, fetch_player_names, TARGET_SPANS, audit_schedule, verify_partition_integrity

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start', default='2024-06-01')
    p.add_argument('--end', default='2024-06-30')
    p.add_argument('--full-target', action='store_true')
    p.add_argument('--combine-only', action='store_true')
    p.add_argument('--no-combine', action='store_true')
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--days', type=int, default=4)
    p.add_argument('--force', action='store_true')
    p.add_argument('--players-only', action='store_true')
    p.add_argument('--audit-schedule', action='store_true')
    a = p.parse_args()
    raw = ROOT / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    (ROOT/'results/logs').mkdir(parents=True, exist_ok=True)
    (ROOT/'results/tables').mkdir(parents=True, exist_ok=True)
    if a.players_only:
        players = fetch_player_names(load_pitches(raw,columns=['batter']).batter.unique(), raw)
        print(f'Resolved {len(players):,} hitter names from public MLB identifiers.')
        return
    if a.audit_schedule:
        report = audit_schedule(load_pitches(raw,columns=['game_pk']), raw)
        report.to_csv(ROOT/'results/tables/data_schedule_completeness.csv',index=False)
        print(report.to_string(index=False))
        return
    if not a.combine_only:
        windows = []
        spans = [(a.start, a.end)]
        if a.full_target:
            # Fixed execution snapshot cutoff; future reruns can override dates.
            spans = TARGET_SPANS
        for start, end in spans:
            windows.extend(date_windows(start, end, a.days))
        with ThreadPoolExecutor(max_workers=max(1, min(a.workers, 4))) as pool:
            jobs = {pool.submit(fetch_window, s, e, raw, force=a.force):(s,e) for s,e in windows}
            for job in as_completed(jobs):
                r = job.result()
                print(f"{r['start']}..{r['end']}: {r['status']} {r['rows']:,} pitches", flush=True)
    manifests = [json.loads(f.read_text()) for f in sorted(raw.glob('statcast_*.json'))]
    (ROOT/'results/logs/data_retrieval_manifest.json').write_text(json.dumps(manifests, indent=2)+'\n')
    if not a.no_combine:
        integrity=verify_partition_integrity(raw)
        if len(integrity) and not integrity.sha256_verified.all():
            raise ValueError('A raw partition fails its recorded SHA256; restore the frozen source before analysis.')
        data = load_pitches(raw)
        output = raw/'statcast.parquet'
        data.to_parquet(output,index=False,compression='zstd')
        info = dict(created_utc=datetime.now(timezone.utc).isoformat(), snapshot_cutoff='2026-09-21',
                    rows=len(data), columns=len(data.columns), first_date=str(data.game_date.min()),
                    last_date=str(data.game_date.max()), games=int(data.game_pk.nunique()),
                    source_partition_rows=sum(r.get('rows',0) for r in manifests if r['status']=='ok'),
                    duplicate_pitch_keys_removed=sum(r.get('rows',0) for r in manifests if r['status']=='ok')-len(data),
                    verified_partition_sha256_count=int(integrity.sha256_verified.sum()),
                    sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    successful_windows=sum(r['status']=='ok' for r in manifests),
                    failed_windows=sum(r['status']=='failed' for r in manifests),
                    empty_windows=sum(r['status']=='empty' for r in manifests))
        if any(r['start']=='2026-09-21' and r['status']=='empty' for r in manifests):
            info['latest_requested_date_status']='2026-09-21 returned no recorded pitches at snapshot; latest actual game_date is '+str(data.game_date.max())
        (raw/'snapshot.json').write_text(json.dumps(info,indent=2)+'\n')
        print(json.dumps(info,indent=2),flush=True)
if __name__ == '__main__':
    main()
