"""Reproducible public Statcast download, with source URLs and hashes.

All pitches are retained: a take is still an exposure. No post-outcome filters
are sent to the service. Four-day requests remain below Savant's row ceiling.
"""
from __future__ import annotations
import hashlib
import io
import json
import subprocess
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BAT_FIELDS = ['bat_speed', 'swing_length', 'attack_angle', 'attack_direction', 'swing_path_tilt', 'intercept_ball_minus_batter_pos_x_inches', 'intercept_ball_minus_batter_pos_y_inches', 'miss_distance']
WHIFF_DESCRIPTIONS = {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'}
CONTACT_DESCRIPTIONS = {'foul', 'foul_tip', 'hit_into_play', 'foul_bunt', 'bunt_foul_tip'}
SWING_DESCRIPTIONS = WHIFF_DESCRIPTIONS | CONTACT_DESCRIPTIONS
TARGET_SPANS = [('2023-07-14','2023-10-01'), ('2024-03-20','2024-09-30'),
                ('2025-03-18','2025-09-28'), ('2026-03-25','2026-09-21')]

def source_url(start: str, end: str) -> str:
    params = dict(all='true', type='details', player_type='pitcher', hfGT='R|',
                  game_date_gt=start, game_date_lt=end, min_pitches=0,
                  min_results=0, group_by='name', sort_col='pitches', sort_order='desc')
    return 'https://baseballsavant.mlb.com/statcast_search/csv?' + urlencode(params)

def date_windows(start: str, end: str, days: int = 4):
    cursor, stop = date.fromisoformat(start), date.fromisoformat(end)
    while cursor <= stop:
        right = min(stop, cursor + timedelta(days=days - 1))
        yield cursor.isoformat(), right.isoformat()
        cursor = right + timedelta(days=1)

def fetch_window(start: str, end: str, raw_dir: Path, *, force: bool = False) -> dict:
    """Fetch one complete date window, never replace an existing valid snapshot."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f'statcast_{start}_{end}.parquet'
    manifest = path.with_suffix('.json')
    if manifest.exists() and not force:
        old = json.loads(manifest.read_text())
        if old.get('status') == 'empty' or (old.get('status') == 'ok' and path.exists()):
            return old
    url = source_url(start, end)
    record = dict(start=start, end=end, source_url=url,
                  retrieved_utc=datetime.now(timezone.utc).isoformat(),
                  source='MLB Baseball Savant Statcast CSV', requested_game_type='R')
    errors = []
    for attempt in range(3):
        try:
            proc = subprocess.run(['curl', '-f', '-sS', '-L', '-A', 'Mozilla/5.0',
                                   '--max-time', '150', url], capture_output=True, check=True)
            body = proc.stdout
            record['source_csv_sha256'] = hashlib.sha256(body).hexdigest()
            record['source_bytes'] = len(body)
            if len(body.strip()) == 0:
                data = pd.DataFrame()
            else:
                data = pd.read_csv(io.BytesIO(body), low_memory=False)
            if len(data) and not {'game_pk', 'pitch_number', 'at_bat_number', 'batter', 'description'} <= set(data.columns):
                raise ValueError('Response lacks required pitch-level identifiers')
            if len(data) >= 30000:
                raise ValueError('Response meets row ceiling; use smaller windows')
            if len(data):
                if not data.game_date.between(start, end).all():
                    raise ValueError('Response dates outside requested interval')
                if not data.game_type.eq('R').all():
                    raise ValueError('Response includes non-regular-season games')
                data = data.sort_values(['game_pk', 'at_bat_number', 'pitch_number'])
                staging = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
                data.to_parquet(staging, index=False, compression='zstd')
                staging.replace(path)
                record.update(status='ok', rows=len(data), columns=list(data.columns),
                              actual_min_date=data.game_date.min(), actual_max_date=data.game_date.max(),
                              parquet_file=path.name, parquet_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            else:
                record.update(status='empty', rows=0)
            break
        except Exception as exc:
            detail = f'{type(exc).__name__}: {exc}'
            if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
                detail += '\n' + exc.stderr.decode('utf-8',errors='replace')[-2000:]
            errors.append(detail)
            if attempt < 2:
                time.sleep(2 + attempt * 3)
    else:
        record.update(status='failed', rows=0, errors=errors)
    if errors:
        record['attempt_errors'] = errors
    staging_manifest = manifest.with_name(manifest.name + '.' + uuid.uuid4().hex + '.tmp')
    staging_manifest.write_text(json.dumps(record, indent=2) + '\n')
    staging_manifest.replace(manifest)
    return record

def load_pitches(path: str | Path | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    """Load the frozen combined snapshot or all downloaded window partitions."""
    path = Path(path) if path else ROOT / 'data/raw/statcast.parquet'
    if path.is_file():
        return pd.read_parquet(path, columns=columns)
    directory = path if path.is_dir() else path.parent
    # Ignore a partition still being written by a concurrent acquisition worker.
    files = sorted(f for f in directory.glob('statcast_*.parquet') if f.with_suffix('.json').exists())
    if not files:
        raise FileNotFoundError(f'No Statcast snapshot in {directory}; run scripts/02_download_statcast.py')
    sorting = ['game_date', 'game_pk', 'at_bat_number', 'pitch_number']
    read_columns = list(dict.fromkeys(columns + sorting)) if columns is not None else None
    data = pd.concat((pd.read_parquet(f, columns=read_columns) for f in files), ignore_index=True)
    data = data.drop_duplicates(['game_pk', 'at_bat_number', 'pitch_number']).sort_values(sorting).reset_index(drop=True)
    return data[columns] if columns is not None else data

def verify_partition_integrity(raw_dir: Path) -> pd.DataFrame:
    records=[]
    for path in sorted(raw_dir.glob('statcast_*.json')):
        manifest=json.loads(path.read_text())
        if manifest['status']=='ok':
            part=raw_dir/manifest['parquet_file']
            records.append(dict(partition=part.name,rows=manifest['rows'],
                                first_date=manifest['actual_min_date'],last_date=manifest['actual_max_date'],
                                source_csv_sha256=manifest['source_csv_sha256'],
                                parquet_sha256=manifest['parquet_sha256'],
                                sha256_verified=part.exists() and hashlib.sha256(part.read_bytes()).hexdigest()==manifest['parquet_sha256']))
    return pd.DataFrame(records)

def coverage_table(data: pd.DataFrame, group: str | None = None) -> pd.DataFrame:
    """Denominators include recorded non-tracking swings and bunt attempts."""
    data = data.copy(deep=False)
    if group == 'season':
        data = data.assign(season=data.game_date.astype(str).str[:4])
    groups = [("all", data)] if group is None else data.groupby(group, dropna=False, observed=True)
    rows = []
    for value, frame in groups:
        desc = frame.description.fillna('')
        subsets = dict(all_pitches=pd.Series(True, index=frame.index), swings=desc.isin(SWING_DESCRIPTIONS),
                       whiffs=desc.isin(WHIFF_DESCRIPTIONS), contact=desc.isin(CONTACT_DESCRIPTIONS))
        for field in BAT_FIELDS:
            present = frame[field].notna() if field in frame else pd.Series(False, index=frame.index)
            item = dict(grouping=group or 'overall', group=str(value), field=field, field_in_schema=field in frame)
            for label, mask in subsets.items():
                denominator, numerator = int(mask.sum()), int((mask & present).sum())
                item[f'n_{label}'] = denominator
                item[f'populated_{label}'] = numerator
                item[f'pct_{label}'] = 100 * numerator / denominator if denominator else np.nan
            rows.append(item)
    return pd.DataFrame(rows)

def fetch_player_names(ids, raw_dir: Path) -> pd.DataFrame:
    """Resolve hitter identifiers from MLB's public people endpoint in batches."""
    ids = sorted({int(i) for i in ids if pd.notna(i)})
    records, sources = [], []
    for offset in range(0, len(ids), 100):
        batch = ids[offset:offset + 100]
        url = 'https://statsapi.mlb.com/api/v1/people?personIds=' + ','.join(map(str, batch))
        proc = subprocess.run(['curl','-fsSL','--max-time','60',url], capture_output=True,check=True)
        payload = json.loads(proc.stdout)
        for person in payload.get('people', []):
            records.append(dict(batter=person['id'], player_name=person['fullName'],
                                birth_date=person.get('birthDate'),
                                bats=person.get('batSide',{}).get('code'),
                                throws=person.get('pitchHand',{}).get('code')))
        sources.append(dict(source_url=url, retrieved_utc=datetime.now(timezone.utc).isoformat(),
                            source_json_sha256=hashlib.sha256(proc.stdout).hexdigest()))
    players = pd.DataFrame(records).drop_duplicates('batter').sort_values('batter')
    missing = sorted(set(ids) - set(players.batter))
    players.to_csv(raw_dir/'players.csv',index=False)
    (raw_dir/'players_manifest.json').write_text(json.dumps(dict(sources=sources, missing_ids=missing,
        sha256=hashlib.sha256((raw_dir/'players.csv').read_bytes()).hexdigest()),indent=2)+'\n')
    return players

def audit_schedule(data: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    """Compare downloaded game identifiers against MLB's public final schedule."""
    rows, sources = [], []
    observed = set(data.game_pk.astype(int))
    for start, end in TARGET_SPANS:
        url = 'https://statsapi.mlb.com/api/v1/schedule?' + urlencode(dict(sportId=1, startDate=start, endDate=end, gameTypes='R'))
        proc = subprocess.run(['curl','-fsSL','--max-time','90',url], capture_output=True,check=True)
        payload = json.loads(proc.stdout)
        for day in payload.get('dates', []):
            for game in day.get('games', []):
                rows.append(dict(season=int(start[:4]), game_pk=game['gamePk'], game_date=game['officialDate'],
                                 schedule_date=day['date'],
                                 game_type=game['gameType'], abstract_state=game['status']['abstractGameState'],
                                 detailed_state=game['status']['detailedState'],
                                 downloaded=int(game['gamePk'] in observed)))
        sources.append(dict(source_url=url,retrieved_utc=datetime.now(timezone.utc).isoformat(),
                            source_json_sha256=hashlib.sha256(proc.stdout).hexdigest()))
    schedule = pd.DataFrame(rows)
    # Rescheduled games can appear once as Postponed and again as actually
    # played, under the same game_pk. MLB also labels cancellations "Final"
    # at the abstract level, so that field alone is not a played-game test.
    schedule['is_completed'] = schedule.abstract_state.eq('Final') & ~schedule.detailed_state.isin(['Cancelled','Postponed'])
    schedule = schedule.sort_values(['game_pk','is_completed','schedule_date']).drop_duplicates('game_pk',keep='last').sort_values(['season','schedule_date','game_pk'])
    schedule.to_csv(raw_dir/'schedule_games.csv',index=False)
    (raw_dir/'schedule_manifest.json').write_text(json.dumps(dict(sources=sources,
        sha256=hashlib.sha256((raw_dir/'schedule_games.csv').read_bytes()).hexdigest()),indent=2)+'\n')
    records=[]
    for season,frame in schedule.groupby('season'):
        final=frame.is_completed
        missing=frame.loc[final & frame.downloaded.eq(0),'game_pk'].tolist()
        records.append(dict(season=season,scheduled_games=len(frame),completed_games=int(final.sum()),
                            completed_games_downloaded=int((final & frame.downloaded.eq(1)).sum()),
                            missing_completed_game_ids=';'.join(map(str,missing)),
                            nonfinal_games_downloaded=int((~final & frame.downloaded.eq(1)).sum())))
    return pd.DataFrame(records)
