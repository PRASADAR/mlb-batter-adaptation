"""Integrity checks for the shipped research artifacts (skip before generation)."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]


def test_source_partition_checksums_and_unique_pitch_ranges():
    records=list((ROOT/'data/raw').glob('statcast_*.json'))
    if not records:pytest.skip('Snapshot not downloaded')
    for f in records:
        m=json.loads(f.read_text())
        assert m['status'] in {'ok','empty'},f
        if m['status']=='ok':
            p=ROOT/'data/raw'/m['parquet_file']
            assert hashlib.sha256(p.read_bytes()).hexdigest()==m['parquet_sha256']
            assert m['rows']<30000


def test_posterior_saved_dimensions_and_summary_reproducibility():
    p=ROOT/'results/posterior/adaptation_2025.npz'
    if not p.exists():pytest.skip('Models not run')
    from src.models.hierarchical import summarize_posterior
    with np.load(p) as archive:a=dict(archive)
    config=json.loads((ROOT/'config/analysis.json').read_text())
    assert a['lambda'].shape==(config['posterior_draws'],len(a['player_ids']))
    regenerated=summarize_posterior(a)
    summary=pd.read_csv(ROOT/'results/tables/batter_adaptation_posteriors.csv')
    summary=summary.loc[summary.season.eq(2025)].sort_values('player_id')
    regenerated=regenerated.sort_values('player_id')
    for c in ['posterior_mean','posterior_median','cri90_low','cri90_high','rank_median','prob_top_decile']:
        np.testing.assert_allclose(regenerated[c],summary[c],atol=1e-10)


def test_abstract_word_count_and_generated_sources():
    p=ROOT/'paper/abstract.md'
    if not p.exists():pytest.skip('Abstract not generated')
    text=p.read_text();sections=re.findall(r'^## (.+)$',text,re.MULTILINE)
    assert sections==['Introduction','Methods','Results','Conclusion']
    words=len(re.sub(r'^#+\s*','',text,flags=re.MULTILINE).split())
    assert words<500
    meta=json.loads((ROOT/'paper/abstract_metadata.json').read_text())
    assert words==meta['word_count_including_title_headings_author']
    sample=json.loads((ROOT/'results/logs/sample.json').read_text())
    assert f"{sample['raw_pitches']:,}" in text


def test_executed_notebook_has_no_errors():
    p=ROOT/'results/logs/notebook_execution.json'
    if not p.exists():pytest.skip('Notebook not executed yet')
    result=json.loads(p.read_text())
    assert result['errors']==0
    assert result['code_cells']==result['executed_code_cells']
    notebook=ROOT/result['notebook']
    assert hashlib.sha256(notebook.read_bytes()).hexdigest()==result['notebook_sha256']
    import nbformat
    nb=nbformat.read(notebook,as_version=4)
    cells=[c for c in nb.cells if c.cell_type=='code']
    assert [c.execution_count for c in cells]==list(range(1,len(cells)+1))
    assert not [o for c in cells for o in c.get('outputs',[]) if o.output_type=='error']


def test_pdf_matches_current_abstract_source():
    path=ROOT/'paper/abstract.pdf'
    if not path.exists():pytest.skip('PDF not generated')
    import fitz
    with fitz.open(path) as pdf:
        assert len(pdf)==1
        assert tuple(pdf[0].rect)[2:]==(612.0,792.0)
        extracted=pdf[0].get_text()
    normalize=lambda text:re.sub(r'\s+',' ',re.sub(r'^#+\s*','',text,flags=re.MULTILINE)).strip()
    assert normalize(extracted)==normalize((ROOT/'paper/abstract.md').read_text())
