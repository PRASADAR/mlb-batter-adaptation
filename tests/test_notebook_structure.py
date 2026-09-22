"""The notebook is a runnable offline walkthrough, not a static mockup."""
from pathlib import Path
import ast
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_cells_compile_and_offline_default():
    notebook = nbformat.read(ROOT / 'notebooks/deja_swing.ipynb', as_version=4)
    nbformat.validate(notebook)
    code = [cell.source for cell in notebook.cells if cell.cell_type == 'code']
    for source in code:
        ast.parse(source)
    joined = '\n'.join(code)
    assert 'RUN_FULL_PIPELINE = False' in joined
    assert 'run_simulations(' in joined
    assert 'compare_hitters(' in joined
    assert 'prepare_pitches(' in joined
    assert 'requests.get(' not in joined
    assert 'fetch_window(' not in joined
    assert joined.count('show_figure(') >= 15
    assert 'residuals_2025*.parquet' in joined
    assert 'future_performance.csv' in joined


def test_notebook_distinguishes_temporal_and_synthetic_validation():
    notebook = nbformat.read(ROOT / 'notebooks/deja_swing.ipynb', as_version=4)
    text = '\n'.join(cell.source for cell in notebook.cells)
    assert 'For the original primary model, **2026 is a temporal holdout**' in text
    assert 'other games later in 2025 may train a given fold' in text
    assert 'known synthetic truth' in text
    assert 'not an identified causal learning rate' in text
    assert 'first-stage SEs' in text
    assert 'Distances for contacts remain missing' in text
    assert "use other players' 2026 outcomes" in text
    assert 'not a forecast available before 2026' in text
    assert 'Automatic balls and strikes' in text
