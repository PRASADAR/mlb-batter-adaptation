"""Execute all cells, fail on errors, and create an HTML reading view."""
from pathlib import Path
import os, json, time, hashlib
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'.mplconfig')
os.environ['JUPYTER_PATH']=str(ROOT/'.venv/share/jupyter')
os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'.jupyter/runtime')
os.environ['IPYTHONDIR']=str(ROOT/'.jupyter/ipython')
for key in ['MPLCONFIGDIR','JUPYTER_RUNTIME_DIR','IPYTHONDIR']:
    Path(os.environ[key]).mkdir(parents=True,exist_ok=True)
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
p=ROOT/'notebooks/deja_swing.ipynb'
nb=nbformat.read(p,as_version=4);started=time.monotonic()
client=NotebookClient(nb,timeout=1200,kernel_name='deja-swing',resources={'metadata':{'path':str(ROOT)}},allow_errors=False)
client.execute()
nbformat.write(nb,p)
exporter=HTMLExporter();exporter.exclude_input_prompt=True;exporter.exclude_output_prompt=True
body,_=exporter.from_notebook_node(nb)
(ROOT/'notebooks/deja_swing.html').write_text(body)
errors=[o for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='error']
assert not errors
log={'code_cells':sum(c.cell_type=='code' for c in nb.cells),'executed_code_cells':sum(c.cell_type=='code' and c.execution_count is not None for c in nb.cells),'errors':len(errors),'elapsed_seconds':round(time.monotonic()-started,2),'notebook':str(p.relative_to(ROOT)),'notebook_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(ROOT/'results/logs/notebook_execution.json').write_text(json.dumps(log,indent=2)+'\n')
print(json.dumps(log,indent=2))
