"""Regenerate publication artifacts without refitting models."""
from pathlib import Path
import json
from source.plotting import generate as plots
from source.diagram_generation import generate as diagrams
from source.report_generation import generate as reports
if __name__=='__main__':
    root=Path(__file__).resolve().parent
    repo=Path(json.loads((root/'config/experiment_spec.json').read_text())['repo'])
    plots(root,repo);diagrams(root,repo);reports(root,repo)
    print('Generated 20 figure categories, six diagrams and research reports.')
