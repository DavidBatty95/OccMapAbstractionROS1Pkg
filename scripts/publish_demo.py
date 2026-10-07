#!/usr/bin/env python3
"""Publish a recorded run and its provenance to docs; never synthesise trajectories."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--name', required=True, help='Unique publication name, e.g. reactor-20261007')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9-]*', args.name):
        parser.error('name must start with a lowercase letter and contain only letters, digits and hyphens')
    summary = json.loads((args.run / 'summary.json').read_text())
    if summary['status'] != 'recorded' or not summary['smoke_pass']:
        parser.error('Only complete recordings with passing pipeline checks can be published')
    media = ROOT / 'docs/media'
    results = ROOT / 'docs/results' / args.name
    media.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    for name in ('manifest.json', 'summary.json', 'ros.log'):
        shutil.copyfile(args.run / name, results / name)
    (results / 'metrics.csv').write_bytes((args.run / 'metrics.csv').read_bytes().replace(b'\r\n', b'\n'))
    for source, destination in [('demo.gif', args.name + '.gif'), ('live.png', args.name + '.png')]:
        shutil.copyfile(args.run / source, media / destination)
    manifest_path = results / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['published_csv_line_endings'] = 'LF (values unchanged)'
    manifest['published_artifacts_sha256'] = {
        'docs/media/' + args.name + '.gif': hashlib.sha256((media / (args.name + '.gif')).read_bytes()).hexdigest(),
        'docs/results/' + args.name + '/metrics.csv': hashlib.sha256((results / 'metrics.csv').read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print('Published', args.name, 'from', args.run)


if __name__ == '__main__':
    main()
