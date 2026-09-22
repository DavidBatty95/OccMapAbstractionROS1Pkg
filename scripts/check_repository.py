#!/usr/bin/env python3
"""Check local documentation targets and hashes of published evidence."""
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    for document in [ROOT / 'README.md', ROOT / 'CONTRIBUTING.md', *sorted((ROOT / 'docs').glob('*.md'))]:
        text = document.read_text()
        targets = re.findall(r'\]\(([^)]+)\)', text) + re.findall(r'(?:src|href)="([^"]+)"', text)
        for target in targets:
            url = urlsplit(target)
            if url.scheme or url.netloc or not url.path:
                continue
            path = document.parent / unquote(url.path)
            if not path.exists():
                errors.append(str(document.relative_to(ROOT)) + ': missing ' + target)
    for manifest_path in sorted((ROOT / 'docs/results').glob('*/manifest.json')):
        manifest = json.loads(manifest_path.read_text())
        for name, expected in manifest['published_artifacts_sha256'].items():
            path = ROOT / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                errors.append('Evidence hash mismatch: ' + name)
        summary = json.loads((manifest_path.parent / 'summary.json').read_text())
        if summary['run'] != manifest['run']:
            errors.append('Run/manifest mismatch: ' + str(manifest_path))
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    print('Documentation targets and published evidence hashes verified.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
