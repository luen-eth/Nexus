"""Fetch pinned HF reference data, verify SHA256; weights require --with-weights."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--with-weights', action='store_true')
args = parser.parse_args()
base = Path(os.environ.get('NEXUS_REFERENCE_MODELS', Path.home() / '.cache/nexus-reference-models'))
specs = json.loads(Path(__file__).with_name('hf-models.json').read_text())

def fetch(item):
    name, spec = item
    dest = base / name
    dest.mkdir(parents=True, exist_ok=True)
    for filename, expected in spec['sha256'].items():
        if filename.endswith('.safetensors') and not args.with_weights:
            continue
        target = dest / filename
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{filename}"
            partial = target.with_suffix(target.suffix + '.part')
            with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as out:
                while data := response.read(1024 * 1024):
                    out.write(data)
            if hashlib.sha256(partial.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'{name}/{filename}: SHA256 mismatch')
            partial.replace(target)
        print(f'Verified {name}/{filename}', flush=True)
    (dest / 'reference-manifest.json').write_text(json.dumps(spec, indent=2))

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(fetch, specs.items()))
