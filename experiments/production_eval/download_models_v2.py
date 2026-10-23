"""Download fixed official model assets only; never execute repository code."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/research/production_eval/experiments_v2/models'
SPECS = [
    ('Qwen/Qwen3-1.7B-GGUF', '90862c4b9d2787eaed51d12237eafdfe7c5f6077',
     ['Qwen3-1.7B-Q8_0.gguf', 'README.md']),
    ('Qwen/Qwen3-Embedding-0.6B', '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3',
     ['model.safetensors', 'config.json', 'config_sentence_transformers.json',
      'modules.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json',
      'merges.txt', '1_Pooling/config.json', 'README.md']),
]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = []
    for repo, revision, filenames in SPECS:
        response = requests.get(f'https://huggingface.co/api/models/{repo}/revision/{revision}',
                                params={'blobs': 'true'}, timeout=(20, 60))
        response.raise_for_status()
        info = response.json()
        if info['sha'] != revision or info.get('gated') is not False:
            raise RuntimeError('Expected ungated pinned official model revision')
        metadata = {row['rfilename']: row for row in info['siblings']}
        for filename in filenames:
            item = metadata[filename]
            size = item.get('size')
            if not isinstance(size, int) or size <= 0:
                raise RuntimeError('Missing exact model file size')
            lfs_sha = (item.get('lfs') or {}).get('sha256')
            if filename.endswith(('.gguf', '.safetensors')) and not lfs_sha:
                raise RuntimeError('Missing upstream SHA256 for model weights')
            plan.append({'repo': repo, 'revision': revision, 'filename': filename,
                         'bytes': size, 'upstream_lfs_sha256': lfs_sha})
    total = sum(item['bytes'] for item in plan)
    if total > 4_000_000_000:
        raise RuntimeError('Exact model asset size exceeds 4 GB download bound')
    manifest = {'status': 'downloading', 'started_at': datetime.now(timezone.utc).isoformat(),
                'exact_planned_bytes': total, 'assets': [], 'remote_code_executed': False,
                'sources': [f'https://huggingface.co/{repo}' for repo, _, _ in SPECS]}
    path = OUT / 'models_manifest.json'
    print(json.dumps({'planned_bytes': total, 'assets': len(plan)}), flush=True)
    for item in plan:
        destination = OUT / item['repo'].split('/')[1] / item['filename']
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{item['repo']}/resolve/{item['revision']}/{item['filename']}"
        reused = destination.exists() and destination.stat().st_size == item['bytes']
        if reused and item['upstream_lfs_sha256']:
            reused = sha(destination) == item['upstream_lfs_sha256']
        if not reused:
            temporary = destination.with_suffix(destination.suffix + '.partial')
            count = 0
            with requests.get(url, stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                with temporary.open('wb') as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        count += len(chunk)
                        if count > item['bytes']:
                            raise RuntimeError('Model response exceeded pinned metadata size')
                        stream.write(chunk)
            if count != item['bytes']:
                raise RuntimeError('Incomplete model asset download')
            if item['upstream_lfs_sha256'] and sha(temporary) != item['upstream_lfs_sha256']:
                raise RuntimeError('Model weight SHA256 mismatch')
            temporary.replace(destination)
        record = {**item, 'local_path': str(destination), 'sha256': sha(destination),
                  'reused': reused, 'source_url': url}
        manifest['assets'].append(record)
        path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps({'downloaded': item['filename'], 'bytes': item['bytes'], 'reused': reused}), flush=True)
    manifest.update(status='complete', completed_at=datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
