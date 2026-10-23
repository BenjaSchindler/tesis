"""Read-only resource and API model inventory. Never prints credentials."""
from pathlib import Path
import json
import platform
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    import torch
    from dotenv import load_dotenv
    from google import genai
    import os
    load_dotenv(ROOT / '.env')
    result = {
        'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_devices': [],
        'paid_generation_calls': 0,
    }
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        result['cuda_devices'].append({'index':i,'name':p.name,'total_memory_bytes':p.total_memory})
    query = subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv'], capture_output=True, text=True)
    result['nvidia_smi'] = {'exit_code': query.returncode, 'output': query.stdout.strip()}
    key = os.environ.get('GOOGLE_API_KEY')
    if key:
        try:
            client = genai.Client(api_key=key)
            models = list(client.models.list())
            result['google_model_inventory'] = [
                {'name':m.name, 'supported_actions':getattr(m,'supported_actions',None), 'input_token_limit':getattr(m,'input_token_limit',None),'output_token_limit':getattr(m,'output_token_limit',None)}
                for m in models if 'flash' in (m.name or '') or 'pro' in (m.name or '')
            ]
        except Exception as exc:
            result['google_inventory_error']={'type':type(exc).__name__,'code':getattr(exc,'code',None)}
    else:
        result['google_inventory_error']={'type':'MissingConfiguredKey'}
    dest = ROOT/'output/research/production_eval/execution/resource_probe.json'
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
