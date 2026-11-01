"""CPU-first checkpoint walk: at most ONE frozen dense matrix on GPU while quantizing.

No model.from_pretrained(device_map=cuda), no persistent GPU model, no CPU NF4
quantizer. Original CUDA bnb quantization preserves its canonical byte semantics.
This is a loader primitive, not yet a complete Transformers/PEFT integration.
"""
import hashlib
import json
from pathlib import Path


def build(snapshot,consume,device=0):
    import torch
    import bitsandbytes.functional as bnb
    from safetensors import safe_open
    from solution.stream_nf4 import HostNF4
    index=json.loads((Path(snapshot)/'model.safetensors.index.json').read_text())['weight_map']
    records=[]
    for name,shard in index.items():
        with safe_open(Path(snapshot)/shard,framework='pt',device='cpu') as f:
            cpu=f.get_tensor(name)
        if '.layers.' in name and name.endswith('_proj.weight'):
            # One matrix only. Largest gate/up/down is26.25MiB FP16.
            dense=cpu.to(device=device,dtype=torch.float16)
            q,state=bnb.quantize_4bit(dense,blocksize=64,compress_statistics=True,quant_type='nf4')
            handle=HostNF4.capture(q,state)
            del dense,q,state
            consume(name,handle)
            records.append({'name':name,'shape':list(handle.state.shape),
                'packed_sha256':hashlib.sha256(handle.packed.numpy().tobytes()).hexdigest()})
        else:
            # Embedding/head MUST remain host. Norm/bias later placed selectively.
            consume(name,cpu)
        del cpu
    return records
