"""Exact fixed-shape byte budget; unknown runtime overhead stays explicit."""
import json
from pathlib import Path
H,I,V,L=1536,8960,151936,28
shapes={'q':(H,H),'k':(256,H),'v':(256,H),'o':(H,H),'gate':(I,H),'up':(I,H),'down':(H,I)}
elements=sum(n*k for n,k in shapes.values())
packed=elements//2
absmax=elements//64
nested=absmax//256*4
lora=L*(16*H+H*16+16*H+256*16)
largest=H*I
budget={'frozen_NF4_host_total_bytes_est':L*(packed+absmax+nested),
    'tied_embedding_host_FP16_bytes':V*H*2,'max_startup_dense_FP16_bytes':largest*2,
    'two_packed_slots_bytes_est':2*(largest//2+largest//64+((largest//64+255)//256)*4+4096),
    'native_dequant_dense_FP16_peak_bytes':largest*2,
    'LoRA_FP32_params_grads_two_Adam_states_bytes':lora*4*4,
    'logits_head_8192x1536_two_FP16_buffers_bytes':2*8192*H*2,
    'unknown':['CUDAcontext/libs/cuBLASworkspace','activations/offload packets/transient gradients',
        'hosthead algorithm temporaries','allocatorreservedfragmentation','loader/PEFT/Accelerate device movement'],
    'limits':'400MiB external peak NOT established. Component totals not all simultaneously live. Hosthead8k slots alone48MiB; NF4 two slots~13.5MiB; quantstartup single matrix26.25MiB. Need actualstartup andcomplete64step external peak.'}
p=Path(__file__).resolve().parents[1]/'research/STREAM_MEMORY_BUDGET.json';p.write_text(json.dumps(budget,indent=2)+'\n');print(json.dumps(budget,indent=2))
