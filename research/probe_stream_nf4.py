"""One bounded layer proof, CUDA parity/lifetime/LoRA updates, not fulltrain."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'harness'))


def child(out):
    import torch
    import bitsandbytes.functional as bnb
    from bitsandbytes.autograd._functions import MatMul4Bit
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from solution.stream_nf4 import HostNF4,NF4Slots,streamed_linear,map_state
    from _train import TritonWatch
    watch=TritonWatch(torch)
    assert watch.error is None and torch.cuda.get_device_capability()==(7,5)
    snapshot=Path(snapshot_download('Qwen/Qwen2.5-1.5B-Instruct',local_files_only=True))
    hosts=[];records=[]
    for projection in ('gate_proj','up_proj','down_proj'):
        key='model.layers.0.mlp.'+projection+'.weight'
        weight=None
        for path in snapshot.glob('*.safetensors'):
            with safe_open(path,framework='pt',device='cpu') as f:
                if key in f.keys():weight=f.get_tensor(key).to(dtype=torch.float16,device='cuda');break
        assert weight is not None
        q,state=bnb.quantize_4bit(weight,blocksize=64,compress_statistics=True,quant_type='nf4')
        host=HostNF4.capture(q,state)
        assert not host.packed.is_cuda and all(not t.is_cuda and t.is_pinned() for t,_ in host.versions)
        def sig(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        records.append({'weight':key,'packed_exact':sig(q)==sig(host.packed),'packed_sha256':sig(q),
            'no_GPU_host_members':True,'shape':list(state.shape)})
        hosts.append(host);del weight,q,state
    pool=NF4Slots(slots=2)
    # All host weights resident simultaneously, only2GPU slots, contexts outlive
    # at least3slot overwrites and then reload in reverse autograd order.
    torch.manual_seed(123)
    results=[]
    for scale in (1.,1024.,65536.):
        inputs=[torch.randn(17,h.state.shape[1],device='cuda',dtype=torch.float16)*.01 for h in hosts]
        incoming=[torch.randn(17,h.state.shape[0],device='cuda',dtype=torch.float16)*.0001 for h in hosts]
        c=[x.detach().requires_grad_() for x in inputs];s=[x.detach().requires_grad_() for x in inputs]
        baseline=[]
        # Reference GPU copies are immediately dropped after reference backward,
        # never kept alongside allcandidateweights. This diagnosticpeak includes
        # referenceinputs/outputs/grads, not a trainingpeak claim.
        for h,x,dy in zip(hosts,c,incoming):
            q=h.packed.to('cuda');state=map_state(h.state,lambda _,t:t.to('cuda'))
            y=MatMul4Bit.apply(x,q,None,None,state)
            y.backward(dy*scale);baseline.append(y.detach());del y,q,state
        values=[streamed_linear(x,h,pool) for x,h in zip(s,hosts)]
        assert all(torch.equal(a.view(torch.int16),b.view(torch.int16)) for a,b in zip(baseline,values))
        for y,dy in reversed(list(zip(values,incoming))):y.backward(dy*scale)
        assert all(torch.equal(a.grad.view(torch.int16),b.grad.view(torch.int16)) for a,b in zip(c,s))
        results.append({'scale':scale,'all3_FW_exact':True,'all3_BW_exact':True,'pool_bytes':pool.gpu_bytes()})
    # Original LoRA/dropout, including3realAdamWupdates and RNG restoration.
    h=hosts[0];x=torch.randn(17,1536,device='cuda',dtype=torch.float32)*.01
    A=torch.randn(16,1536,device='cuda')*.01;B=torch.randn(8960,16,device='cuda')*.01
    p=[torch.nn.Parameter(A.clone()),torch.nn.Parameter(B.clone())]
    q=[torch.nn.Parameter(A.clone()),torch.nn.Parameter(B.clone())]
    op=torch.optim.AdamW(p,lr=.0002);oq=torch.optim.AdamW(q,lr=.0002)
    for step in range(3):
        rng=torch.cuda.get_rng_state()
        def run(params,stream):
            xx=x.detach().requires_grad_();drop=torch.nn.functional.dropout(xx,p=.05,training=True)
            if stream:base=streamed_linear(xx.half(),h,pool)
            else:
                packed=h.packed.to('cuda');state=map_state(h.state,lambda _,t:t.to('cuda'))
                base=MatMul4Bit.apply(xx.half(),packed,None,None,state)
            delta=torch.nn.functional.linear(torch.nn.functional.linear(drop,params[0]),params[1])
            y=base.float()+(delta*2).half().float()
            (y.sum()*.001).backward();return y.detach(),xx.grad
        ref,gref=run(p,False);rng_end=torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(rng);actual,gactual=run(q,True)
        assert torch.equal(ref,actual) and torch.equal(gref,gactual)
        assert torch.equal(rng_end,torch.cuda.get_rng_state())
        assert all(torch.equal(a.grad,b.grad) for a,b in zip(p,q))
        op.step();oq.step();op.zero_grad();oq.zero_grad()
        assert all(torch.equal(a,b) for a,b in zip(p,q))
    # Mutation must fail rather than silently change backward frozenweights.
    hosts[0].packed[0].add_(1)
    try:
        with pool.acquire(hosts[0]):pass
    except RuntimeError:mutation=True
    else:raise AssertionError('hostmutationaccepted')
    result={'status':'LAYER_STREAM_PROOF_PASS','weights':records,'parity':results,'LoRA_AdamW_updates_exact':3,
        'dropout_RNG_exact':True,'host_mutation_detected':mutation,'slots':2,'pool_bytes':pool.gpu_bytes(),
        'meter_error':watch.error,'source_manifest':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('solution','research') for p in (ROOT/folder).glob('*.py')},
        'limits':'Layer proof only. CPUfirstfullmodel/PEFTintegration andcompletecoldpeak NOT proven. No fulltrain/Kaggle/record.'}
    (out/'PROOF.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--child',action='store_true');p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out).resolve()
    if a.child:child(out)
    else:
        from common import OFFLINE_ENV,run_logged
        assert not out.exists();out.mkdir()
        result=run_logged([sys.executable,__file__,'--child','--out',str(out)],out/'probe.log',180,
            dict(OFFLINE_ENV,TRITON_CACHE_DIR=str(out/'triton_cache'),TORCHINDUCTOR_CACHE_DIR=str(out/'inductor_cache')))
        (out/'RUN.json').write_text(json.dumps(dict(zip(('exit_code','wall_s','peak_mib','gpu_start_mib'),result)),indent=2)+'\n')
        print(result);raise SystemExit(result[0])
