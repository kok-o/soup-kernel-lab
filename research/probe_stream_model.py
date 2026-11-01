"""CPU-first model loading and one full-model FW/BW update, no fulltrain score."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'harness'))

def child(out):
    import torch
    import solution
    from solution.stream_loader import install,place_lora
    from transformers import AutoModelForCausalLM,BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training,get_peft_model,LoraConfig
    solution.apply();install()
    started=time.perf_counter()
    torch.manual_seed(1)
    model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',torch_dtype=torch.float16,
        device_map={'':'cuda:0'},quantization_config=BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True),local_files_only=True)
    loaded=time.perf_counter()
    def census():
        quant=[m for m in model.modules() if isinstance(m,__import__('bitsandbytes').nn.Linear4bit)]
        return {'quantized':len(quant),'host_quantized':sum(m.weight.device.type=='cpu' and hasattr(m.weight,'_stream_nf4_host') for m in quant),
            'GPU_parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters() if p.is_cuda),
            'trainable_CPU':sum(p.numel() for p in model.parameters() if p.requires_grad and not p.is_cuda)}
    initial=census();(out/'LOAD.json').write_text(json.dumps(initial,indent=2)+'\n')
    assert initial['quantized']==initial['host_quantized']==196,initial
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
    model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','v_proj'],bias='none',task_type='CAUSAL_LM'))
    place_lora(model);after=census();assert after['host_quantized']==196 and after['trainable_CPU']==0,after
    model.train();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0002)
    # Synthetic short batch is a lifetime/integration proof only. Fixed contest
    # data/order and64steps need an independent unchanged-harness run later.
    ids=torch.arange(64,device='cuda').reshape(1,64)%1000
    with torch.autocast('cuda',dtype=torch.float16):
        output=model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=ids.clone(),use_cache=False)
        loss=output.loss
    loss.backward();optimizer.step();torch.cuda.synchronize()
    final=census();assert final['host_quantized']==196 and final['trainable_CPU']==0
    report={'status':'MODEL_FW_BW_STREAM_PASS','load_s':loaded-started,'all_s':time.perf_counter()-started,
        'initial':initial,'after_peft':after,'final':final,'loss':float(loss.item()),'slot_bytes':install.pool().gpu_bytes(),
        'limits':'ONE syntheticshortbatchmodelFW/BW/update. Notfixedtask/fullcold/quality/400MiB/53sproof. Tiedembedding/headstillGPU.'}
    (out/'MODEL.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--child',action='store_true');a.add_argument('--out',required=True);args=a.parse_args();out=Path(args.out).resolve()
    if args.child:child(out)
    else:
        from common import OFFLINE_ENV,run_logged
        assert not out.exists();out.mkdir()
        r=run_logged([sys.executable,__file__,'--child','--out',str(out)],out/'probe.log',240,
            dict(OFFLINE_ENV,TRITON_CACHE_DIR=str(out/'triton_cache'),TORCHINDUCTOR_CACHE_DIR=str(out/'inductor_cache')))
        (out/'RUN.json').write_text(json.dumps(dict(zip(('exit_code','wall_s','peak_mib','gpu_start_mib'),r)),indent=2)+'\n')
        print(r);raise SystemExit(r[0])
