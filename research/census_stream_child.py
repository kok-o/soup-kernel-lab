"""Unchanged harness with scalar parameter/allocation census, no added CUDA work."""
import json,os,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'harness'))
import solution
original=solution.apply
records=[]
def apply():
    original()
    import torch,transformers
    from bitsandbytes.nn import Linear4bit
    previous=transformers.Trainer.__init__
    def record(model,stage):
        weights=[m.weight for m in model.modules() if isinstance(m,Linear4bit)]
        row={'stage':stage,'NF4_count':len(weights),'NF4_CPU':sum(w.device.type=='cpu' for w in weights),
            'GPU_parameters_bytes':sum(p.numel()*p.element_size() for p in model.parameters() if p.is_cuda),
            'CPU_trainable':sum(p.numel() for p in model.parameters() if p.requires_grad and not p.is_cuda),
            'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),
            'max_allocated':torch.cuda.max_memory_allocated(),'max_reserved':torch.cuda.max_memory_reserved()}
        records.append(row);Path(os.environ['STREAM_CENSUS_FILE']).write_text(json.dumps(records,indent=2)+'\n')
        assert row['NF4_count']==row['NF4_CPU']==196 and row['CPU_trainable']==0,row
    class Census(transformers.TrainerCallback):
        def on_step_end(self,args,state,control,model=None,**kwargs):
            if state.global_step in (1,2,64):record(model,'step'+str(state.global_step))
    def init(self,*a,**kw):
        previous(self,*a,**kw);record(self.model,'Trainer.init');self.add_callback(Census())
    transformers.Trainer.__init__=init
solution.apply=apply
runpy.run_path(str(ROOT/'harness/_train.py'),run_name='__main__')
