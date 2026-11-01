"""Bounded frozen NF4 host storage. Experimental; deliberately not installed by apply()."""
import copy
import threading
from contextlib import contextmanager
from dataclasses import dataclass

VIEW_CACHE_LIMIT=512


def map_state(state, transform):
    """Clone the QuantState tree without mutating shared original metadata."""
    import torch
    result=copy.copy(state)
    for name,value in vars(state).items():
        if isinstance(value,torch.Tensor):setattr(result,name,transform(name,value))
        elif name=='state2' and value is not None:setattr(result,name,map_state(value,lambda key,t:transform('state2.'+key,t)))
    return result


@dataclass(frozen=True)
class HostNF4:
    packed: object
    state: object
    versions: tuple
    slab: object
    layout: object

    @classmethod
    def capture(cls,packed,state):
        """D2H complete before publication; no GPU tensor survives in handle."""
        import torch
        tensors=[];layout={};size=0
        def host(name,value):
            nonlocal size
            cpu=value.detach().to('cpu').contiguous()
            size=(size+255)&~255
            layout[name]=(size,cpu.numel()*cpu.element_size(),tuple(cpu.shape),cpu.dtype)
            tensors.append((name,cpu));size+=layout[name][1]
            return cpu
        data=host('packed',packed)
        meta=map_state(state,host)
        slab=torch.zeros((size+255)&~255,dtype=torch.uint8,pin_memory=True)
        def view(name):
            offset,n,shape,dtype=layout[name]
            return slab[offset:offset+n].view(dtype).view(shape)
        for name,t in tensors:view(name).copy_(t)
        meta=map_state(meta,lambda name,_:view(name))
        return cls(view('packed'),meta,((slab,slab._version),),slab,layout)

    def check(self):
        if any(t._version!=version for t,version in self.versions):
            raise RuntimeError('frozen host NF4 bytes modified after capture')


class NF4Slots:
    """Fixed slot count, bounded capacities, ready AND last-reader events.

    No GPU slot/view may be stored in an autograd context. Lock prevents leases
    from overlapping on the host; GPU reuse is ordered by the read_done event.
    """
    def __init__(self,device=0,slots=2,max_elements=8960*1536):
        import torch
        self.device=device;self.count=slots;self.max_elements=max_elements
        self.lock=threading.Lock();self.next=0;self.copy=torch.cuda.Stream(device=device)
        self.slots=[{'buffers':{},'done':None,'views':{}} for _ in range(slots)]
        self.stats={'view_hits':0,'view_builds':0,'view_uncached':0}

    @contextmanager
    def acquire(self,host):
        import torch
        host.check()
        with self.lock,torch.cuda.device(self.device):
            slot=self.slots[self.next];self.next=(self.next+1)%self.count
            compute=torch.cuda.current_stream(self.device)
            with torch.cuda.stream(self.copy):
                if slot['done'] is not None:self.copy.wait_event(slot['done'])
                limit=self.max_elements//2+self.max_elements//64+self.max_elements//4096+8192
                if host.slab.numel()>limit:raise ValueError('weight exceeds slot capacity')
                key=('slab',torch.uint8)
                existing=slot['buffers'].get(key)
                if existing is None:
                    existing=torch.empty(limit,device=self.device,dtype=torch.uint8)
                    slot['buffers'][key]=existing
                existing[:host.slab.numel()].copy_(host.slab,non_blocking=True)
                entry=slot['views'].get(id(host))
                if entry is not None and entry[0] is host and entry[1] is existing:
                    data,state=entry[2],entry[3]
                    self.stats['view_hits']+=1
                else:
                    def stage(name,value):
                        offset,n,shape,dtype=host.layout[name]
                        return existing[offset:offset+n].view(dtype).view(shape)
                    data=stage('packed',host.packed)
                    state=map_state(host.state,stage)
                    self.stats['view_builds']+=1
                    # CPU object cache only: every Tensor aliases this slot's
                    # existing slab. No extra GPU storage or weight reuse.
                    if len(slot['views'])<VIEW_CACHE_LIMIT or id(host) in slot['views']:
                        slot['views'][id(host)]=(host,existing,data,state)
                    else:self.stats['view_uncached']+=1
                ready=self.copy.record_event()
            compute.wait_event(ready)
            existing.record_stream(compute)
            try:yield data,state
            finally:
                slot['done']=compute.record_event()

    def gpu_bytes(self):
        return sum(t.numel()*t.element_size() for s in self.slots for t in s['buffers'].values())


def streamed_linear(x,host,pool,bias=None):
    """Frozen native bnb FW and original dX BW, with CPU-only autograd state."""
    import torch
    from bitsandbytes.autograd._functions import MatMul4Bit
    import bitsandbytes.functional as F

    class Stream(torch.autograd.Function):
        @staticmethod
        def forward(ctx,a,b):
            ctx.host=host;ctx.pool=pool;ctx.bias_dtype=None if b is None else b.dtype
            ctx.need_bias=b is not None and b.requires_grad
            # Native dispatch is delegated, but its temporary context must NOT
            # retain a packed GPU view through the whole model forward.
            with pool.acquire(host) as (packed,state):
                result=MatMul4Bit.apply(a,packed,None,b,state)
            return result

        @staticmethod
        def backward(ctx,dy):
            db=dy.sum(0,dtype=ctx.bias_dtype) if ctx.need_bias else None
            dx=None
            if ctx.needs_input_grad[0]:
                with ctx.pool.acquire(ctx.host) as (packed,state):
                    dx=torch.matmul(dy,F.dequantize_4bit(packed,state).to(dy.dtype))
            return dx,db
    if x.numel()==0:raise ValueError('empty input requires native fallback before streaming')
    return Stream.apply(x,bias)
