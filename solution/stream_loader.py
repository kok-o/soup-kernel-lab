"""Experimental pinned HF5.16/bnb0.50 CPU-first NF4 loader integration.

Installed explicitly by research drivers only. Never changes global device_map
policy outside the original weight conversion call. Host embedding/head is a
separate required integration; this prototype keeps that matrix GPU resident.
"""
import functools
import threading


def install():
    import torch
    import bitsandbytes as bnb
    from transformers import core_model_loading as loading
    from transformers import modeling_utils
    from transformers.integrations.bitsandbytes import Bnb4bitQuantize
    from .stream_nf4 import HostNF4,NF4Slots,streamed_linear
    if getattr(install,'active',False):return
    install.active=True
    scope=threading.local()
    pool=None
    original_device=loading.get_device
    original_load=modeling_utils.convert_and_load_state_dict_in_model
    original_convert=Bnb4bitQuantize.convert
    original_apply=bnb.nn.Linear4bit._apply
    original_forward=bnb.nn.Linear4bit.forward
    original_set=loading.set_param_for_module
    original_warmup=modeling_utils.caching_allocator_warmup

    def target(model,quantizer,placement):
        config=getattr(quantizer,'quantization_config',None)
        return (getattr(getattr(model,'config',None),'model_type',None)=='qwen2'
            and bool(getattr(config,'load_in_4bit',False))
            and isinstance(placement,dict)
            and any(isinstance(v,int) or str(v).startswith('cuda') for v in placement.values()))

    def host_name(name):
        return ('.layers.' in name and name.endswith('_proj.weight')) or name in ('model.embed_tokens.weight','lm_head.weight')

    def warmup(model,expanded_device_map,hf_quantizer):
        if not target(model,hf_quantizer,expanded_device_map):return original_warmup(model,expanded_device_map,hf_quantizer)
        # Correct the loader's residency estimate, preserving the allocator,
        # function body and its handling of every actually GPU-resident tensor.
        corrected={n:('cpu' if host_name(n) else v) for n,v in expanded_device_map.items()}
        return original_warmup(model,corrected,hf_quantizer)

    def set_param(model,name,param,*args,**kwargs):
        result=original_set(model,name,param,*args,**kwargs)
        if getattr(scope,'active',False) and name in ('model.embed_tokens.weight','lm_head.weight'):
            model.get_parameter(name)._stream_host_embedding=True
        return result

    for cls in (torch.nn.Embedding,torch.nn.Linear):
        prior=cls._apply
        def keep_host(self,fn,recurse=True,_prior=prior):
            def filtered(t):
                if getattr(t,'_stream_host_embedding',False):return t
                return fn(t)
            return _prior(self,filtered,recurse=recurse)
        cls._apply=keep_host

    def device(*args,**kwargs):
        if getattr(scope,'active',False):
            # Only projection weights materialize onCPU. Ordinary tiedembedding
            # remains a known445MiB floor until repairedHostHead is supplied.
            name=args[1] if len(args)>1 else ''
            if ('.layers.' in name and name.endswith('_proj.weight')) or name in ('model.embed_tokens.weight','lm_head.weight'):
                return torch.device('cpu') if kwargs.get('valid_torch_device') else 'cpu'
        return original_device(*args,**kwargs)

    @functools.wraps(original_load)
    def load(*args,**kwargs):
        model=args[0] if args else kwargs['model']
        config=args[2] if len(args)>2 else kwargs['load_config']
        scope.active=target(model,config.hf_quantizer,config.device_map)
        try:return original_load(*args,**kwargs)
        finally:scope.active=False

    @functools.wraps(original_convert)
    def convert(self,input_dict,*args,**kwargs):
        if not getattr(scope,'active',False):return original_convert(self,input_dict,*args,**kwargs)
        if not all('.layers.' in n and n.endswith('_proj.weight') for n in input_dict):
            return original_convert(self,input_dict,*args,**kwargs)
        values={name:(value[0] if isinstance(value,list) else value).to('cuda') for name,value in input_dict.items()}
        converted=original_convert(self,values,*args,**kwargs)
        result={}
        for name,param in converted.items():
            host=HostNF4.capture(param.data,param.quant_state)
            module=param.module
            new=bnb.nn.Params4bit(host.packed,requires_grad=False,quant_state=host.state,
                blocksize=param.blocksize,compress_statistics=param.compress_statistics,
                quant_type=param.quant_type,quant_storage=param.quant_storage,bnb_quantized=True,module=module)
            new._stream_nf4_host=host
            if module is not None:module.quant_state=host.state
            result[name]=new
        return result

    def apply(self,fn,recurse=True):
        weight=self._parameters.get('weight')
        if weight is None or not hasattr(weight,'_stream_nf4_host'):
            return original_apply(self,fn,recurse=recurse)
        # Device movement by HF/Accelerate/Trainer must not resurrect allweights.
        # Preserve Parameter identity, hoststatetree and versioncounter.
        del self._parameters['weight']
        try:return original_apply(self,fn,recurse=recurse)
        finally:self._parameters['weight']=weight

    @functools.wraps(original_forward)
    def forward(self,x):
        nonlocal pool
        host=getattr(self.weight,'_stream_nf4_host',None)
        if host is None:return original_forward(self,x)
        if not x.is_cuda:raise RuntimeError('streamed NF4 expects CUDA activation')
        if pool is None:pool=NF4Slots(device=x.device,slots=2)
        if not self.compute_type_is_set:self.set_compute_type(x);self.compute_type_is_set=True
        from .dtype_cast import training_cast
        dtype=x.dtype
        a=training_cast(x,self.compute_dtype) if self.compute_dtype is not None else x
        bias=None if self.bias is None else self.bias.to(dtype=a.dtype,device=x.device)
        return training_cast(streamed_linear(a,host,pool,bias),dtype)

    loading.get_device=device
    loading.set_param_for_module=set_param
    modeling_utils.convert_and_load_state_dict_in_model=load
    modeling_utils.caching_allocator_warmup=warmup
    Bnb4bitQuantize.convert=convert
    bnb.nn.Linear4bit._apply=apply
    bnb.nn.Linear4bit.forward=forward
    install.pool=lambda:pool
    import peft,sys
    original_peft=peft.get_peft_model
    @functools.wraps(original_peft)
    def get_peft(*args,**kwargs):
        result=original_peft(*args,**kwargs)
        if any(hasattr(p,'_stream_nf4_host') for p in result.parameters()):place_lora(result)
        return result
    for name,module in list(sys.modules.items()):
        if module is not None and name.split('.')[0] in ('peft','soup_cli','trl') and module.__dict__.get('get_peft_model') is original_peft:
            module.get_peft_model=get_peft


def place_lora(model):
    """PEFT initializes adapters next to CPU frozen weight; move trainables only."""
    import torch
    for module in model.modules():
        for name,param in list(module._parameters.items()):
            if param is not None and param.requires_grad and param.device.type!='cuda':
                module._parameters[name]=torch.nn.Parameter(param.to('cuda'),requires_grad=True)
    return model
