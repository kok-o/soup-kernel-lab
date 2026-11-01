"""Real native NF4 parity and storage aliasing; executed by Kaggle's full suite."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_native_nf4_slot_view_cache_parity_alias_and_mutation():
    import bitsandbytes.functional as bnb
    from bitsandbytes.autograd._functions import MatMul4Bit
    from solution.stream_nf4 import HostNF4,NF4Slots,streamed_linear,map_state,VIEW_CACHE_LIMIT

    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(93271)
        shapes=((96,64),(64,128),(128,96))
        native=[];hosts=[]
        for shape in shapes:
            weight=torch.randn(shape,device='cuda',dtype=torch.float16)*.01
            packed,state=bnb.quantize_4bit(weight,blocksize=64,compress_statistics=True,quant_type='nf4')
            native.append((packed,state));hosts.append(HostNF4.capture(packed,state))
        pool=NF4Slots(max_elements=max(n*k for n,k in shapes))
        for scale in (1.,1024.,65536.):
            inputs=[torch.randn(17,k,device='cuda',dtype=torch.float16)*.01 for n,k in shapes]
            incoming=[torch.randn(17,n,device='cuda',dtype=torch.float16)*.0001 for n,k in shapes]
            reference=[];reference_grads=[]
            for x,dy,(packed,state) in zip(inputs,incoming,native):
                xx=x.detach().requires_grad_();yy=MatMul4Bit.apply(xx,packed,None,None,state)
                yy.backward(dy*scale);reference.append(yy.detach());reference_grads.append(xx.grad)
            actual_inputs=[x.detach().requires_grad_() for x in inputs]
            outputs=[streamed_linear(x,h,pool) for x,h in zip(actual_inputs,hosts)]
            for yy,dy in reversed(list(zip(outputs,incoming))):yy.backward(dy*scale)
            assert all(torch.equal(a.view(torch.int16),b.view(torch.int16)) for a,b in zip(reference,outputs))
            assert all(torch.equal(a.view(torch.int16),b.grad.view(torch.int16)) for a,b in zip(reference_grads,actual_inputs))
        assert pool.stats=={'view_hits':12,'view_builds':6,'view_uncached':0}
        pointers=set()
        for slot in pool.slots:
            assert len(slot['buffers'])==1 and len(slot['views'])==3<=VIEW_CACHE_LIMIT
            slab=slot['buffers'][('slab',torch.uint8)]
            pointers.add(slab.untyped_storage().data_ptr())
            for host,buffer,packed,state in slot['views'].values():
                assert buffer is slab
                def alias(name,tensor):
                    offset,n,shape,dtype=host.layout[name]
                    assert tensor.untyped_storage().data_ptr()==slab.untyped_storage().data_ptr()
                    assert tensor.data_ptr()==slab.data_ptr()+offset
                    assert tuple(tensor.shape)==shape and tensor.dtype==dtype
                    return tensor
                alias('packed',packed);map_state(state,alias)
        limit=pool.max_elements//2+pool.max_elements//64+pool.max_elements//4096+8192
        assert len(pointers)==2 and pool.gpu_bytes()==2*limit
        # Nested state shares the same host slab version, including on cache hits.
        before=dict(pool.stats)
        hosts[0].state.state2.code[0].add_(1)
        with pytest.raises(RuntimeError,match='modified'):
            with pool.acquire(hosts[0]):pass
        assert pool.stats==before
