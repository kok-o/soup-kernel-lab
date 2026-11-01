"""Offload quantized saved-activation packets and lossless raw activations to pinned host memory.

Transfers use stream-ordered D2H with reusable pinned host buffers and double-buffered
H2D prefetching into compute-stream-allocated staging slots.
"""

import weakref

_PINNED_SLOTS = []
_PINNED_LAYER_SLABS = [[] for _ in range(28)]
_HOST_SLAB_BYTES = 48 * 2**20
_H2D_STREAMS = {}
_LAYER_EVENTS = [None] * 28
_SLOT_FREE_EVENTS = [None, None]
_SLOT_DONE_EVENTS = [None, None]
_LAYER_ITEMS = [[] for _ in range(28)]
_LAYER_BYTES = [0] * 28
_LAYER_PREFETCHED = [False] * 28
_SLOT_ITEMS = [[], []]
_GPU_STAGING = [None, None]


class _Offloaded:
    __slots__ = (
        "packet", "device", "event", "prefetched_packed", "prefetched_scales",
        "prefetch_event", "__weakref__"
    )

    def __init__(self, packet, device, event):
        self.packet, self.device, self.event = packet, device, event
        self.prefetched_packed = None
        self.prefetched_scales = None
        self.prefetch_event = None


class _RawOffloaded:
    __slots__ = (
        "host_tensor", "shape", "dtype", "device", "event", "original", "version",
        "prefetched_tensor", "prefetch_event", "__weakref__"
    )

    def __init__(self, host_tensor, shape, dtype, device, event, original, version):
        self.host_tensor = host_tensor
        self.shape, self.dtype = shape, dtype
        self.device, self.event = device, event
        self.original, self.version = original, version
        self.prefetched_tensor = None
        self.prefetch_event = None


def _get_h2d_stream(device):
    import torch

    idx = device.index if device.index is not None else torch.cuda.current_device()
    stream = _H2D_STREAMS.get(idx)
    if stream is None:
        with torch.cuda.device(idx):
            stream = torch.cuda.Stream(device=device)
        _H2D_STREAMS[idx] = stream
    return stream


def _get_layer_event(layer):
    import torch

    ev = _LAYER_EVENTS[layer]
    if ev is None:
        ev = torch.cuda.Event()
        _LAYER_EVENTS[layer] = ev
    return ev


def _alloc_pinned(slot_idx, shape, dtype, nbytes):
    import torch

    aligned = (max(nbytes, 1) + 262143) & ~262143
    if slot_idx < len(_PINNED_SLOTS):
        buf, ref = _PINNED_SLOTS[slot_idx]
        if buf.numel() >= nbytes and (ref is None or ref() is None):
            view = buf[:nbytes].view(dtype).reshape(shape)
            _PINNED_SLOTS[slot_idx] = (buf, weakref.ref(view))
            return view
    buf = torch.empty(max(aligned, 262144), dtype=torch.uint8, device="cpu", pin_memory=True)
    view = buf[:nbytes].view(dtype).reshape(shape)
    if slot_idx < len(_PINNED_SLOTS):
        _PINNED_SLOTS[slot_idx] = (buf, weakref.ref(view))
    else:
        _PINNED_SLOTS.append((buf, weakref.ref(view)))
    return view


def _alloc_layer_pinned(layer, allocations, shape, dtype, nbytes):
    """Place saved host views in aligned slabs, without reusing live views.

    A fresh hooks closure is one forward generation. If a previous graph still
    owns any view, its slab remains leased. Large layers safely use more slabs.
    This changes host storage only; GPU staging capacities remain unchanged.
    """
    import torch

    size = (max(nbytes, 1) + 255) & ~255
    chunks = allocations.setdefault(layer, [])
    if not chunks or chunks[-1][1] + size > chunks[-1][0]['buffer'].numel():
        entry = next((e for e in _PINNED_LAYER_SLABS[layer]
                      if e['buffer'].numel() >= size and not any(r() is not None for r in e['views'])), None)
        if entry is None:
            capacity = (max(_HOST_SLAB_BYTES, size) + 65535) & ~65535
            entry = {'buffer': torch.empty(capacity, dtype=torch.uint8, device='cpu', pin_memory=True), 'views': []}
            _PINNED_LAYER_SLABS[layer].append(entry)
        else:
            entry['views'].clear()
        chunks.append([entry, 0])
    entry, offset = chunks[-1]
    view = entry['buffer'][offset:offset+nbytes].view(dtype).reshape(shape)
    chunks[-1][1] += size
    entry['views'].append(weakref.ref(view))
    return view


def _copy_prefetch_views(buffer, views):
    """Copy adjacent host-slab views as byte ranges into existing GPU storage."""
    import torch

    runs = []
    for item, first, second in views:
        sources = ((item.packet.packed, first), (item.packet.scales, second)) if second is not None else ((item.host_tensor, first),)
        for host, target in sources:
            storage = host.untyped_storage()
            start = host.storage_offset() * host.element_size()
            destination = target.storage_offset() * target.element_size()
            nbytes = host.numel() * host.element_size()
            if (runs and runs[-1][0].data_ptr() == storage.data_ptr()
                    and start-runs[-1][1] == destination-runs[-1][2]
                    and start >= runs[-1][1]+runs[-1][3]):
                runs[-1][3] = start+nbytes-runs[-1][1]
            else:
                runs.append([storage, start, destination, nbytes])
    for storage, start, destination, nbytes in runs:
        # Alias the pinned storage; the CPU tensor below owns no new data.
        host_bytes = torch.empty(0, dtype=torch.uint8, device='cpu').set_(
            storage, start, (nbytes,), (1,))
        buffer[destination:destination+nbytes].copy_(host_bytes, non_blocking=True)


def _clear_slot(slot):
    for item in _SLOT_ITEMS[slot]:
        if isinstance(item, _Offloaded):
            item.prefetched_packed = None
            item.prefetched_scales = None
            item.prefetch_event = None
        else:
            item.prefetched_tensor = None
            item.prefetch_event = None
    _SLOT_ITEMS[slot].clear()


def reset_offload_step():
    clear_staging()
    for i in range(28):
        _LAYER_ITEMS[i].clear()
        _LAYER_BYTES[i] = 0
        _LAYER_PREFETCHED[i] = False


def clear_staging():
    import torch

    _clear_slot(0)
    _clear_slot(1)
    for slot in (0, 1):
        buf = _GPU_STAGING[slot]
        if buf is not None:
            h2d = _H2D_STREAMS.get(buf.device.index or torch.cuda.current_device())
            if h2d is not None:
                buf.record_stream(h2d)
            _GPU_STAGING[slot] = None


def sync_d2h():
    pass


def prefetch_layer(layer_idx):
    import torch

    if layer_idx is None or not (0 <= layer_idx < 28):
        return
    if _LAYER_PREFETCHED[layer_idx] or not _LAYER_ITEMS[layer_idx]:
        return
    _LAYER_PREFETCHED[layer_idx] = True
    items = _LAYER_ITEMS[layer_idx]
    first = items[0]
    device = first.device
    slot = layer_idx % 2
    _clear_slot(slot)
    max_bytes = max(_LAYER_BYTES)
    needed = (max(max_bytes, _LAYER_BYTES[layer_idx]) + 65535) & ~65535
    with torch.cuda.device(device):
        compute_stream = torch.cuda.current_stream(device)
        buf = _GPU_STAGING[slot]
        if buf is None or buf.numel() < needed or buf.device != device:
            buf = torch.empty(needed, dtype=torch.uint8, device=device)
            _GPU_STAGING[slot] = buf
        slot_free = _SLOT_FREE_EVENTS[slot]
        if slot_free is None:
            slot_free = torch.cuda.Event()
            _SLOT_FREE_EVENTS[slot] = slot_free
        slot_free.record(compute_stream)
        h2d_stream = _get_h2d_stream(device)
        h2d_stream.wait_event(slot_free)
        layer_ev = _LAYER_EVENTS[layer_idx]
        if layer_ev is not None:
            h2d_stream.wait_event(layer_ev)
        offset = 0
        views = []
        for item in items:
            if isinstance(item, _Offloaded):
                hp, hs = item.packet.packed, item.packet.scales
                nb_p = hp.numel() * hp.element_size()
                vp = buf[offset:offset + nb_p].view(hp.dtype).reshape(hp.shape)
                offset = (offset + nb_p + 255) & ~255
                nb_s = hs.numel() * hs.element_size()
                vs = buf[offset:offset + nb_s].view(hs.dtype).reshape(hs.shape)
                offset = (offset + nb_s + 255) & ~255
                views.append((item, vp, vs))
            else:
                ht = item.host_tensor
                nb = ht.numel() * ht.element_size()
                vt = buf[offset:offset + nb].view(ht.dtype).reshape(ht.shape)
                offset = (offset + nb + 255) & ~255
                views.append((item, vt, None))
        done = _SLOT_DONE_EVENTS[slot]
        if done is None:
            done = torch.cuda.Event()
            _SLOT_DONE_EVENTS[slot] = done
        with torch.cuda.stream(h2d_stream):
            _copy_prefetch_views(buf, views)
            done.record(h2d_stream)
        for item, v1, v2 in views:
            if v2 is not None:
                item.prefetched_packed = v1
                item.prefetched_scales = v2
            else:
                item.prefetched_tensor = v1
            item.prefetch_event = done
            _SLOT_ITEMS[slot].append(item)


def offload_packed_hooks(
    pack, unpack, *, min_bytes=262144, should_offload=None, layer_getter=None, offload_raw=False,
):
    import torch

    from .activation_storage import _Packet

    wrappers = {}
    raw_wrappers = {}
    slot_cursor = 0
    layer_allocations = {}

    def _d2h_copy_packet(packet, layer):
        nonlocal slot_cursor
        device = packet.packed.device
        nb_p = packet.packed.numel() * packet.packed.element_size()
        nb_s = packet.scales.numel() * packet.scales.element_size()
        with torch.cuda.device(device):
            stream = torch.cuda.current_stream(device)
            if layer is not None and 0 <= layer < 28:
                host_packed = _alloc_layer_pinned(layer, layer_allocations, packet.packed.shape, packet.packed.dtype, nb_p)
                slot_cursor += 1
                host_scales = _alloc_layer_pinned(layer, layer_allocations, packet.scales.shape, packet.scales.dtype, nb_s)
                slot_cursor += 1
                ready = _get_layer_event(layer)
            else:
                host_packed = torch.empty_like(packet.packed, device="cpu", pin_memory=True)
                host_scales = torch.empty_like(packet.scales, device="cpu", pin_memory=True)
                ready = torch.cuda.Event()
            host_packed.copy_(packet.packed, non_blocking=True)
            host_scales.copy_(packet.scales, non_blocking=True)
            packet.packed.record_stream(stream)
            packet.scales.record_stream(stream)
            ready.record(stream)
        packet.packed, packet.scales = host_packed, host_scales
        result = _Offloaded(packet, device, ready)
        if layer is not None and 0 <= layer < 28:
            _LAYER_ITEMS[layer].append(result)
            _LAYER_BYTES[layer] += ((nb_p + 255) & ~255) + ((nb_s + 255) & ~255)
        return result

    def _d2h_copy_raw(tensor, layer):
        nonlocal slot_cursor
        device = tensor.device
        nb = tensor.numel() * tensor.element_size()
        host_tensor = _alloc_layer_pinned(layer, layer_allocations, tensor.shape, tensor.dtype, nb)
        slot_cursor += 1
        with torch.cuda.device(device):
            stream = torch.cuda.current_stream(device)
            host_tensor.copy_(tensor, non_blocking=True)
            tensor.record_stream(stream)
            ready = _get_layer_event(layer)
            ready.record(stream)
        version_source = tensor._base if tensor._base is not None else tensor
        result = _RawOffloaded(
            host_tensor, tensor.shape, tensor.dtype, device, ready,
            weakref.ref(version_source), tensor._version,
        )
        if 0 <= layer < 28:
            _LAYER_ITEMS[layer].append(result)
            _LAYER_BYTES[layer] += (nb + 255) & ~255
        return result

    def offload(tensor):
        packet = pack(tensor)
        layer = layer_getter() if layer_getter is not None else None
        if not isinstance(packet, _Packet):
            if (not offload_raw or layer is None or not (0 <= layer < 28)
                    or should_offload is not None and not should_offload()):
                return packet
            is_param = isinstance(tensor, torch.nn.Parameter) or (
                tensor._base is not None and isinstance(tensor._base, torch.nn.Parameter)
            )
            if not (isinstance(tensor, torch.Tensor) and tensor.is_cuda
                    and tensor.layout == torch.strided and tensor.is_contiguous()
                    and not is_param
                    and tensor.dtype in (torch.float16, torch.float32, torch.uint8)
                    and tensor.numel() * tensor.element_size() >= 4096
                    and ((tensor.requires_grad and not tensor.is_leaf)
                         or tensor.ndim == 4
                         or (tensor.dtype == torch.uint8 and tensor.ndim == 1)
                         or (tensor.ndim == 3 and tensor.shape[1] == 6))):
                return packet
            key = id(tensor)
            prev = raw_wrappers.get(key)
            if prev is not None:
                orig_ref, ver, res_ref = prev
                res = res_ref()
                if orig_ref() is tensor and ver == tensor._version and res is not None:
                    return res
            res = _d2h_copy_raw(tensor, layer)
            raw_wrappers[key] = (weakref.ref(tensor), tensor._version, weakref.ref(res))
            return res
        cached = wrappers.get(id(packet))
        previous = cached() if cached is not None else None
        if previous is not None:
            return previous
        if (packet.packed.numel() * packet.packed.element_size() < min_bytes
                or should_offload is not None and not should_offload()):
            wrappers[id(packet)] = weakref.ref(packet)
            return packet
        result = _d2h_copy_packet(packet, layer)
        wrappers[id(packet)] = weakref.ref(result)
        return result

    def restore(value):
        if isinstance(value, _RawOffloaded):
            original = value.original()
            if original is not None and original._version != value.version:
                raise RuntimeError("a packed saved activation was modified in-place")
            with torch.cuda.device(value.device):
                if value.prefetched_tensor is not None:
                    torch.cuda.current_stream(value.device).wait_event(value.prefetch_event)
                    return value.prefetched_tensor
                torch.cuda.current_stream(value.device).wait_event(value.event)
                return value.host_tensor.to(value.device, non_blocking=True)
        if not isinstance(value, _Offloaded):
            return unpack(value)
        packet = value.packet
        cached = packet.decoded() if packet.decoded is not None else None
        if cached is not None:
            return unpack(packet)
        host_packed, host_scales = packet.packed, packet.scales
        with torch.cuda.device(value.device):
            if value.prefetched_packed is not None:
                torch.cuda.current_stream(value.device).wait_event(value.prefetch_event)
                packet.packed = value.prefetched_packed
                packet.scales = value.prefetched_scales
            else:
                torch.cuda.current_stream(value.device).wait_event(value.event)
                packet.packed = host_packed.to(value.device, non_blocking=True)
                packet.scales = host_scales.to(value.device, non_blocking=True)
            try:
                return unpack(packet)
            finally:
                packet.packed, packet.scales = host_packed, host_scales

    return offload, restore
