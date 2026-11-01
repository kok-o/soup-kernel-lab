"""Reuse pinned activation storage and exactly two compressed GPU staging slots.

D2H stays ordered on the producer stream, allowing immediate GPU allocator reuse.
Only H2D uses a copy stream. A slot waits for the previous decoder read before
being overwritten. Host leases wait for their last H2D read before reuse.
No device synchronization or allocator configuration change is used in training.
"""

import weakref

_managers = {}


class _Lease:
    def __init__(self, packed_count, scale_count):
        import torch

        self.packed = torch.empty(packed_count, dtype=torch.uint8, pin_memory=True)
        self.scales = torch.empty(scale_count, dtype=torch.float32, pin_memory=True)
        self.owner = None
        self.dependency = None


class _Transfer:
    def __init__(self, packet, device, ready, lease):
        self.packet, self.device, self.ready, self.lease = packet, device, ready, lease
        self.slot = None
        self.consumed = False


class _Slot:
    def __init__(self, device):
        import torch

        # Fixed task upper bound B=4, T=512, intermediate width=8960, INT4/64.
        self.packed = torch.empty(4 * 512 * 8960 // 2, dtype=torch.uint8, device=device)
        self.scales = torch.empty(4 * 512 * 8960 // 64, dtype=torch.float32, device=device)
        self.owner = None
        self.loaded = None
        self.read_done = None


class _Manager:
    def __init__(self, device):
        import torch

        self.device = device
        with torch.cuda.device(device):
            self.stream = torch.cuda.Stream(device=device)
            self.slots = [_Slot(device), _Slot(device)]
        self.leases = []
        self.hits = self.misses = 0

    def acquire_host(self, packet):
        packed_count, scale_count = packet.packed.numel(), packet.scales.numel()
        for lease in self.leases:
            if (lease.owner is None or lease.owner() is None) and (
                lease.packed.numel() >= packed_count and lease.scales.numel() >= scale_count
            ):
                return lease
        lease = _Lease(packed_count, scale_count)
        self.leases.append(lease)
        return lease

    def prepare(self, transfer, *, demand=False):
        import torch

        if transfer.slot is not None:
            slot = transfer.slot
            if slot.owner is not None and slot.owner() is transfer:
                return slot
        slot = None
        for candidate in self.slots:
            owner = candidate.owner() if candidate.owner is not None else None
            if owner is None or owner.consumed:
                slot = candidate
                break
        if slot is None and not demand:
            return None
        if slot is None:
            # A speculative packet can be evicted: the immutable host copy survives.
            slot = self.slots[0]
        old_owner = slot.owner() if slot.owner is not None else None
        if old_owner is not None:
            old_owner.slot = None
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            if slot.read_done is not None:
                self.stream.wait_event(slot.read_done)
            self.stream.wait_event(transfer.ready)
            packet = transfer.packet
            slot.packed[:packet.packed.numel()].view(packet.packed.dtype).copy_(
                packet.packed, non_blocking=True)
            slot.scales[:packet.scales.numel()].copy_(packet.scales, non_blocking=True)
            slot.loaded = torch.cuda.Event()
            slot.loaded.record(self.stream)
        slot.read_done = None
        slot.owner = weakref.ref(transfer)
        transfer.slot, transfer.consumed = slot, False
        transfer.lease.dependency = slot.loaded
        return slot


def pooled_offload_hooks(pack, unpack, *, should_offload, min_bytes=262144):
    import torch

    from .activation_storage import _Packet

    wrappers = {}
    records = []
    manager = None

    def offload(tensor):
        nonlocal manager
        packet = pack(tensor)
        if not isinstance(packet, _Packet):
            return packet
        reference = wrappers.get(id(packet))
        previous = reference() if reference is not None else None
        if previous is not None:
            return previous
        eligible = (should_offload() and packet.packed.numel() >= min_bytes
                    and packet.packed.numel() <= 4 * 512 * 8960 // 2
                    and packet.scales.numel() <= 4 * 512 * 8960 // 64)
        if not eligible:
            wrappers[id(packet)] = weakref.ref(packet)
            return packet
        device = packet.packed.device
        key = (device.type, device.index)
        if key not in _managers:
            _managers[key] = _Manager(device)
        manager = _managers[key]
        lease = manager.acquire_host(packet)
        host_packed = lease.packed[:packet.packed.numel()].view(packet.packed.dtype)
        host_scales = lease.scales[:packet.scales.numel()]
        with torch.cuda.device(device):
            producer = torch.cuda.current_stream(device)
            if lease.dependency is not None:
                producer.wait_event(lease.dependency)
            host_packed.copy_(packet.packed, non_blocking=True)
            host_scales.copy_(packet.scales, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(producer)
        packet.packed, packet.scales = host_packed, host_scales
        transfer = _Transfer(packet, device, ready, lease)
        lease.owner, lease.dependency = weakref.ref(transfer), ready
        wrappers[id(packet)] = weakref.ref(transfer)
        records.append(weakref.ref(transfer))
        return transfer

    def prefetch():
        if manager is None:
            return
        for reference in reversed(records):
            transfer = reference()
            if transfer is not None and not transfer.consumed:
                if manager.prepare(transfer) is None:
                    break

    def restore(value):
        if not isinstance(value, _Transfer):
            prefetch()
            return unpack(value)
        packet = value.packet
        cached = packet.decoded() if packet.decoded is not None else None
        if cached is not None:
            return unpack(packet)  # Includes the original version check.
        original = packet.original()
        if original is not None and original._version != packet.version:
            raise RuntimeError("a packed saved activation was modified in-place")
        if value.slot is None:
            manager.misses += 1
        else:
            manager.hits += 1
        slot = manager.prepare(value, demand=True)
        host_packed, host_scales = packet.packed, packet.scales
        with torch.cuda.device(value.device):
            consumer = torch.cuda.current_stream(value.device)
            consumer.wait_event(slot.loaded)
            packet.packed = slot.packed[:host_packed.numel()].view(host_packed.dtype)
            packet.scales = slot.scales[:host_scales.numel()]
            try:
                restored = unpack(packet)
            finally:
                packet.packed, packet.scales = host_packed, host_scales
            slot.read_done = torch.cuda.Event()
            slot.read_done.record(consumer)
        value.consumed = True
        prefetch()
        return restored

    return offload, restore
