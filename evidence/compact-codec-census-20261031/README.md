# Exact-view packet census (diagnostic)

Runtimec369cc7, all64 fixed training batches, original task/allocator, no eval claim.
74624 save-hook requests;14080 eligible compressed saves.1728 distinct packets
covered identical storage views in the same live root family, dtype, version,
shape/stride/offset, codec and placement. GPU duplicated live packets reached
38.2139MiB. Summed bytes across all steps are traffic estimates, not peak savings.
The observer retains only weak tensor/packet references and integer metadata.

Stacks identify compact_decoder.dropout_forward index_copy_ at the q/v LoRA
input reconstruction. All inspected actual varlen Q/K/V/output saves remain
4D and uncompressed. census.json.gz records complete counts, scopes and boundaries.
The later compact_dropout implementation removes these unneeded saves directly.

summary.json contains the main counters and one actual caller stack.
