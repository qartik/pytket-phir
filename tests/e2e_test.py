from pytket.phir.sharding.shard import Shard
from pytket.phir.sharding.sharder import Sharder
from tests.sample_data import QasmFiles, get_qasm_as_circuit


def parse_shards(shards: set[Shard]) -> list[list[list[int]]]:
    """Parse a set of shards and return a circuit representation for placement."""
    layers: list[list[list[int]]] = []
    scheduled: set[int] = set()
    num_shards: int = len(shards)

    while len(scheduled) < num_shards:
        layer: list[list[int]] = []
        to_schedule: list[Shard] = []
        # get all the shards with no dependencies
        for shard in enumerate(shards):
            s = shard[1]
            deps = s.depends_upon
            # dependencies of the shard that have already been scheduled
            scheduled_deps = deps.intersection(scheduled)
            already_scheduled = s.ID in scheduled

            if scheduled_deps == deps and not already_scheduled:
                to_schedule.append(s)

        for shard in to_schedule:  # type: ignore [assignment]
            op: list[int] = []
            # if there are more than 2 qubits used, treat them all as parallel sq ops
            # one qubit will just be a single sq op
            # 3 or more will be 3 or more parallel sq ops
            if len(shard.qubits_used) != 2:  # type: ignore [attr-defined, misc]
                for qubit in shard.qubits_used:  # type: ignore [attr-defined, misc]
                    op = qubit.index  # type: ignore [misc]
                    layer.append(op)
            else:
                for qubit in shard.qubits_used:  # type: ignore [attr-defined, misc]
                    op.append(qubit.index[0])  # type: ignore [misc]
                layer.append(op)

            scheduled.add(shard.ID)  # type: ignore [attr-defined, misc]

        layers.append(layer)

    return layers


if __name__ == "__main__":
    circuit = get_qasm_as_circuit(QasmFiles.barrier_complex)
    # check simple.qasm?
    sharder = Sharder(circuit)
    shards = sharder.shard()
    shard_set = set(shards)
    circuit_rep = parse_shards(shard_set)
    # print(circuit_rep)
