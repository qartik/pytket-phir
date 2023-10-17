import bisect
import math

from pytket.phir.routing import inverse


class GateOpportunitiesError(Exception):
    """Exception raised when gating zones cannot accommodate all operations."""

    def __init__(self) -> None:
        """Exception"""  # noqa: D415
        super().__init__("Not enough gating opportunities for all ops in this layer")


class InvalidParallelOpsError(Exception):
    """Exception raised when a layer attempts to gate the same qubit more than once in parallel."""  # noqa: E501

    def __init__(self, q: int) -> None:
        """Exception
        Args: q: a qubit
        """  # noqa: D205, D415
        super().__init__(f"Cannot gate qubit {q} more than once in the same layer")


class PlacementCheckError(Exception):
    """Exception raised when placement check fails."""

    def __init__(self) -> None:
        """Exception"""  # noqa: D415
        super().__init__("Placement Check Failed")


def placement_check(
    ops: list[list[int]],
    tq_options: set[int],
    sq_options: set[int],
    state: list[int],
) -> bool:
    """Ensure that the qubits end up in the right gating zones."""
    placement_valid = False
    inv = inverse(state)

    # assume ops look like this [[1,2],[3],[4],[5,6],[7],[8],[9,10]]
    for op in ops:
        if len(op) == 2:  # tq operation   # noqa: PLR2004
            q1, q2 = op[0], op[1]
            # check that the q1 is next to q2 and they are in the right zone
            zone = (inv[q1] in tq_options) | (inv[q2] in tq_options)
            neighbor = (state.index(q2) == state.index(q1) + 1) | (
                state.index(q1) == state.index(q2) + 1
            )
            placement_valid = zone & neighbor

        else:  # sq operation
            q = op[0]
            zone = state.index(q) in sq_options
            placement_valid = zone

    return placement_valid


def nearest(zone: int, options: set[int]) -> int:
    """Return the nearest available zone to the given zone."""
    lst = sorted(options)
    ind = bisect.bisect_left(lst, zone)

    if ind == 0:
        nearest_zone = lst[0]
    elif ind == len(lst):
        nearest_zone = lst[-1]
    else:
        l = lst[ind - 1]  # noqa: E741
        r = lst[ind]
        nearest_zone = l if r - zone > zone - l else r

    return nearest_zone


def place_tq_ops(
    tq_ops: list[list[int]],
    placed_qubits: set[int],
    order: list[int],
    tq_zones: set[int],
    sq_zones: set[int],
) -> list[int]:
    """A helper function to place the TQ operations."""
    for op in tq_ops:
        q1, q2 = op[0], op[1]
        # check to make sure that the qubits have not already been placed
        if q1 in placed_qubits:
            raise InvalidParallelOpsError(q1)
        if q2 in placed_qubits:
            raise InvalidParallelOpsError(q2)
        midpoint = math.floor(abs(q2 - q1) / 2) + min(q1, q2)
        # find the tq gating zone closest to the midpoint of the 2 qubits
        nearest_tq_zone = nearest(midpoint, tq_zones)
        order[nearest_tq_zone] = q1
        order[nearest_tq_zone + 1] = q2
        # remove the occupied zones in the tap from tq and sq options
        tq_zones.discard(nearest_tq_zone)
        tq_zones.discard(nearest_tq_zone + 1)
        sq_zones.discard(nearest_tq_zone)
        sq_zones.discard(nearest_tq_zone + 1)
        placed_qubits.add(q1)
        placed_qubits.add(q2)
    return order


def place(  # noqa: PLR0912
    ops: list[list[int]],
    tq_options: set[int],
    sq_options: set[int],
    num_qubits: int,
) -> list[int]:
    """Place the qubits in the right order."""
    # assume ops look like this [[1,2],[3],[4],[5,6],[7],[8],[9,10]]
    order = [-1] * num_qubits
    placed_qubits: set[int] = set()

    tq_zones = tq_options.copy()
    sq_zones = sq_options.copy()

    tq_ops = []
    sq_ops = []

    # get separate lists of tq and sq operations
    for op in ops:
        if len(op) == 2:  # tq operation  # noqa: PLR2004
            tq_ops.append(op)
        else:  # sq_operation
            sq_ops.append(op)

    # sort the tq_ops by distance apart [[furthest] -> [closest]]
    tq_ops_sorted = sorted(tq_ops, key=lambda x: abs(x[0] - x[1]), reverse=True)  # type: ignore [misc] # noqa: E501, RUF100

    # check to make sure that there are zones available for all ops
    if len(tq_ops) > len(tq_zones):
        raise GateOpportunitiesError
    if len(sq_ops) > len(sq_zones) - 2 * len(tq_ops):
        # Because SQ zones are offsets of TQ zones, each tq op covers 2 sq zones
        raise GateOpportunitiesError

    # place the tq ops
    order = place_tq_ops(tq_ops_sorted, placed_qubits, order, tq_zones, sq_zones)

    # place the sq ops
    for op in sq_ops:
        q1 = op[0]
        # check to make sure that the qubits have not already been placed
        if q1 in placed_qubits:
            raise InvalidParallelOpsError(q1)
        # place the qubit in the first available zone
        for i in range(num_qubits):
            if i in sq_zones:
                order[i] = q1
                tq_zones.discard(i)
                sq_zones.discard(i)
                placed_qubits.add(q1)
                break

    # fill in the rest of the slots in the order with the inactive qubits
    for i in range(num_qubits):
        if i not in placed_qubits:
            for j in range(num_qubits):
                if order[j] == -1:
                    order[j] = i
                    break

    if placement_check(ops, tq_options, sq_options, order):
        return order
    else:
        raise PlacementCheckError


def optimized_place(  # noqa: PLR0912
    ops: list[list[int]],
    tq_options: set[int],
    sq_options: set[int],
    num_qubits: int,
    prev_state: list[int],
) -> list[int]:
    """Place the qubits in the right order."""
    # assume ops look like this [[1,2],[3],[4],[5,6],[7],[8],[9,10]]
    order = [-1] * num_qubits
    placed_qubits: set[int] = set()

    tq_zones = tq_options.copy()
    sq_zones = sq_options.copy()

    tq_ops = []
    sq_ops = []

    # get separate lists of tq and sq operations
    for op in ops:
        if len(op) == 2:  # tq operation  # noqa: PLR2004
            tq_ops.append(op)
        else:  # sq_operation
            sq_ops.append(op)

    # sort the tq_ops by distance apart [[furthest] -> [closest]]
    tq_ops_sorted = sorted(tq_ops, key=lambda x: abs(x[0] - x[1]), reverse=True)  # type: ignore [misc] # noqa: E501, RUF100

    # check to make sure that there are zones available for all ops
    if len(tq_ops) > len(tq_zones):
        raise GateOpportunitiesError
    if len(sq_ops) > len(sq_zones) - 2 * len(tq_ops):
        # Because SQ zones are offsets of TQ zones, each tq op covers 2 sq zones
        raise GateOpportunitiesError

    # place the tq ops
    order = place_tq_ops(tq_ops_sorted, placed_qubits, order, tq_zones, sq_zones)
    # run a check to avoid unnecessary swaps
    for zone in tq_zones:
        # this condition is true if there was a swap due to the order of qubits_used
        # in the shard creating the TQ op,
        # even if those two qubits were already in a TQ zone
        # example: ops = [[0,1]] prev_state = [0,1,2,3] new order = [1,0,2,3]
        # this check s to prevent the above situation
        if (order[zone] == prev_state[zone + 1]) & (
            order[zone + 1] == prev_state[zone]
        ):
            swapped = order[zone + 1]
            order[zone + 1] = order[zone]
            order[zone] = swapped
    # place the sq ops
    for op in sq_ops:
        q1 = op[0]
        # check to make sure that the qubits have not already been placed
        if q1 in placed_qubits:
            raise InvalidParallelOpsError(q1)

        prev_index = prev_state.index(q1)
        nearest_sq_zone = nearest(prev_index, sq_zones)
        order[nearest_sq_zone] = q1
        sq_zones.discard(nearest_sq_zone)
        placed_qubits.add(q1)

    # fill in the rest of the slots in the order with the inactive qubits
    for i in range(num_qubits):
        if i not in placed_qubits:
            prev_index = prev_state.index(i)
            nearest_sq_zone = nearest(prev_index, sq_zones)
            order[nearest_sq_zone] = i
            sq_zones.discard(nearest_sq_zone)

    if placement_check(ops, tq_options, sq_options, order):
        return order
    else:
        raise PlacementCheckError