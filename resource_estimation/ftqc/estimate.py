# Copyright 2026 Infleqtion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import collections
import typing
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import cirq
import networkx as nx
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

if typing.TYPE_CHECKING:  # pragma: no cover
    from resource_estimation.ftqc.architecture import Architecture
    from resource_estimation.ftqc.layout import Layout
    from resource_estimation.typing import GateCounts, GateKey

from resource_estimation.typing import _require_gate_operation

warnings.filterwarnings("ignore", category=RuntimeWarning)


class ResourceEstimator:
    """Class for resource estimator objects defined by the given architecture"""

    def __init__(self, arc: Architecture) -> None:
        self.arc = arc

    def validate_circuit_ops(self, circuit: cirq.Circuit) -> None:
        """Checks that the input circuit contains only valid operations and warns of operations still in progress"""
        unrecognized = [op for op in circuit.all_operations() if op not in self.arc.primitives]
        if unrecognized:
            error_message = """This circuit has gates that are incompatible with the input architecture parameters.\nThe following gates in this circuit are not recognized:"""
            for op in unrecognized:
                error_message += f"\n{op!s}"
            raise ValueError(error_message)

    def serial_circuit_cost(
        self,
        circuit: cirq.Circuit,
        layout: Layout,
        verbose: int = 0,
    ) -> dict[GateKey, int]:
        """Counts up the total physical gates from all logical primitives in the input circuit"""
        self.validate_circuit_ops(circuit=circuit)
        cost: collections.Counter[GateKey] = collections.Counter()
        for op in tqdm(
            circuit.all_operations(),
            total=len(list(circuit.all_operations())),
            colour="cyan",
            disable=not bool(verbose),
        ):
            cost += collections.Counter(self.arc.gate_cost(op, layout=layout))
        return {op: val for op, val in cost.items()}

    def serial_circuit_time(self, circuit: cirq.Circuit, layout: Layout) -> float:
        """Adds up the total physical time from all logical primitives in the input circuit"""
        self.validate_circuit_ops(circuit=circuit)
        return sum(self.arc.op_time(op, layout=layout) for op in circuit.all_operations())

    def parallel_circuit_time(
        self, circuit: cirq.Circuit, layout: Layout, verbose: int = 0
    ) -> float:
        """Estimation of the critical path in the input circuit according to the most expensive operation per moment"""
        qubit_times: dict[cirq.Qid, float] = dict.fromkeys(circuit.all_qubits(), 0.0)
        total_ops = len(list(circuit.all_operations()))
        for op in tqdm(
            circuit.all_operations(), disable=not verbose, total=total_ops, colour="cyan"
        ):
            big_time = max(qubit_times[q] for q in op.qubits)
            big_time += self.arc.op_time(op, layout=layout)
            for qubit in op.qubits:
                qubit_times[qubit] = big_time
        return max(qubit_times.values())

    def critical_path(
        self, circuit: cirq.Circuit, layout: Layout, verbose: int = 0
    ) -> list[cirq.Operation]:
        """Returns the circuit's critical path in terms of the logical primitive operations
        Is very slow and expensive
        """
        warnings.warn(
            "This function can be very expensive.\nFor physical operations or circuit time, use `parallel_circuit_cost` or `parallel_circuit_time`, respectively.",
        )
        qubit_paths: dict[cirq.Qid, list[cirq.Operation]] = {
            qubit: [] for qubit in circuit.all_qubits()
        }
        qubit_times: dict[cirq.Qid, float] = dict.fromkeys(circuit.all_qubits(), 0.0)
        total_ops = len(list(circuit.all_operations()))
        for op in tqdm(
            circuit.all_operations(),
            disable=not verbose,
            total=total_ops,
            colour="cyan",
        ):
            op_qubits = op.qubits
            # This qubit currently has the longest path
            big_qubit = max(op_qubits, key=qubit_times.__getitem__)
            big_path = qubit_paths[big_qubit]
            big_time = qubit_times[big_qubit]
            big_path.append(op)
            big_time += self.arc.op_time(op, layout=layout)
            for qubit in op_qubits:
                qubit_paths[qubit] = big_path.copy()
                qubit_times[qubit] = big_time
        critical_qubit = max(qubit_times, key=qubit_times.__getitem__)
        critical_path = qubit_paths[critical_qubit]
        return critical_path

    def parallel_circuit_cost(
        self,
        circuit: cirq.Circuit,
        layout: Layout,
        verbose: int = 0,
    ) -> GateCounts:
        """Estimation of the physical operations in critical path of the input circuit according to the most expensive operation per moment"""
        qubit_paths: dict[cirq.Qid, collections.Counter[GateKey]] = {
            qubit: collections.Counter() for qubit in circuit.all_qubits()
        }
        qubit_times: dict[cirq.Qid, float] = dict.fromkeys(circuit.all_qubits(), 0.0)
        total_ops = len(list(circuit.all_operations()))
        for op in tqdm(
            circuit.all_operations(), disable=not verbose, total=total_ops, colour="cyan"
        ):
            op_qubits = op.qubits
            # This qubit currently has the longest path
            big_qubit = max(op_qubits, key=qubit_times.__getitem__)
            big_time = qubit_times[big_qubit] + self.arc.op_time(op, layout=layout)
            big_path = qubit_paths[big_qubit] + collections.Counter(
                self.arc.moment_cost(op, layout=layout)
            )
            for qubit in op_qubits:
                qubit_paths[qubit] = big_path
                qubit_times[qubit] = big_time

        big_qubit = max(op_qubits, key=qubit_times.__getitem__)
        big_time = qubit_times[big_qubit]
        big_path = qubit_paths[big_qubit]

        return big_path

    def physical_qubits(self, circuit: cirq.Circuit) -> int:
        """Calculates the physical qubit cost of the requested circuit"""
        return cirq.num_qubits(circuit) * self.arc.patch.num_physical_qubits


ReactionTreeNode = tuple[int, int]
Q = typing.TypeVar("Q", bound=cirq.Qid)


@dataclass(frozen=True)
class ReactionDynamics(typing.Generic[Q]):
    """Reaction dynamic for each delayed choice measurement in a factory.

    Attributes:
        dependency_paulis: Paulis which create a dependency when any one of them
            anti-commutes with a propagated Pauli.
        outputs: Pauli corrections output by the vertex. Dynamics should
            use `cirq.LineQubit(i)` for operation-local qubit `i`.
    """

    dependency_paulis: tuple[cirq.PauliString[Q], ...]
    outputs: tuple[cirq.PauliString[Q], ...]


class _CliffordFrame:
    """Phase-free Pauli frame that pulls Paulis back to the start of a circuit.

    Paulis are bit-packed as `x | z << num_qubits` over the sorted circuit qubits.
    Applying a Clifford operation only updates the pulled-back images of the X and Z
    generators on its qubits, so Paulis pulled back at different times never need to
    be propagated. Conjugation preserves commutation, so two pulled-back Paulis
    anti-commute exactly when their propagated counterparts do.
    """

    def __init__(self, qubits: Iterable[cirq.Qid]) -> None:
        self._qubit_indices = {qubit: index for index, qubit in enumerate(sorted(qubits))}
        self._num_qubits = len(self._qubit_indices)
        self._low_mask = (1 << self._num_qubits) - 1
        self._x_images = [1 << index for index in range(self._num_qubits)]
        self._z_images = [1 << (index + self._num_qubits) for index in range(self._num_qubits)]
        self._local_pull_back_images: dict[cirq.Gate, tuple[int, ...]] = {}

    def pull_back(self, pauli: cirq.PauliString[cirq.Qid]) -> int:
        """Packed Pauli at the circuit start that propagates to `pauli` (up to phase)."""
        packed_pauli = 0
        for qubit, pauli_gate in pauli.items():
            index = self._qubit_indices[qubit]
            if pauli_gate != cirq.Z:
                packed_pauli ^= self._x_images[index]
            if pauli_gate != cirq.X:
                packed_pauli ^= self._z_images[index]
        return packed_pauli

    def apply(self, operation: cirq.Operation) -> None:
        """Advance the frame past a Clifford operation."""
        indices = [self._qubit_indices[qubit] for qubit in operation.qubits]
        generator_images = [self._x_images[index] for index in indices] + [
            self._z_images[index] for index in indices
        ]
        new_images = []
        for local_image in self._local_pull_back_images_for(operation):
            image = 0
            for bit, generator_image in enumerate(generator_images):
                if local_image >> bit & 1:
                    image ^= generator_image
            new_images.append(image)
        for position, index in enumerate(indices):
            self._x_images[index] = new_images[position]
            self._z_images[index] = new_images[position + len(indices)]

    def anticommutes(self, pauli: int, other: int) -> bool:
        """Whether two packed Paulis anti-commute."""
        swapped_other = (other >> self._num_qubits) | ((other & self._low_mask) << self._num_qubits)
        return (pauli & swapped_other).bit_count() % 2 == 1

    def _local_pull_back_images_for(self, operation: cirq.Operation) -> tuple[int, ...]:
        gate = _require_gate_operation(operation).gate
        if gate not in self._local_pull_back_images:
            self._local_pull_back_images[gate] = _local_pull_back_images(
                gate, len(operation.qubits)
            )
        return self._local_pull_back_images[gate]


def _local_pull_back_images(gate: cirq.Gate, num_qubits: int) -> tuple[int, ...]:
    """Images of the local X then Z generators pulled back across `gate`.

    Paulis move forward across a Clifford `C` as `P -> C P C†`, so pulling back is
    `P -> C† P C`, which is `cirq.PauliString.conjugated_by`. Images are packed over the
    gate's local qubits.
    """
    local_qubits = cirq.LineQubit.range(num_qubits)
    local_frame = _CliffordFrame(local_qubits)
    operation = gate.on(*local_qubits)
    return tuple(
        local_frame.pull_back(cirq.PauliString(pauli(qubit)).conjugated_by(operation))
        for pauli in (cirq.X, cirq.Z)
        for qubit in local_qubits
    )


class _DepthSpans:
    """GF(2) bases of the pulled-back output Paulis created at each reaction depth.

    The symplectic form is linear, so a Pauli anti-commutes with some element of a set
    iff it anti-commutes with some element of the set's span. A reduced basis per depth
    (at most `2 * num_qubits` vectors) therefore replaces the full set of outputs.
    """

    def __init__(self, frame: _CliffordFrame) -> None:
        self._frame = frame
        self._bases: list[dict[int, int]] = []

    def insert(self, depth: int, pauli: int) -> None:
        """Add a packed output Pauli created at `depth`, which starts at one."""
        while len(self._bases) < depth:
            self._bases.append({})
        basis = self._bases[depth - 1]
        while pauli:
            pivot = pauli.bit_length()
            if pivot not in basis:
                basis[pivot] = pauli
                return
            pauli ^= basis[pivot]

    def max_dependency_depth(self, dependency_paulis: tuple[int, ...]) -> int:
        """Deepest output anti-commuting with any dependency Pauli, or zero if none."""
        for depth in range(len(self._bases), 0, -1):
            if any(
                self._frame.anticommutes(output, dependency_pauli)
                for output in self._bases[depth - 1].values()
                for dependency_pauli in dependency_paulis
            ):
                return depth
        return 0


# Renumbering waits until the index space exceeds twice the live vertices by this many bits.
_RENUMBERING_SLACK = 64


class _ReactionFrames:
    """Phase-free Pauli frames of factory outputs, propagated forward and bit-sliced by qubit.

    Bit `i` of every column and ancestor mask refers to tracked vertex `i`, in circuit order.
    In lane `j`, bit `i` of the X (Z) column of a qubit is set when the `j`-th output of vertex
    `i`, propagated to the current point of the circuit, has an X (Z) component on that qubit.
    Dependency Paulis are operation-local, so the vertices with an output anti-commuting with
    one are an XOR of a few columns, and a Clifford operation only recombines the columns of
    its qubits.
    """

    def __init__(self, qubits: Iterable[cirq.Qid]) -> None:
        self._qubit_indices = {qubit: index for index, qubit in enumerate(sorted(qubits))}
        self._lanes: list[tuple[list[int], list[int]]] = []
        self._nodes: list[ReactionTreeNode] = []
        self._ancestor_masks: list[int] = []
        self._next_renumbering = _RENUMBERING_SLACK
        self._local_forward_sources: dict[cirq.Gate, tuple[tuple[int, ...], ...]] = {}

    def apply(self, operation: cirq.Operation) -> None:
        """Propagate every tracked output forward across a Clifford operation."""
        local_forward_sources = self._local_forward_sources_for(operation)
        if not local_forward_sources:
            return
        indices = [self._qubit_indices[qubit] for qubit in operation.qubits]
        for x_columns, z_columns in self._lanes:
            local_columns = [x_columns[index] for index in indices] + [
                z_columns[index] for index in indices
            ]
            new_columns = []
            for sources in local_forward_sources:
                column = 0
                for source in sources:
                    column ^= local_columns[source]
                new_columns.append(column)
            for position, index in enumerate(indices):
                x_columns[index] = new_columns[position]
                z_columns[index] = new_columns[position + len(indices)]

    def reduced_parents(
        self, dependency_paulis: tuple[cirq.PauliString[cirq.Qid], ...]
    ) -> tuple[list[ReactionTreeNode], int]:
        """Transitive-reduction parents of a new vertex, and the mask of its tracked ancestors.

        Every vertex reachable from a candidate comes after it, so the latest candidate reaches
        no other candidate and is a parent. Clearing it and its ancestors leaves the candidates
        that reach none of the parents found so far.
        """
        parents: list[ReactionTreeNode] = []
        ancestors = 0
        candidates = self._anticommuting(dependency_paulis)
        while candidates:
            parent = candidates.bit_length() - 1
            parents.append(self._nodes[parent])
            ancestors |= self._ancestor_masks[parent]
            candidates &= ~self._ancestor_masks[parent]
        return parents, ancestors

    def add(
        self,
        node: ReactionTreeNode,
        outputs: tuple[cirq.PauliString[cirq.Qid], ...],
        ancestors: int,
    ) -> None:
        """Track a new vertex given the mask of its tracked ancestors.

        Each output is also absorbed into the tracked outputs of the ancestors that share its
        first component. Multiplying a tracked output by a descendant's output leaves the parents
        of every later vertex unchanged: a vertex with no descendant that the later vertex depends
        on anti-commutes with it exactly as before, and any other vertex reaches a parent either
        way. Vertices whose tracked outputs all become the identity can never be parents again.
        """
        vertex_mask = 1 << len(self._nodes)
        self._nodes.append(node)
        self._ancestor_masks.append(ancestors | vertex_mask)
        while len(self._lanes) < len(outputs):
            self._lanes.append(([0] * len(self._qubit_indices), [0] * len(self._qubit_indices)))
        for (x_columns, z_columns), output in zip(self._lanes, outputs):
            components: list[tuple[list[int], int]] = []
            for qubit, pauli_gate in output.items():
                index = self._qubit_indices[qubit]
                if pauli_gate != cirq.Z:
                    components.append((x_columns, index))
                if pauli_gate != cirq.X:
                    components.append((z_columns, index))
            if components:
                pivot_columns, pivot_index = components[0]
                absorbed = ancestors & pivot_columns[pivot_index]
                for columns, index in components:
                    columns[index] ^= absorbed | vertex_mask

    def renumber_if_sparse(self) -> None:
        """Drop the vertices whose tracked outputs are all the identity, once they dominate.

        Identity outputs stay the identity under propagation and absorption, so dropped vertices
        never return. Checks are spaced out geometrically when few vertices drop out.
        """
        num_bits = len(self._nodes)
        if num_bits < self._next_renumbering:
            return
        live = 0
        for x_columns, z_columns in self._lanes:
            for column in (*x_columns, *z_columns):
                live |= column
        if num_bits >= 2 * live.bit_count() + _RENUMBERING_SLACK:
            positions = _bit_positions(live, num_bits)
            for x_columns, z_columns in self._lanes:
                x_columns[:] = [_gather_bits(column, positions, num_bits) for column in x_columns]
                z_columns[:] = [_gather_bits(column, positions, num_bits) for column in z_columns]
            self._nodes = [self._nodes[position] for position in positions]
            self._ancestor_masks = [
                _gather_bits(self._ancestor_masks[position], positions, num_bits)
                for position in positions
            ]
        self._next_renumbering = 2 * len(self._nodes) + _RENUMBERING_SLACK

    def _anticommuting(self, paulis: tuple[cirq.PauliString[cirq.Qid], ...]) -> int:
        vertices = 0
        for pauli in paulis:
            for x_columns, z_columns in self._lanes:
                anticommuting = 0
                for qubit, pauli_gate in pauli.items():
                    index = self._qubit_indices[qubit]
                    if pauli_gate != cirq.X:
                        anticommuting ^= x_columns[index]
                    if pauli_gate != cirq.Z:
                        anticommuting ^= z_columns[index]
                vertices |= anticommuting
        return vertices

    def _local_forward_sources_for(self, operation: cirq.Operation) -> tuple[tuple[int, ...], ...]:
        gate = _require_gate_operation(operation).gate
        if gate not in self._local_forward_sources:
            self._local_forward_sources[gate] = _local_forward_sources(gate, len(operation.qubits))
        return self._local_forward_sources[gate]


def _local_forward_sources(gate: cirq.Gate, num_qubits: int) -> tuple[tuple[int, ...], ...]:
    """Old local X then Z columns XORed into each new one when propagating forward across `gate`.

    Propagating forward inverts pulling back. Clifford conjugation is symplectic, so the inverse
    of the pulled-back images `B` is `Ω Bᵀ Ω`, where `Ω` swaps X and Z: new column `a` is the
    XOR of the old columns at the swapped positions of the pulled-back image of generator `Ω a`.
    Returns an empty tuple when propagation does nothing, as for Paulis and measurements.
    """
    images = _local_pull_back_images(gate, num_qubits)
    swapped = [*range(num_qubits, 2 * num_qubits), *range(num_qubits)]
    sources = tuple(
        tuple(swapped[bit] for bit in range(2 * num_qubits) if images[swapped[position]] >> bit & 1)
        for position in range(2 * num_qubits)
    )
    if all(position_sources == (position,) for position, position_sources in enumerate(sources)):
        return ()
    return sources


def _bit_positions(bits: int, num_bits: int) -> npt.NDArray[np.intp]:
    """Increasing positions of the set bits of `bits`, which is below `2**num_bits`."""
    return np.flatnonzero(_unpack_bits(bits, num_bits))


def _gather_bits(bits: int, positions: npt.NDArray[np.intp], num_bits: int) -> int:
    """Bitset whose bit `i` is bit `positions[i]` of `bits`, which is below `2**num_bits`."""
    gathered = np.packbits(_unpack_bits(bits, num_bits)[positions], bitorder="little")
    return int.from_bytes(gathered.tobytes(), "little")


def _unpack_bits(bits: int, num_bits: int) -> npt.NDArray[np.uint8]:
    """Little-endian bits of `bits`, which is below `2**num_bits`, one per array entry."""
    return np.unpackbits(
        np.frombuffer(bits.to_bytes((num_bits + 7) // 8, "little"), dtype=np.uint8),
        bitorder="little",
    )


class ReactionDepthEstimator:
    """Estimator for logical reaction depth in a Clifford+factory circuit.

    Factory operations become graph vertices. Their output Paulis propagate
    forward through later Clifford operations: reading the circuit left to right,
    `- P - C -` becomes `- C - P' -` with `P' = C P C†`. An output creates a dependency
    when it anti-commutes with any of the target vertex's dependency Paulis.
    """

    _DEFAULT_FACTORIES: typing.ClassVar[dict[cirq.Gate, bool]] = {
        cirq.T: True,
        cirq.S: True,
        cirq.CCZ: True,
    }
    _NON_CLIFFORD_ERROR: typing.ClassVar[str] = (
        "Reaction-depth estimator encountered a non-Clifford operation without "
        "factory reaction dynamics: {operation!r}."
    )

    def __init__(
        self,
        factories: dict[GateKey, bool] | None = None,
        factory_reaction_dynamics: Mapping[
            tuple[cirq.Gate, bool],
            Sequence[ReactionDynamics[cirq.LineQubit]],
        ]
        | None = None,
    ) -> None:
        """Configure factory gates and their reaction dynamics.

        Args:
            factories: Factory gates mapped to their auto-correction setting.
                Defaults to auto-corrected T, S, and CCZ gates.
            factory_reaction_dynamics: Factory reaction dynamics keyed by
                `(gate, auto_corrected)`. Entries override defaults;
                `factories` still determines which gates are factory-backed.
                Pauli qubits are operation-local `cirq.LineQubit` indices from
                zero through the gate's arity minus one.

        Raises:
            ValueError: If a configured factory has no reaction dynamics or a
                dynamic uses a qubit index outside the factory gate's arity.
        """
        self.factories = dict(self._DEFAULT_FACTORIES if factories is None else factories)
        local_qubits = cirq.LineQubit.range(3)
        local_x: cirq.PauliString[cirq.LineQubit] = cirq.PauliString(cirq.X(local_qubits[0]))
        local_z: tuple[cirq.PauliString[cirq.LineQubit], ...] = tuple(
            cirq.PauliString(cirq.Z(qubit)) for qubit in local_qubits
        )
        self._factory_reaction_dynamics: dict[
            tuple[GateKey, bool], tuple[ReactionDynamics[cirq.LineQubit], ...]
        ] = {
            (cirq.T, True): (ReactionDynamics((local_z[0],), (local_z[0],)),),
            (cirq.T, False): (ReactionDynamics((local_x, local_z[0]), (local_x, local_z[0])),),
            (cirq.S, True): (ReactionDynamics((local_z[0],), (local_z[0],)),),
            (cirq.CCZ, True): (
                ReactionDynamics((local_z[0],), (local_z[1] * local_z[2],)),
                ReactionDynamics((local_z[1],), (local_z[0] * local_z[2],)),
                ReactionDynamics((local_z[2],), (local_z[0] * local_z[1],)),
            ),
        }
        if factory_reaction_dynamics is not None:
            self._factory_reaction_dynamics.update(
                {
                    key: tuple(factory_dynamics)
                    for key, factory_dynamics in factory_reaction_dynamics.items()
                }
            )

        unsupported_pairs = [
            (gate, auto_corrected)
            for gate, auto_corrected in self.factories.items()
            if (gate, auto_corrected) not in self._factory_reaction_dynamics
        ]
        if unsupported_pairs:
            raise ValueError(
                "No factory reaction dynamics are defined for: "
                + ", ".join(
                    f"({gate!r}, {auto_corrected!r})" for gate, auto_corrected in unsupported_pairs
                ),
            )

        for gate, auto_corrected in self.factories.items():
            factory_local_qubits = tuple(cirq.LineQubit.range(cirq.num_qubits(gate)))
            for reaction_dynamics in self._factory_reaction_dynamics[(gate, auto_corrected)]:
                for pauli in (
                    *reaction_dynamics.dependency_paulis,
                    *reaction_dynamics.outputs,
                ):
                    if not set(pauli.qubits).issubset(factory_local_qubits):
                        raise ValueError(
                            f"Reaction Pauli {pauli!r} must use only operation-local qubits "
                            f"{factory_local_qubits} for factory gate {gate!r}."
                        )

    def reaction_depth(self, circuit: cirq.Circuit) -> int:
        """Compute the logical factory reaction depth for a circuit.

        Args:
            circuit: Logical circuit whose factory-backed operations and
                Clifford propagation should be tracked.

        Returns:
            Longest factory-vertex dependency chain. Circuits with no factory
            vertices have depth zero.
        """
        frame = _CliffordFrame(circuit.all_qubits())
        depth_spans = _DepthSpans(frame)
        reaction_depth = 0

        for operation, factory_dynamics in self._operation_dynamics(circuit.all_operations()):
            if factory_dynamics is None:
                frame.apply(operation)
                continue

            node_depths = tuple(
                depth_spans.max_dependency_depth(
                    tuple(frame.pull_back(pauli) for pauli in reaction_dynamics.dependency_paulis)
                )
                + 1
                for reaction_dynamics in factory_dynamics
            )
            reaction_depth = max((reaction_depth, *node_depths))

            for reaction_dynamics, node_depth in zip(factory_dynamics, node_depths, strict=True):
                for output in reaction_dynamics.outputs:
                    depth_spans.insert(node_depth, frame.pull_back(output))

        return reaction_depth

    def reaction_tree(self, circuit: cirq.Circuit) -> nx.DiGraph:
        """Build the transitively reduced factory-vertex dependency graph.

        Nodes are `(operation_index, vertex_index)` tuples. Edges point from a
        source factory vertex to a later factory vertex when a propagated source
        output anti-commutes with a target vertex dependency Pauli.

        Args:
            circuit: Logical circuit whose factory-backed operations and
                Clifford propagation should be tracked.

        Returns:
            NetworkX DAG with operation metadata in `graph["operations"]` and
            per-node longest-chain depths in node attribute `"depth"`.
        """
        operations = tuple(circuit.all_operations())
        reaction_tree = nx.DiGraph(operations=operations)
        frames = _ReactionFrames(circuit.all_qubits())
        for operation_index, (operation, factory_dynamics) in enumerate(
            self._operation_dynamics(operations)
        ):
            if factory_dynamics is None:
                frames.apply(operation)
                continue

            # Vertices of one operation never depend on each other, so their outputs are only
            # tracked once all of them have their parents.
            ancestor_masks: list[int] = []
            for vertex_index, reaction_dynamics in enumerate(factory_dynamics):
                node = (operation_index, vertex_index)
                reaction_tree.add_node(
                    node,
                    dependency_paulis=reaction_dynamics.dependency_paulis,
                    outputs=reaction_dynamics.outputs,
                )
                parents, ancestor_mask = frames.reduced_parents(reaction_dynamics.dependency_paulis)
                reaction_tree.add_edges_from((parent, node) for parent in parents)
                ancestor_masks.append(ancestor_mask)
            for vertex_index, (reaction_dynamics, ancestor_mask) in enumerate(
                zip(factory_dynamics, ancestor_masks, strict=True)
            ):
                frames.add(
                    (operation_index, vertex_index), reaction_dynamics.outputs, ancestor_mask
                )
            frames.renumber_if_sparse()

        for depth, nodes in enumerate(nx.topological_generations(reaction_tree), start=1):
            for node in nodes:
                reaction_tree.nodes[node]["depth"] = depth

        return reaction_tree

    def _operation_dynamics(
        self, operations: Iterable[cirq.Operation]
    ) -> Iterator[tuple[cirq.GateOperation, tuple[ReactionDynamics[cirq.Qid], ...] | None]]:
        """Pair each operation with its factory dynamics, or `None` for a Clifford operation.

        The dynamics only depend on the gate and its qubits, which repeat throughout a circuit,
        so each pair is only checked and localized once.
        """
        cache: dict[
            tuple[cirq.Gate, tuple[cirq.Qid, ...]], tuple[ReactionDynamics[cirq.Qid], ...] | None
        ] = {}
        for operation in operations:
            operation = _require_gate_operation(operation)
            key = (operation.gate, operation.qubits)
            if key not in cache:
                cache[key] = self._factory_dynamics(operation)
            yield operation, cache[key]

    def _factory_dynamics(
        self, operation: cirq.Operation
    ) -> tuple[ReactionDynamics[cirq.Qid], ...] | None:
        operation = _require_gate_operation(operation)
        if operation.gate not in self.factories:
            if not cirq.has_stabilizer_effect(operation.gate):
                raise ValueError(self._NON_CLIFFORD_ERROR.format(operation=operation))
            return None

        local_qubit_map: dict[cirq.LineQubit, cirq.Qid] = dict(
            zip(cirq.LineQubit.range(len(operation.qubits)), operation.qubits, strict=True)
        )
        return tuple(
            ReactionDynamics(
                tuple(
                    pauli.map_qubits(local_qubit_map)
                    for pauli in reaction_dynamics.dependency_paulis
                ),
                tuple(output.map_qubits(local_qubit_map) for output in reaction_dynamics.outputs),
            )
            for reaction_dynamics in self._factory_reaction_dynamics[
                (operation.gate, self.factories[operation.gate])
            ]
        )
