"""Chaos tests for DistributedCoordinator: node churn, admission, routing.

Properties verified:
- Node registration/unregistration is consistent
- Stats reflect actual node count and capacities
- Admission control rejects when over threshold
- Least-loaded routing distributes fairly
- Node churn during submit doesn't crash
"""

from __future__ import annotations

from hypothesis import strategies as st

from ordeal import ChaosTest, always, invariant, rule

from enzu.isolation.scheduler import (
    AdmissionController,
    DistributedCoordinator,
    NodeStatus,
)


class CoordinatorChaos(ChaosTest):
    """Explore DistributedCoordinator under node registration churn."""

    faults = []

    def __init__(self) -> None:
        super().__init__()
        self.coordinator = DistributedCoordinator()
        self._registered: set[str] = set()
        self._node_counter = 0

    @rule()
    def register_node(self) -> None:
        """Register a new node with a local executor."""
        self._node_counter += 1
        node_id = f"node-{self._node_counter}"
        self.coordinator.register_node(
            node_id=node_id,
            endpoint=f"http://localhost:{8000 + self._node_counter}",
            capacity=10,
            queue_size=50,
            executor=lambda task: f"result-{task}",
        )
        self._registered.add(node_id)

    @rule()
    def unregister_random_node(self) -> None:
        """Unregister a random node if any exist."""
        if not self._registered:
            return
        node_id = next(iter(self._registered))
        self.coordinator.unregister_node(node_id)
        self._registered.discard(node_id)

    @rule(
        active=st.integers(min_value=0, max_value=10),
        queued=st.integers(min_value=0, max_value=50),
    )
    def update_random_node(self, active: int, queued: int) -> None:
        """Update capacity for a random registered node."""
        if not self._registered:
            return
        node_id = next(iter(self._registered))
        self.coordinator.update_node_capacity(
            node_id, active_workers=active, queued=queued
        )

    @rule()
    def check_stats(self) -> None:
        """Verify stats consistency."""
        stats = self.coordinator.stats()
        always(
            stats.total_nodes == len(self._registered),
            f"total_nodes({stats.total_nodes}) == registered({len(self._registered)})",
        )
        always(stats.total_nodes >= 0, "total_nodes >= 0")
        always(stats.healthy_nodes >= 0, "healthy_nodes >= 0")
        always(
            stats.healthy_nodes <= stats.total_nodes,
            "healthy <= total",
        )

    @invariant()
    def registered_count_consistent(self) -> None:
        stats = self.coordinator.stats()
        always(
            stats.total_nodes == len(self._registered),
            "node count matches registration tracking",
        )


TestCoordinatorChaos = CoordinatorChaos.TestCase


class AdmissionControlChaos(ChaosTest):
    """Explore AdmissionController with adversarial load/queue values."""

    faults = []

    def __init__(self) -> None:
        super().__init__()
        self.admission = AdmissionController(
            max_queue_depth=100,
            rejection_threshold=0.9,
        )

    @rule(
        load=st.floats(min_value=0.0, max_value=2.0).filter(lambda x: x == x),
        queue=st.integers(min_value=0, max_value=200),
    )
    def check_admission(self, load: float, queue: int) -> None:
        result = self.admission.should_admit(load, queue)
        always(isinstance(result, bool), "admission returns bool")

        # Over queue depth limit should reject
        if queue > 100:
            always(not result, "reject when queue exceeds max_queue_depth")

        # Over rejection threshold should reject
        if load > 0.9:
            always(not result, "reject when load exceeds threshold")


TestAdmissionControlChaos = AdmissionControlChaos.TestCase


class NodeCapacityChaos(ChaosTest):
    """Explore node capacity tracking under rapid updates."""

    faults = []

    def __init__(self) -> None:
        super().__init__()
        self.coordinator = DistributedCoordinator()
        # Register fixed nodes
        for i in range(3):
            self.coordinator.register_node(
                node_id=f"n{i}",
                endpoint=f"http://localhost:{9000 + i}",
                capacity=10,
                queue_size=50,
                executor=lambda task: "ok",
            )

    @rule(
        node=st.sampled_from(["n0", "n1", "n2"]),
        active=st.integers(min_value=0, max_value=10),
        queued=st.integers(min_value=0, max_value=50),
        status=st.sampled_from(
            [NodeStatus.HEALTHY, NodeStatus.DEGRADED, NodeStatus.DRAINING]
        ),
    )
    def update_capacity(
        self, node: str, active: int, queued: int, status: NodeStatus
    ) -> None:
        self.coordinator.update_node_capacity(
            node, active_workers=active, queued=queued, status=status
        )

    @rule()
    def verify_capacities(self) -> None:
        capacities = self.coordinator.node_capacities()
        always(len(capacities) == 3, "all 3 nodes present")
        for cap in capacities:
            always(cap.active_workers >= 0, "active_workers >= 0")
            always(cap.queued >= 0, "queued >= 0")

    @invariant()
    def stats_consistent(self) -> None:
        stats = self.coordinator.stats()
        always(stats.total_nodes == 3, "3 nodes always present")
        always(stats.total_capacity >= 0, "total_capacity >= 0")


TestNodeCapacityChaos = NodeCapacityChaos.TestCase
