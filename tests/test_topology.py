"""Tests for topology extraction: node indexing, edge connectivity, Y-bus."""

import numpy as np
import pytest


def test_topology_node_count(feeder_graph):
    """IEEE 123-bus feeder should produce 278 energised bus-phase nodes."""
    assert feeder_graph.num_nodes == 278, f"Expected 278, got {feeder_graph.num_nodes}"


def test_topology_has_edges(feeder_graph):
    """Topology must have edges."""
    assert feeder_graph.num_edges > 0


def test_node_indexing_deterministic(master_file):
    """Node indexing must be identical across two independent extractions."""
    from dsse.topology import extract_feeder_topology
    g1 = extract_feeder_topology(master_file)
    g2 = extract_feeder_topology(master_file)
    labels1 = [n.node_id for n in g1.nodes]
    labels2 = [n.node_id for n in g2.nodes]
    assert labels1 == labels2, "Node ordering is non-deterministic!"


def test_ybus_is_square(feeder_graph):
    """Y-bus matrix must be square and match the number of nodes."""
    if feeder_graph.y_bus_real is not None:
        N = feeder_graph.num_nodes
        assert feeder_graph.y_bus_real.shape == (N, N)
        assert feeder_graph.y_bus_imag.shape == (N, N)


def test_slack_bus_exists(feeder_graph):
    """At least one slack node must be identified."""
    slack_nodes = [n for n in feeder_graph.nodes if n.is_slack]
    assert len(slack_nodes) >= 1, "No slack node found in topology."


def test_phase_labels(feeder_graph):
    """All nodes must have phase labels in {1, 2, 3}."""
    for node in feeder_graph.nodes:
        assert node.phase in {1, 2, 3}, f"Invalid phase {node.phase} for node {node.node_id}"
