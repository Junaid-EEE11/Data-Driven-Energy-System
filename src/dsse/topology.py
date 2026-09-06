"""Feeder topology extraction, graph representation, and phase-preserving indexing."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from dsse.reproducibility import setup_logger

logger = setup_logger("topology")


@dataclass
class NodeMetadata:
    node_idx: int
    bus_name: str
    phase: int  # 1 (A), 2 (B), 3 (C)
    node_id: str  # e.g. "1.1", "150.2"
    base_kv_ln: float
    is_slack: bool = False
    is_load: bool = False
    is_cap: bool = False
    is_der: bool = False


@dataclass
class EdgeMetadata:
    edge_idx: int
    from_bus: str
    to_bus: str
    from_node_idx: int
    to_node_idx: int
    element_type: str  # "line", "transformer", "switch", "regulator"
    element_name: str
    r_series: float  # ohms
    x_series: float  # ohms
    b_shunt: float  # siemens
    length: float  # miles / kft


@dataclass
class FeederGraph:
    """Complete graph representation of the unbalanced distribution feeder."""
    name: str
    nodes: List[NodeMetadata]
    edges: List[EdgeMetadata]
    node_id_to_idx: Dict[str, int]
    bus_to_nodes: Dict[str, List[int]]
    edge_index: torch.Tensor  # [2, E]
    edge_attr: torch.Tensor  # [E, F_edge]
    node_features: torch.Tensor  # [N, F_node]
    y_bus_sparse: sp.csr_matrix
    y_bus_dense: np.ndarray
    networkx_graph: nx.Graph

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    @property
    def y_bus_real(self) -> np.ndarray:
        return np.real(self.y_bus_dense)

    @property
    def y_bus_imag(self) -> np.ndarray:
        return np.imag(self.y_bus_dense)


def extract_feeder_topology(dss_source: Any, feeder_name: str = "ieee123") -> FeederGraph:
    """Extracts deterministic phase-preserving graph and admittance matrices from active OpenDSS circuit or master file."""
    if isinstance(dss_source, (str, Path)):
        import opendssdirect as dss_direct
        master_path = Path(dss_source).resolve()
        dss_direct.Basic.ClearAll()
        dss_direct.Text.Command(f'compile "{master_path}"')
        dss_direct.Solution.Solve()
        dss_module = dss_direct
    else:
        dss_module = dss_source

    all_node_names = dss_module.Circuit.AllNodeNames()
    if not all_node_names:
        raise ValueError("OpenDSS circuit has no nodes. Please compile the circuit first.")

    # Sort node names deterministically
    # OpenDSS returns node names like "150.1", "1.1", "1.2", "1.3"
    def node_sort_key(name: str) -> Tuple[str, int]:
        parts = name.split(".")
        bus = parts[0]
        phase = int(parts[1]) if len(parts) > 1 else 1
        return (bus, phase)

    # Deterministic sorting
    sorted_node_names = sorted(list(set(all_node_names)), key=lambda x: (x.split(".")[0], int(x.split(".")[1]) if len(x.split(".")) > 1 else 1))

    node_id_to_idx: Dict[str, int] = {name: i for i, name in enumerate(sorted_node_names)}
    bus_to_nodes: Dict[str, List[int]] = {}

    nodes: List[NodeMetadata] = []
    slack_bus = "150"

    # Identify load buses, capacitor buses
    all_load_names = dss_module.Loads.AllNames()
    load_buses: Set[str] = set()
    for lname in all_load_names:
        dss_module.Loads.Name(lname)
        bus = dss_module.CktElement.BusNames()[0].split(".")[0]
        load_buses.add(bus)

    all_cap_names = dss_module.Capacitors.AllNames()
    cap_buses: Set[str] = set()
    for cname in all_cap_names:
        dss_module.Capacitors.Name(cname)
        bus = dss_module.CktElement.BusNames()[0].split(".")[0]
        cap_buses.add(bus)

    for idx, node_name in enumerate(sorted_node_names):
        parts = node_name.split(".")
        bus_name = parts[0]
        phase = int(parts[1]) if len(parts) > 1 else 1

        dss_module.Circuit.SetActiveBus(bus_name)
        kv_base_ln = dss_module.Bus.kVBase()
        if kv_base_ln <= 0:
            kv_base_ln = 2.401777  # Default 4.16kV line-to-line / sqrt(3)

        meta = NodeMetadata(
            node_idx=idx,
            bus_name=bus_name,
            phase=phase,
            node_id=node_name,
            base_kv_ln=kv_base_ln,
            is_slack=(bus_name == slack_bus),
            is_load=(bus_name in load_buses),
            is_cap=(bus_name in cap_buses),
            is_der=False,
        )
        nodes.append(meta)
        bus_to_nodes.setdefault(bus_name, []).append(idx)

    # Extract Lines, Transformers, Switches for edges
    edges: List[EdgeMetadata] = []
    G_nx = nx.Graph()
    for n in nodes:
        G_nx.add_node(n.node_idx, bus=n.bus_name, phase=n.phase, id=n.node_id)

    edge_list_src: List[int] = []
    edge_list_dst: List[int] = []
    edge_attr_list: List[List[float]] = []

    # Iterate over lines
    dss_module.Lines.First()
    num_lines = dss_module.Lines.Count()
    for _ in range(num_lines):
        line_name = dss_module.Lines.Name()
        bus1 = dss_module.Lines.Bus1()
        bus2 = dss_module.Lines.Bus2()
        phases = dss_module.Lines.Phases()
        length = dss_module.Lines.Length()
        r1 = dss_module.Lines.R1()
        x1 = dss_module.Lines.X1()
        c1 = dss_module.Lines.C1()

        b1_parts = bus1.split(".")
        b2_parts = bus2.split(".")
        b1_name = b1_parts[0]
        b2_name = b2_parts[0]

        if len(b1_parts) > 1:
            p1_list = [int(p) for p in b1_parts[1:]]
        else:
            p1_list = list(range(1, phases + 1))

        if len(b2_parts) > 1:
            p2_list = [int(p) for p in b2_parts[1:]]
        else:
            p2_list = list(range(1, phases + 1))

        for p1, p2 in zip(p1_list, p2_list):
            src_id = f"{b1_name}.{p1}"
            dst_id = f"{b2_name}.{p2}"
            if src_id in node_id_to_idx and dst_id in node_id_to_idx:
                u = node_id_to_idx[src_id]
                v = node_id_to_idx[dst_id]

                # Bidirectional graph edges
                r_val = float(r1) if r1 > 0 else 0.001
                x_val = float(x1) if x1 > 0 else 0.001
                b_val = float(c1) * 376.99 * 1e-9 if c1 > 0 else 0.0
                len_val = float(length) if length > 0 else 0.1

                e_fwd = EdgeMetadata(
                    edge_idx=len(edges),
                    from_bus=b1_name,
                    to_bus=b2_name,
                    from_node_idx=u,
                    to_node_idx=v,
                    element_type="line",
                    element_name=line_name,
                    r_series=r_val,
                    x_series=x_val,
                    b_shunt=b_val,
                    length=len_val,
                )
                edges.append(e_fwd)

                # Forward edge
                edge_list_src.append(u)
                edge_list_dst.append(v)
                edge_attr_list.append([r_val, x_val, b_val, len_val])

                # Backward edge
                edge_list_src.append(v)
                edge_list_dst.append(u)
                edge_attr_list.append([r_val, x_val, b_val, len_val])

                G_nx.add_edge(u, v, r=r_val, x=x_val, length=len_val, name=line_name)

        dss_module.Lines.Next()

    # Extract Transformers
    dss_module.Transformers.First()
    num_xfmr = dss_module.Transformers.Count()
    for _ in range(num_xfmr):
        xfmr_name = dss_module.Transformers.Name()
        buses = dss_module.CktElement.BusNames()
        if len(buses) >= 2:
            b1_parts = buses[0].split(".")
            b2_parts = buses[1].split(".")
            b1_name = b1_parts[0]
            b2_name = b2_parts[0]
            phases = dss_module.CktElement.NumPhases()

            p1_list = [int(p) for p in b1_parts[1:]] if len(b1_parts) > 1 else list(range(1, phases + 1))
            p2_list = [int(p) for p in b2_parts[1:]] if len(b2_parts) > 1 else list(range(1, phases + 1))

            for p1, p2 in zip(p1_list, p2_list):
                src_id = f"{b1_name}.{p1}"
                dst_id = f"{b2_name}.{p2}"
                if src_id in node_id_to_idx and dst_id in node_id_to_idx:
                    u = node_id_to_idx[src_id]
                    v = node_id_to_idx[dst_id]
                    r_val = 0.001
                    x_val = 0.005
                    b_val = 0.0
                    len_val = 0.01

                    e_xfmr = EdgeMetadata(
                        edge_idx=len(edges),
                        from_bus=b1_name,
                        to_bus=b2_name,
                        from_node_idx=u,
                        to_node_idx=v,
                        element_type="transformer",
                        element_name=xfmr_name,
                        r_series=r_val,
                        x_series=x_val,
                        b_shunt=b_val,
                        length=len_val,
                    )
                    edges.append(e_xfmr)
                    edge_list_src.extend([u, v])
                    edge_list_dst.extend([v, u])
                    edge_attr_list.extend([[r_val, x_val, b_val, len_val], [r_val, x_val, b_val, len_val]])
                    G_nx.add_edge(u, v, r=r_val, x=x_val, length=len_val, name=xfmr_name)

        dss_module.Transformers.Next()

    # Extract System Y matrix
    N = len(nodes)
    try:
        y_system = dss_module.Circuit.SystemY()
        # y_system is returned as flat array of real, imag pairs or complex matrix
        if y_system is not None and len(y_system) == 2 * N * N:
            y_real = np.array(y_system[0::2]).reshape((N, N))
            y_imag = np.array(y_system[1::2]).reshape((N, N))
            y_bus_dense = y_real + 1j * y_imag
        else:
            # Construct standard nodal admittance from extracted edges
            y_bus_dense = np.zeros((N, N), dtype=complex)
            for e in edges:
                u = e.from_node_idx
                v = e.to_node_idx
                z_series = complex(e.r_series, e.x_series)
                y_series = 1.0 / z_series if abs(z_series) > 1e-7 else 1e3
                y_shunt = complex(0.0, e.b_shunt / 2.0)
                y_bus_dense[u, v] -= y_series
                y_bus_dense[v, u] -= y_series
                y_bus_dense[u, u] += (y_series + y_shunt)
                y_bus_dense[v, v] += (y_series + y_shunt)
    except Exception as ex:
        logger.warning(f"Could not extract direct SystemY from OpenDSS: {ex}. Using synthesized Ybus.")
        y_bus_dense = np.zeros((N, N), dtype=complex)
        for e in edges:
            u = e.from_node_idx
            v = e.to_node_idx
            z_series = complex(e.r_series, e.x_series)
            y_series = 1.0 / z_series if abs(z_series) > 1e-7 else 1e3
            y_bus_dense[u, v] -= y_series
            y_bus_dense[v, u] -= y_series
            y_bus_dense[u, u] += y_series
            y_bus_dense[v, v] += y_series

    y_bus_sparse = sp.csr_matrix(y_bus_dense)

    # Build node features: [is_slack, is_load, is_cap, phase_1, phase_2, phase_3]
    node_feat_list: List[List[float]] = []
    for n in nodes:
        phase_1 = 1.0 if n.phase == 1 else 0.0
        phase_2 = 1.0 if n.phase == 2 else 0.0
        phase_3 = 1.0 if n.phase == 3 else 0.0
        node_feat_list.append([
            1.0 if n.is_slack else 0.0,
            1.0 if n.is_load else 0.0,
            1.0 if n.is_cap else 0.0,
            phase_1,
            phase_2,
            phase_3,
        ])

    if edge_list_src:
        edge_index = torch.tensor([edge_list_src, edge_list_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float32)

    node_features = torch.tensor(node_feat_list, dtype=torch.float32)

    feeder_graph = FeederGraph(
        name=feeder_name,
        nodes=nodes,
        edges=edges,
        node_id_to_idx=node_id_to_idx,
        bus_to_nodes=bus_to_nodes,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_features=node_features,
        y_bus_sparse=y_bus_sparse,
        y_bus_dense=y_bus_dense,
        networkx_graph=G_nx,
    )

    logger.info(
        f"Extracted topology: {feeder_graph.num_nodes} bus-phase nodes, "
        f"{feeder_graph.num_edges} directed graph edges."
    )
    return feeder_graph
