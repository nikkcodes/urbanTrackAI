"""
UrbanTrack AI - Camera Graph Regression Tests
============================================
Tests camera_graph.json topology, referential integrity, edge schemas,
and scenario S05 verified transitions (resolving isolated cameras C010 and C033).
"""

import json
from pathlib import Path
from collections import defaultdict
import pytest


GRAPH_PATH = Path("data/config/camera_graph.json")
LOCATIONS_PATH = Path("data/config/camera_locations.json")
DATASET_S05_PATH = Path(r"C:\Users\kanis\Downloads\AICity22_Track1_MTMC_Tracking\validation\S05")


@pytest.fixture(scope="module")
def camera_graph():
    assert GRAPH_PATH.exists(), f"Graph file not found: {GRAPH_PATH}"
    with open(GRAPH_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def camera_locations():
    assert LOCATIONS_PATH.exists(), f"Locations file not found: {LOCATIONS_PATH}"
    with open(LOCATIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_camera_graph_nodes_count(camera_graph):
    """Verify exactly 65 scenario-qualified nodes exist in camera_graph.json."""
    nodes = camera_graph.get("nodes", [])
    assert len(nodes) == 65, f"Expected 65 nodes, found {len(nodes)}"
    assert len(nodes) == len(set(nodes)), "Duplicate node entries found in graph nodes list"


def test_camera_graph_node_location_correspondence(camera_graph, camera_locations):
    """Verify 1:1 correspondence between camera_graph nodes and camera_locations entries."""
    graph_nodes = set(camera_graph.get("nodes", []))
    location_nodes = set(camera_locations.keys())
    assert graph_nodes == location_nodes, (
        f"Mismatch between graph nodes and locations: diff={graph_nodes ^ location_nodes}"
    )


def test_camera_graph_no_invalid_edge_references(camera_graph):
    """Verify every edge source and target exists in graph nodes."""
    nodes = set(camera_graph.get("nodes", []))
    edges = camera_graph.get("edges", [])
    invalid_edges = []
    for e in edges:
        src = e.get("source")
        tgt = e.get("target")
        if src not in nodes or tgt not in nodes:
            invalid_edges.append((src, tgt))
    assert not invalid_edges, f"Edges with invalid node references found: {invalid_edges}"


def test_camera_graph_no_duplicate_edges(camera_graph):
    """Verify no duplicate (source, target) directed edges exist."""
    edges = camera_graph.get("edges", [])
    seen = set()
    duplicates = []
    for e in edges:
        pair = (e.get("source"), e.get("target"))
        if pair in seen:
            duplicates.append(pair)
        seen.add(pair)
    assert not duplicates, f"Duplicate directed edges found: {duplicates}"


def test_camera_graph_no_self_loops(camera_graph):
    """Verify no self-loops (source == target) exist."""
    edges = camera_graph.get("edges", [])
    self_loops = [e for e in edges if e.get("source") == e.get("target")]
    assert not self_loops, f"Self-loops detected: {self_loops}"


def test_camera_graph_edge_schema(camera_graph):
    """Verify all edges conform strictly to the required schema."""
    edges = camera_graph.get("edges", [])
    assert len(edges) > 0, "No edges found in graph"
    valid_confidences = {"HIGH", "MEDIUM", "LOW"}

    for idx, e in enumerate(edges):
        assert "source" in e and isinstance(e["source"], str), f"Edge {idx} missing valid source"
        assert "target" in e and isinstance(e["target"], str), f"Edge {idx} missing valid target"
        assert e.get("relationship") == "possible_transition", (
            f"Edge {idx} ({e['source']}->{e['target']}) must use 'possible_transition'"
        )
        assert "distance_m" in e and isinstance(e["distance_m"], (int, float)), f"Edge {idx} missing distance_m"
        assert e["distance_m"] > 0, f"Edge {idx} distance must be positive: {e['distance_m']}"
        assert "bearing_deg" in e and isinstance(e["bearing_deg"], (int, float)), f"Edge {idx} missing bearing_deg"
        assert 0.0 <= e["bearing_deg"] < 360.0, f"Edge {idx} bearing out of [0, 360): {e['bearing_deg']}"
        assert "evidence" in e and isinstance(e["evidence"], list) and len(e["evidence"]) > 0, (
            f"Edge {idx} missing non-empty evidence list"
        )
        assert e.get("confidence") in valid_confidences, (
            f"Edge {idx} invalid confidence: {e.get('confidence')}"
        )


def test_s05_c010_transitions(camera_graph):
    """Verify CAM_S05_C010 is not isolated and has verified bidirectional transitions with CAM_S05_C017."""
    edges = camera_graph.get("edges", [])
    c010_outgoing = [e for e in edges if e["source"] == "CAM_S05_C010"]
    c010_incoming = [e for e in edges if e["target"] == "CAM_S05_C010"]

    assert len(c010_outgoing) > 0, "CAM_S05_C010 has no outgoing edges (isolated)"
    assert len(c010_incoming) > 0, "CAM_S05_C010 has no incoming edges (isolated)"

    # Verify bidirectional link with C017
    out_targets = {e["target"] for e in c010_outgoing}
    in_sources = {e["source"] for e in c010_incoming}
    assert "CAM_S05_C017" in out_targets, "Missing CAM_S05_C010 -> CAM_S05_C017 edge"
    assert "CAM_S05_C017" in in_sources, "Missing CAM_S05_C017 -> CAM_S05_C010 edge"


def test_s05_c033_transitions(camera_graph):
    """Verify CAM_S05_C033 is not isolated and has verified bidirectional transitions with CAM_S05_C034."""
    edges = camera_graph.get("edges", [])
    c033_outgoing = [e for e in edges if e["source"] == "CAM_S05_C033"]
    c033_incoming = [e for e in edges if e["target"] == "CAM_S05_C033"]

    assert len(c033_outgoing) > 0, "CAM_S05_C033 has no outgoing edges (isolated)"
    assert len(c033_incoming) > 0, "CAM_S05_C033 has no incoming edges (isolated)"

    out_targets = {e["target"] for e in c033_outgoing}
    in_sources = {e["source"] for e in c033_incoming}
    assert "CAM_S05_C034" in out_targets, "Missing CAM_S05_C033 -> CAM_S05_C034 edge"
    assert "CAM_S05_C034" in in_sources, "Missing CAM_S05_C034 -> CAM_S05_C033 edge"


def test_s05_connectivity_and_no_isolated_nodes(camera_graph):
    """Verify that no node in the graph (and specifically in S05) is isolated."""
    nodes = camera_graph.get("nodes", [])
    edges = camera_graph.get("edges", [])
    s05_nodes = [n for n in nodes if n.startswith("CAM_S05")]
    s05_edges = [e for e in edges if e["source"].startswith("CAM_S05")]

    # Build undirected adjacency for S05
    adj = defaultdict(set)
    for e in s05_edges:
        adj[e["source"]].add(e["target"])
        adj[e["target"]].add(e["source"])

    isolated_s05 = [n for n in s05_nodes if len(adj[n]) == 0]
    assert not isolated_s05, f"Isolated nodes found in S05: {isolated_s05}"

    # Weakly connected component count for S05
    visited = set()
    components = []
    for n in s05_nodes:
        if n not in visited:
            comp = []
            q = [n]
            visited.add(n)
            while q:
                curr = q.pop()
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)
            components.append(comp)

    assert len(components) == 1, (
        f"Expected S05 to form 1 connected component, but found {len(components)}: "
        f"{[len(c) for c in components]}"
    )


def test_s05_ground_truth_empirical_validation():
    """Verify empirical vehicle transitions in S05 gt.txt if dataset is present."""
    if not DATASET_S05_PATH.exists():
        pytest.skip(f"S05 dataset directory not available: {DATASET_S05_PATH}")

    camera_records = defaultdict(list)
    for cam_p in DATASET_S05_PATH.glob("c*"):
        gt_file = cam_p / "gt" / "gt.txt"
        if not gt_file.exists():
            continue
        c_name = f"CAM_S05_{cam_p.name.upper()}"
        with open(gt_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    frame = int(parts[0])
                    vid = int(parts[1])
                    camera_records[c_name].append((vid, frame))

    v_data = defaultdict(lambda: defaultdict(list))
    for cam, records in camera_records.items():
        for vid, frame in records:
            v_data[vid][cam].append(frame)

    vehicle_paths = {}
    for vid, cams in v_data.items():
        intervals = []
        for cam, frames in cams.items():
            intervals.append({"cam": cam, "start": min(frames), "end": max(frames)})
        intervals.sort(key=lambda x: x["start"])
        vehicle_paths[vid] = intervals

    consecutive_counts = defaultdict(int)
    for vid, path in vehicle_paths.items():
        for i in range(len(path) - 1):
            consecutive_counts[(path[i]["cam"], path[i+1]["cam"])] += 1

    # Assert transitions exist in dataset
    assert consecutive_counts[("CAM_S05_C010", "CAM_S05_C017")] >= 1, "C010 -> C017 transition missing in GT"
    assert consecutive_counts[("CAM_S05_C017", "CAM_S05_C010")] >= 1, "C017 -> C010 transition missing in GT"
    assert consecutive_counts[("CAM_S05_C033", "CAM_S05_C034")] >= 1, "C033 -> C034 transition missing in GT"
    assert consecutive_counts[("CAM_S05_C034", "CAM_S05_C033")] >= 1, "C034 -> C033 transition missing in GT"
    assert consecutive_counts[("CAM_S05_C034", "CAM_S05_C029")] >= 1, "C034 -> C029 transition missing in GT"
