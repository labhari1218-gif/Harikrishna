"""
sufficiency_metrics.py - Research-grade evidence sufficiency metrics

Implements publication-ready metrics aligned with gap-assessment research:
- ESI (Evidence Sufficiency Index): Multi-dimensional sufficiency score
- CR@k (Counter-Evidence Retention@k): Polarity-aware retention metric
- RPI@k (Recovery Potential Index@k): Backtrack-ready diagnostics

Uses stdlib BFS/DFS for graph connectivity (no networkx dependency).
"""

import math
from collections import deque, defaultdict
from typing import List, Dict, Set, Tuple

from .evidence import EvidenceItem


class SimpleGraph:
    """
    Simple undirected graph using stdlib only (no networkx).
    
    Supports:
    - Adding nodes and edges
    - BFS for connectivity checking
    - Connected components computation
    """
    
    def __init__(self):
        self.adj = defaultdict(set)  # adjacency list
        self.nodes = set()
    
    def add_edge(self, u: str, v: str):
        """Add undirected edge between u and v."""
        self.nodes.add(u)
        self.nodes.add(v)
        self.adj[u].add(v)
        self.adj[v].add(u)
    
    def is_connected(self, u: str, v: str) -> bool:
        """Check if u and v are connected via BFS."""
        if u not in self.nodes or v not in self.nodes:
            return False
        if u == v:
            return True
        
        # BFS from u to v
        visited = {u}
        queue = deque([u])
        
        while queue:
            current = queue.popleft()
            if current == v:
                return True
            
            for neighbor in self.adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return False
    
    def get_connected_components(self) -> List[Set[str]]:
        """Get all connected components using DFS."""
        visited = set()
        components = []
        
        for node in self.nodes:
            if node not in visited:
                # DFS to find component
                component = set()
                stack = [node]
                
                while stack:
                    current = stack.pop()
                    if current in visited:
                        continue
                    
                    visited.add(current)
                    component.add(current)
                    
                    for neighbor in self.adj[current]:
                        if neighbor not in visited:
                            stack.append(neighbor)
                
                components.append(component)
        
        return components


def build_entity_graph(evidence_items: List[EvidenceItem]) -> SimpleGraph:
    """
    Build undirected entity graph from evidence triples.
    
    Nodes: entities appearing as subj/obj in triples
    Edges: (subj, obj) for each triple (relation ignored for connectivity)
    
    Args:
        evidence_items: List of EvidenceItem with kind="kg_triple"
        
    Returns:
        SimpleGraph with entity connectivity
        
    Example:
        >>> items = [
        ...     EvidenceItem("e1", "kg_triple", {"triple": ["A", "rel", "B"]}),
        ...     EvidenceItem("e2", "kg_triple", {"triple": ["B", "rel", "C"]})
        ... ]
        >>> graph = build_entity_graph(items)
        >>> graph.is_connected("A", "C")
        True
    """
    graph = SimpleGraph()
    
    for item in evidence_items:
        if item.kind != "kg_triple":
            continue  # Skip non-triple evidence
        
        triple = item.content.get("triple", [])
        if len(triple) != 3:
            continue  # Skip malformed triples
        
        subj, rel, obj = triple
        graph.add_edge(subj, obj)
    
    return graph


def compute_connectivity(graph: SimpleGraph, claim_entities: List[str]) -> float:
    """
    Compute fraction of claim entity pairs connected in graph.
    
    connectivity = (# connected pairs) / (# total pairs)
    
    Edge cases:
    - If < 2 claim entities present in graph: return 1.0 (trivially connected)
    - If no claim entities: return 1.0
    
    Args:
        graph: Entity graph
        claim_entities: List of claim entity strings
        
    Returns:
        Connectivity score in [0, 1]
    """
    # Filter to entities present in graph
    present_entities = [e for e in claim_entities if e in graph.nodes]
    
    # Edge case: < 2 entities = trivially connected (1.0, not 0.0)
    if len(present_entities) < 2:
        return 1.0
    
    # Count connected pairs
    connected_count = 0
    total_pairs = 0
    
    for i, u in enumerate(present_entities):
        for v in present_entities[i+1:]:
            total_pairs += 1
            if graph.is_connected(u, v):
                connected_count += 1
    
    if total_pairs == 0:
        return 1.0
    
    return connected_count / total_pairs


def compute_esi(A: List[EvidenceItem], claim_entities: List[str]) -> Dict[str, float]:
    """
    Compute Evidence Sufficiency Index and components.
    
    Returns TWO ESI variants:
    - ESI_prod: M̂_A × Cov_A_smooth × Conn_A_smooth (original, conservative)
    - ESI_geom: (M̂_A × Cov_A_smooth × Conn_A_smooth)^(1/3) (geometric mean, more robust)
    
    ESI_geom is RECOMMENDED for decision-making because NLI models on triple 
    verbalizations produce conservative probabilities, and the product metric 
    collapses to very small values even when evidence is structurally sound.
    
    Smoothing (ε=0.05) prevents zero-collapse and improves interpretability.
    
    Edge cases:
    - If A empty: mass_A=0, esi=0, starve_score=1
    - If |E_c| == 0: coverage_A=1.0 (no entities to cover), connectivity_A=1.0
    - If < 2 claim entities in A: connectivity_A=1.0 (trivially connected)
    
    Args:
        A: Evidence set to evaluate sufficiency on.
           For symmetric sufficiency in current pipeline this should be A + C.
        claim_entities: List of claim entity strings
        
    Returns:
        Dict with keys:
        - esi_prod: Product-based ESI [0,1] (conservative)
        - esi_geom: Geometric mean ESI [0,1] (recommended)
        - starve_score_prod: 1 - esi_prod
        - starve_score_geom: 1 - esi_geom
        - mass_A: Unnormalized information mass (key kept for backward compatibility)
        - mass_A_normalized: M̂_A
        - coverage_A: Entity coverage [0,1] (raw, pre-smoothing; key kept for backward compatibility)
        - connectivity_A: Entity connectivity [0,1] (raw, pre-smoothing; key kept for backward compatibility)
    """
    EPSILON = 0.05  # Smoothing factor to prevent zero-collapse
    
    # Edge case: empty evidence set
    if not A:
        return {
            "esi_prod": 0.0,
            "esi_geom": 0.0,
            "starve_score_prod": 1.0,
            "starve_score_geom": 1.0,
            "mass_A": 0.0,
            "mass_A_normalized": 0.0,
            "coverage_A": 0.0,
            "connectivity_A": 0.0
        }
    
    # Informativeness uses relevance rel = p_entail + p_contra.
    # The caller controls whether this is A-only or A+C by passing the desired evidence set.
    mass_A = sum(e.pv.rel for e in A)
    alpha = max(1, len(A))
    mass_A_normalized = 1.0 - math.exp(-mass_A / alpha)
    
    # Edge case: no claim entities
    if not claim_entities:
        # No entities to cover or connect = trivially satisfied
        return {
            "esi_prod": mass_A_normalized,
            "esi_geom": mass_A_normalized,
            "starve_score_prod": 1.0 - mass_A_normalized,
            "starve_score_geom": 1.0 - mass_A_normalized,
            "mass_A": mass_A,
            "mass_A_normalized": mass_A_normalized,
            "coverage_A": 1.0,
            "connectivity_A": 1.0
        }
    
    # Build entity graph from the provided evidence set (A or A+C)
    graph = build_entity_graph(A)
    
    # Compute coverage
    entities_in_A = set(graph.nodes)
    claim_entities_set = set(claim_entities)
    coverage_A_raw = len(claim_entities_set & entities_in_A) / len(claim_entities_set)
    
    # Compute connectivity (now returns 1.0 for <2 entities)
    connectivity_A_raw = compute_connectivity(graph, claim_entities)
    
    # Apply smoothing to prevent zero-collapse
    coverage_A_smooth = EPSILON + (1 - EPSILON) * coverage_A_raw
    connectivity_A_smooth = EPSILON + (1 - EPSILON) * connectivity_A_raw
    
    # Compute BOTH ESI variants
    # Product (original): very conservative, collapses when NLI is cautious
    esi_prod = mass_A_normalized * coverage_A_smooth * connectivity_A_smooth
    
    # Geometric mean (recommended): more robust to conservative NLI scores
    esi_geom = (mass_A_normalized * coverage_A_smooth * connectivity_A_smooth) ** (1/3)
    
    return {
        "esi_prod": esi_prod,
        "esi_geom": esi_geom,
        "starve_score_prod": 1.0 - esi_prod,
        "starve_score_geom": 1.0 - esi_geom,
        "mass_A": mass_A,
        "mass_A_normalized": mass_A_normalized,
        "coverage_A": coverage_A_raw,  # Store raw for debugging
        "connectivity_A": connectivity_A_raw  # Store raw for debugging
    }


def compute_neutral_dominance(A: List[EvidenceItem]) -> float:
    """
    Compute neutral-dominance in Active set.
    
    NeutralRate_A = mean(p_neutral for e in A)
    
    Args:
        A: Active evidence set
        
    Returns:
        Mean p_neutral, or 0.0 if A empty
    """
    if not A:
        return 0.0
    
    return sum(e.pv.p_neutral for e in A) / len(A)


def compute_cr_at_k(
    *,
    pool_items: List[EvidenceItem],
    counter_items: List[EvidenceItem],
    ks: List[int] = None
) -> Dict[str, float]:
    """
    Compute Counter-Evidence Retention@k (keyword-only to prevent argument order bugs).
    
    CR@k = |TopContra_k(Pool) ∩ C| / min(k, |Pool|)
    
    Measures whether the strongest contradictions from the pool were retained in Counter set.
    
    Args:
        pool_items: Full evidence pool (keyword-only)
        counter_items: Counter evidence set C (keyword-only)
        ks: List of k values to compute (default: [5, 10]) (keyword-only)
        
    Returns:
        Dict with keys:
        - cr_at_5, cr_at_10: Retention scores
        - max_contra_all: Max p_contra in Pool
        - max_contra_C: Max p_contra in C (or 0.0 if C empty)
        - contra_mass_C: Sum of p_contra in C
    
    Example:
        >>> cr = compute_cr_at_k(pool_items=pool, counter_items=C, ks=[5, 10])
    """
    if ks is None:
        ks = [5, 10]
    
    result = {}
    
    # Sort pool by p_contra descending
    pool_sorted = sorted(pool_items, key=lambda e: e.pv.p_contra, reverse=True)
    C_ids = {e.evidence_id for e in counter_items}
    
    # Compute CR@k for each k
    for k in ks:
        top_k_ids = {e.evidence_id for e in pool_sorted[:k]}
        retained = len(top_k_ids & C_ids)
        # Fix for small pools: denominator should be min(k, pool_size)
        denom = min(k, len(pool_sorted))
        result[f"cr_at_{k}"] = retained / denom if denom > 0 else 0.0
    
    # Max contra in pool and C
    result["max_contra_all"] = pool_sorted[0].pv.p_contra if pool_sorted else 0.0
    result["max_contra_C"] = max((e.pv.p_contra for e in counter_items), default=0.0)
    
    # Contra mass in C
    result["contra_mass_C"] = sum(e.pv.p_contra for e in counter_items)
    
    return result


def compute_rpi_at_k(
    A: List[EvidenceItem],
    S: List[EvidenceItem],
    claim_entities: List[str],
    k_values: List[int] = [1, 3, 5]
) -> Dict[str, float]:
    """
    Compute Recovery Potential Index@k.
    
    RPI@k measures how much connectivity would improve by recovering
    k bridging triples from Suspended set.
    
    Algorithm:
    1. Build graph G_A from Active set
    2. Find connected components of G_A
    3. Identify suspended triples that bridge disconnected components
    4. Greedily add top-k bridging triples by rel
    5. RPI@k = Conn_(A+k) - Conn_A
    
    Args:
        A: Active evidence set
        S: Suspended evidence set
        claim_entities: List of claim entity strings
        k_values: List of k values (default: [1, 3, 5])
        
    Returns:
        Dict with keys:
        - bridge_count_S: Number of bridging triples in S
        - bridge_rel_mass_S: Sum of rel for bridging triples
        - rpi_at_1, rpi_at_3, rpi_at_5: Recovery potential indices
    """
    result = {}
    
    # Build graph from Active set
    graph_A = build_entity_graph(A)
    
    # Get initial connectivity
    conn_A = compute_connectivity(graph_A, claim_entities)
    
    # Get connected components
    components = graph_A.get_connected_components()
    
    # Map each node to its component index
    node_to_component = {}
    for idx, component in enumerate(components):
        for node in component:
            node_to_component[node] = idx
    
    # Find bridging triples in S
    bridging_triples = []
    for item in S:
        if item.kind != "kg_triple":
            continue
        
        triple = item.content.get("triple", [])
        if len(triple) != 3:
            continue
        
        subj, rel, obj = triple
        
        # Check if subj and obj are in different components
        comp_subj = node_to_component.get(subj)
        comp_obj = node_to_component.get(obj)
        
        # Bridge if: both nodes present in G_A and in different components
        if comp_subj is not None and comp_obj is not None and comp_subj != comp_obj:
            bridging_triples.append(item)
    
    # Sort bridges by rel descending
    bridging_triples.sort(key=lambda e: e.pv.rel, reverse=True)
    
    result["bridge_count_S"] = len(bridging_triples)
    result["bridge_rel_mass_S"] = sum(e.pv.rel for e in bridging_triples)
    
    # Compute RPI@k for each k
    for k in k_values:
        if k > len(bridging_triples):
            # Not enough bridges: use all available
            k_actual = len(bridging_triples)
        else:
            k_actual = k
        
        # Add top-k bridges to A
        A_plus_k = A + bridging_triples[:k_actual]
        
        # Recompute connectivity
        graph_A_plus_k = build_entity_graph(A_plus_k)
        conn_A_plus_k = compute_connectivity(graph_A_plus_k, claim_entities)
        
        # RPI@k = improvement in connectivity
        rpi = conn_A_plus_k - conn_A
        result[f"rpi_at_{k}"] = rpi
    
    return result
