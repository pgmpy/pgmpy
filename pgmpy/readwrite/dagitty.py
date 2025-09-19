# pgmpy/readwrite/dagitty.py
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

# ---- Public API -----------------------------------------------------------

def serialize_to_dagitty(
    graph,
    graph_kind: str,
    *,
    allow_bidirected_in_dag_header: bool = True,
) -> str:
    """
    Serialize a pgmpy graph to dagitty syntax.

    Parameters
    ----------
    graph:
        A pgmpy graph instance (DAG/PDAG/ADMG/MAG).
        Must expose node attributes via `graph.nodes[data=True]` and
        edge accessors via the helpers below.
    graph_kind:
        One of: "dag", "pdag", "admg", "mag", "pag".
        Note: dagitty historically uses `dag { ... }` for ADMG as per the
        project note; we follow that by mapping "admg" -> "dag" header while
        still emitting "<->" edges if present.
    allow_bidirected_in_dag_header:
        dagitty accepts `<->` inside `dag {}` for ADMGs. Keep True.

    Returns
    -------
    str
        dagitty string like: dag { X [exposure]; Y [outcome]; X -> Y; U <-> Y }
    """
    header = _dagitty_header_for_kind(graph_kind)
    nodes_s = _nodes_with_roles_string(graph)
    dir_edges = list(iter_directed_edges(graph))
    bidi_edges = list(iter_bidirected_edges(graph))

    edges_parts: List[str] = []
    for u, v in dir_edges:
        edges_parts.append(f"{_id(u)} -> {_id(v)}")
    for u, v in bidi_edges:
        if header == "dag" and not allow_bidirected_in_dag_header:
            # If a consumer forbids this, refuse to serialize.
            raise ValueError("Bidirected edges not allowed under 'dag' header.")
        edges_parts.append(f"{_id(u)} <-> {_id(v)}")

    body = "; ".join(filter(None, [nodes_s] + edges_parts))
    return f"{header} {{ {body} }}"


def parse_from_dagitty(
    text: str,
) -> Tuple[str, List[str], List[Tuple[str, str]], List[Tuple[str, str]], Dict[str, Dict[str, bool]]]:
    """
    Parse dagitty text.

    Returns
    -------
    (kind, nodes, directed_edges, bidirected_edges, roles_by_node)

    kind: 'dag' | 'pdag' | 'mag' | 'pag' | 'adm g-as-dag'
          (for ADMG represented under `dag {}` we just return 'dag' and the caller decides)
    """
    cleaned = " ".join(text.strip().split())
    m = re.match(r"^(dag|pdag|mag|pag)\s*\{(.*)\}$", cleaned, flags=re.IGNORECASE)
    if not m:
        # Many ADMGs are represented with `dag { ... }`, so if no header, try to recover:
        raise ValueError("Unrecognized dagitty string: must start with dag|pdag|mag|pag { ... }")
    kind = m.group(1).lower()
    body = m.group(2).strip()

    # Split on ';' but ignore empty fragments
    parts = [p.strip() for p in body.split(";") if p.strip()]

    nodes: List[str] = []
    roles: Dict[str, Dict[str, bool]] = {}
    dir_e: List[Tuple[str, str]] = []
    bidi_e: List[Tuple[str, str]] = []

    node_decl_re = re.compile(r"^([A-Za-z0-9_./:-]+)(?:\s*\[(.+)\])?$")
    dir_re = re.compile(r"^([A-Za-z0-9_./:-]+)\s*->\s*([A-Za-z0-9_./:-]+)$")
    bidi_re = re.compile(r"^([A-Za-z0-9_./:-]+)\s*<->\s*([A-Za-z0-9_./:-]+)$")

    for part in parts:
        # Node decl with optional role tags
        nm = node_decl_re.match(part)
        if nm and ("->" not in part and "<->" not in part):
            node = nm.group(1)
            nodes.append(node)
            role_str = nm.group(2)
            if role_str:
                rflags = _parse_roles_list(role_str)
                roles[node] = rflags
            continue

        # Directed edge
        dm = dir_re.match(part)
        if dm:
            dir_e.append((dm.group(1), dm.group(2)))
            continue

        # Bidirected edge
        bm = bidi_re.match(part)
        if bm:
            bidi_e.append((bm.group(1), bm.group(2)))
            continue

        # Ignore unknown fragments? Better fail to keep IO strict.
        raise ValueError(f"Unrecognized dagitty fragment: {part}")

    # Ensure each mentioned node has a roles dict (possibly empty)
    for n in nodes:
        roles.setdefault(n, {})

    return kind, nodes, dir_e, bidi_e, roles


# ---- Internal utilities ---------------------------------------------------

def _dagitty_header_for_kind(kind: str) -> str:
    k = kind.lower()
    if k == "admg":
        # Project note: dagitty represents ADMG using dag { ... }.
        return "dag"
    if k in {"dag", "pdag", "mag", "pag"}:
        return k
    raise ValueError(f"Unsupported kind: {kind}")

def _id(x: str) -> str:
    # dagitty identifiers are typically bare; keep as-is and rely on caller to ensure valid names
    return x

def _nodes_with_roles_string(graph) -> str:
    parts: List[str] = []
    for n, data in getattr(graph, "nodes", lambda: [])(data=True):  # type: ignore
        flags: List[str] = []
        if data.get("exposure"):
            flags.append("exposure")
        if data.get("outcome"):
            flags.append("outcome")
        if data.get("latent"):
            flags.append("latent")
        if flags:
            parts.append(f"{_id(n)} [{', '.join(flags)}]")
        else:
            parts.append(f"{_id(n)}")
    return "; ".join(parts)

def _parse_roles_list(s: str) -> Dict[str, bool]:
    # Accept "exposure, outcome" or "exposure outcome" or "exposure"
    tokens = [t.strip().lower() for t in re.split(r"[,\s]+", s) if t.strip()]
    valid = {"exposure", "outcome", "latent"}
    flags = {k: False for k in valid}
    for t in tokens:
        if t not in valid:
            raise ValueError(f"Unknown role tag in dagitty: {t}")
        flags[t] = True
    return {k: v for k, v in flags.items() if v}

# The following edge iterators are resilient across different pgmpy graph types.

def iter_directed_edges(graph) -> Iterable[Tuple[str, str]]:
    # Preferred: explicit API on the graph
    for attr_name in ("directed_edges", "arcs", "edges_directed"):
        d = getattr(graph, attr_name, None)
        if callable(d):
            return d()  # type: ignore
        if isinstance(d, (list, tuple, set)):
            return iter(d)  # type: ignore

    # Generic fallback for directed graphs (e.g., DAG):
    is_dir = getattr(graph, "is_directed", None)
    if callable(is_dir) and is_dir():
        return graph.edges()  # type: ignore

    # Mixed graphs sometimes store directed edges under an attribute:
    getter = getattr(graph, "get_edges", None)
    if callable(getter):
        try:
            return getter("directed")  # type: ignore
        except Exception:
            pass

    # Nothing found: return empty
    return iter(())

def iter_bidirected_edges(graph) -> Iterable[Tuple[str, str]]:
    # Preferred explicit attributes
    for attr_name in ("bidirected_edges", "edges_bidirected"):
        d = getattr(graph, attr_name, None)
        if callable(d):
            return d()  # type: ignore
        if isinstance(d, (list, tuple, set)):
            return iter(d)  # type: ignore

    # Fallback: Some implementations store bidirected as undirected edges with a marker.
    edges_data = getattr(graph, "edges", None)
    if callable(edges_data):
        try:
            for u, v, data in graph.edges(data=True):  # type: ignore
                if isinstance(data, dict) and data.get("bidirected"):
                    yield (u, v)
        except Exception:
            pass

    return iter(())
