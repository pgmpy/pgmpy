# pgmpy/tests/test_readwrite/test_dagitty_io.py
from __future__ import annotations

import re
import pytest

# Import your classes; adjust paths if your project organizes them differently.
from pgmpy.base.dag import DAG
from pgmpy.base.pdag import PDAG
from pgmpy.base.admg import ADMG
from pgmpy.base.mag import MAG

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def test_dag_roles_roundtrip():
    g = DAG()
    g.add_nodes_from(["X", "Y", "Z"])
    g.nodes["X"]["exposure"] = True
    g.nodes["Y"]["outcome"] = True
    g.add_edge("X", "Y")

    s = g.to_dagitty()
    assert "X [exposure]" in s
    assert "Y [outcome]" in s
    assert "X -> Y" in s

    g2 = DAG.from_dagitty(s)
    assert g2.has_edge("X", "Y")
    assert g2.nodes["X"].get("exposure") is True
    assert g2.nodes["Y"].get("outcome") is True
    assert not g2.nodes["Z"].get("exposure", False)

def test_admg_with_bidirected_and_roles_roundtrip():
    g = ADMG()
    g.add_nodes_from(["X", "Y", "U"])
    g.nodes["U"]["latent"] = True

    # directed: X -> Y
    g.add_edge("X", "Y")

    # bidirected: X <-> U (prefer method if available)
    add_bidi = getattr(g, "add_bidirected_edge", None)
    if callable(add_bidi):
        add_bidi("X", "U")
    else:
        g.add_edge("X", "U", bidirected=True)  # fallback

    s = g.to_dagitty()
    # ADMG serialized under dag { ... } by convention
    assert s.lower().startswith("dag {")
    assert "X -> Y" in s
    assert "X <-> U" in s
    assert "U [latent]" in s

    g2 = ADMG.from_dagitty(s)
    assert g2.has_edge("X", "Y")
    # Check one bidirected exists (best effort depending on class API)
    bidi = list(_iter_bidi(g2))
    assert ("X", "U") in bidi or ("U", "X") in bidi
    assert g2.nodes["U"].get("latent") is True

def test_mag_with_bidirected_roundtrip():
    g = MAG()
    g.add_nodes_from(["A", "B"])
    g.add_edge("A", "B")
    add_bidi = getattr(g, "add_bidirected_edge", None)
    if callable(add_bidi):
        add_bidi("B", "A")
    else:
        g.add_edge("B", "A", bidirected=True)

    s = g.to_dagitty()
    assert s.lower().startswith("mag {")
    assert "A -> B" in s
    assert "<->" in s

    g2 = MAG.from_dagitty(s)
    assert g2.has_edge("A", "B")
    bidi = list(_iter_bidi(g2))
    assert ("B", "A") in bidi or ("A", "B") in bidi

def _iter_bidi(g):
    # helper for tests only; tries common APIs
    for name in ("bidirected_edges", "edges_bidirected"):
        f = getattr(g, name, None)
        if callable(f):
            yield from f()
            return
    try:
        for u, v, d in g.edges(data=True):
            if d.get("bidirected"):
                yield (u, v)
    except Exception:
        pass
