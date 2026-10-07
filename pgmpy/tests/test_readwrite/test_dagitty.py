import unittest

from pgmpy.base import ADMG, DAG, MAG, PDAG


class TestDagittyReadWrite(unittest.TestCase):
    def test_dag_roundtrip(self):
        s = 'dag {\n"a b" [exposure]\nc [outcome]\n"a b" -> c [beta=0.5]\n}'
        d = DAG.from_dagitty(s)
        self.assertIn("a b", d.nodes())
        self.assertIn("c", d.nodes())

        # wait, it might be LGBN because of beta
        # let's write a simple DAG
        s2 = 'dag {\n"a b" [exposure]\nc [outcome]\n"a b" -> c\n}'
        d2 = DAG.from_dagitty(s2)
        s2_out = d2.to_dagitty()
        self.assertIn('"a b"', s2_out)
        self.assertIn("c [outcome]", s2_out)

    def test_mag_roundtrip(self):
        s = "mag {\na [exposure]\nb [outcome]\na -> b\na <-> b\na -- b\n}"
        m = MAG.from_dagitty(s)
        s_out = m.to_dagitty()
        self.assertIn("a -> b", s_out)
        self.assertIn("a <-> b", s_out)
        self.assertIn("a -- b", s_out)

    def test_admg_roundtrip(self):
        s = "admg {\na [exposure]\nb [outcome]\na -> b\na <-> b\n}"
        m = ADMG.from_dagitty(s)
        s_out = m.to_dagitty()
        self.assertIn("a -> b", s_out)
        self.assertIn("a <-> b", s_out)

    def test_pdag_roundtrip(self):
        s = "pdag {\na\nb\na -> b\na -- b\n}"
        m = PDAG.from_dagitty(s)
        s_out = m.to_dagitty()
        self.assertIn("a -> b", s_out)
        self.assertIn("a -- b", s_out)

    def test_edge_types(self):
        # Test directed edge
        s1 = "mag {\na -> b\n}"
        m1 = MAG.from_dagitty(s1)
        self.assertIn("a -> b", m1.to_dagitty())
        
        # Test reverse edge with different nodes
        s2 = "mag {\nb <- a\n}"
        m2 = MAG.from_dagitty(s2)
        self.assertIn("b <- a", m2.to_dagitty())
        
        # Test bidirectional edge
        s3 = "mag {\na <-> b\n}"
        m3 = MAG.from_dagitty(s3)
        self.assertIn("a <-> b", m3.to_dagitty())
        
        # Test undirected edge
        s4 = "mag {\na -- b\n}"
        m4 = MAG.from_dagitty(s4)
        self.assertIn("a -- b", m4.to_dagitty())


if __name__ == "__main__":
    unittest.main()
