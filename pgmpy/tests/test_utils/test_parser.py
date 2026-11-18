from pgmpy.utils.parser import parse_dagitty, parse_lavaan


class TestParser:
    def test_parse_lavaan_regression(self):
        lavaan_str = """
        # this is a comment
        y ~ x1 + x2
        f1 =~ y1 + y2 + y3
        y1 ~~ y2
        """
        lines = lavaan_str.strip().split("\n")
        ebunch, latents, err_corr, err_var = parse_lavaan(lines)

        expected_ebunch = [
            ("x1", "y"),
            ("x2", "y"),
            ("f1", "y1"),
            ("f1", "y2"),
            ("f1", "y3"),
        ]
        assert set(ebunch) == set(expected_ebunch)
        assert latents == ["f1"]
        assert err_corr == [("y1", "y2")]
        assert err_var == []

    def test_parse_dagitty_dag(self):
        dagitty_str = "dag { A -> B \n C -> D \n B <-> C }"
        lines = dagitty_str.strip().split("\n")
        ebunch, roles, betas, nodes = parse_dagitty(lines)

        assert ("A", "B") in ebunch
        assert ("C", "D") in ebunch

        latent = "u_B_C"
        assert (latent, "B") in ebunch
        assert (latent, "C") in ebunch

        assert roles["latents"] == [latent]
        assert roles["outcomes"] == []
        assert roles["exposures"] == []
        assert set(nodes) == {"A", "B", "C", "D", latent}
        assert betas == {}

    def test_parse_dagitty_mag(self):
        mag_str = "mag { A -> B \n C <-> D }"
        lines = mag_str.strip().split("\n")
        ebunch, roles, betas, nodes = parse_dagitty(lines)

        assert ("A", "B", "-", ">") in ebunch
        assert ("C", "D", ">", ">") in ebunch or ("D", "C", ">", ">") in ebunch
        assert set(nodes) == {"A", "B", "C", "D"}

    def test_parse_dagitty_betas(self):
        dagitty_str = "dag { X -> Y [beta=0.3] \n Y -> Z [beta=0.1] }"
        lines = dagitty_str.strip().split("\n")
        ebunch, roles, betas, nodes = parse_dagitty(lines)

        assert ("X", "Y") in ebunch
        assert ("Y", "Z") in ebunch
        assert betas == {"Y": {"X": 0.3}, "Z": {"Y": 0.1}}

    def test_parse_dagitty_roles_and_subgraphs(self):
        dagitty_str = 'dag { bb="0,0,1,1" \n X [exposure] \n Y [outcome] \n U [latent] \n {X U} -> Y }'
        lines = dagitty_str.strip().split("\n")
        ebunch, roles, betas, nodes = parse_dagitty(lines)

        assert ("X", "Y") in ebunch
        assert ("U", "Y") in ebunch

        assert "X" in roles["exposures"]
        assert "Y" in roles["outcomes"]
        assert "U" in roles["latents"]
