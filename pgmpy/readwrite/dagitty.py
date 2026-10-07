import re

try:
    from pyparsing import (
        Combine,
        Group,
        Literal,
        OneOrMore,
        Or,
        ParseResults,
        QuotedString,
        Suppress,
        Word,
        ZeroOrMore,
        alphanums,
        nestedExpr,
        pyparsing_common,
    )
    from pyparsing import Optional as PyOptional
except ImportError as e:
    raise ImportError(
        f"{e}. pyparsing is required for using dagitty syntax. Please install using: pip install pyparsing"
    ) from None

# Map dagitty edge characters to _CoreGraph edge types
# dagitty: <->, ->, <-, --, @->, <-@, @-@, @--, --@
EDGE_MAP = {
    "->": "->",
    "<-": "<-",
    "<->": "<>",
    "--": "--",
    "@->": "o>",
    "<-@": "<o",
    "@-@": "oo",
    "@--": "o-",
    "--@": "-o",
}


class DagittyReader:
    def __init__(self, string: str | None = None, filename: str | None = None):
        if filename:
            with open(filename) as f:
                self.string = f.read()
        elif string:
            self.string = string
        else:
            raise ValueError("Either `string` or `filename` must be provided.")

        self.ebunch = []
        self.roles = {"exposures": set(), "outcomes": set(), "latents": set(), "adjusted": set(), "selected": set()}
        self.betas = {}
        self.nodes = set()

        self._parse()

    def _split_at_betas(self, lines: list[str]) -> list[str]:
        split_regex = r'(?<=\])\s+(?=[\w"\'`]+\s*(?:->|<-|<->|--|@->|<-@|@-@|@--|--@))'
        new_lines = []
        for line in lines:
            new_lines.extend(re.split(split_regex, line))
        return new_lines

    def _parse(self):
        content = self.string.strip()

        # Strip header and outer braces if present
        m = re.match(r"^\s*(?:[a-zA-Z]+\s*)?\{([\s\S]*)\}\s*$", content)
        if m:
            content = m.group(1)
        else:
            m = re.match(r"^\s*(?:[a-zA-Z]+\s*)?\{([\s\S]*)", content)
            if m:
                content = m.group(1)

        lines = [line.strip() for line in content.split("\n")]
        lines = self._split_at_betas(lines)

        var = Word(alphanums + "_" + ".") ^ QuotedString('"') ^ QuotedString("'") ^ QuotedString("`")
        option = nestedExpr("[", "]")
        var_stat = var + PyOptional(option)
        subgraph = nestedExpr("{", "}")
        var_or_subgraph = subgraph ^ var

        edge_choices = [Literal(e) for e in EDGE_MAP.keys()]
        edge = Or(edge_choices)

        beta = Suppress("[") + Group(Word("beta") + Suppress("=") + pyparsing_common.number()) + Suppress("]")

        edge_relation = (
            var_or_subgraph + OneOrMore(edge + var_or_subgraph) + PyOptional(beta.setResultsName("annotation"))
        )

        bb_re = Combine("bb=" + QuotedString('"'))
        pos_re = Combine("[pos=" + QuotedString('"') + "]")

        statement = (
            edge_relation.setResultsName("edge_stat*")
            ^ var_stat.setResultsName("var_stat*")
            ^ subgraph.setResultsName("edge_stat*")
            ^ bb_re
            ^ pos_re
        )

        dagitty_line = ZeroOrMore(statement + PyOptional(";"))

        def handle_edge_stat(edge_stat, betas):
            if not isinstance(edge_stat, ParseResults) and not isinstance(edge_stat, list):
                return {str(edge_stat).strip('"').strip("'").strip("`").rstrip(",")}

            length = len(edge_stat)
            if length == 1:
                return handle_edge_stat(edge_stat[0], betas)

            if length > 3:
                start_i = 0
                all_vars = set()
                while start_i < length - 1:
                    token = edge_stat[start_i + 1]

                    if token in EDGE_MAP:
                        end_i = start_i + 2
                    else:
                        end_i = start_i + 1

                    # Handle beta
                    if end_i + 1 < length and isinstance(edge_stat[end_i + 1], ParseResults):
                        if isinstance(edge_stat[end_i + 1][0], str) and edge_stat[end_i + 1][0] == "beta":
                            source = edge_stat[start_i]
                            target = edge_stat[end_i]
                            beta_val = edge_stat[end_i + 1][1]

                            src_vars = handle_edge_stat(source, betas)
                            tgt_vars = handle_edge_stat(target, betas)
                            for s in src_vars:
                                for t in tgt_vars:
                                    if t not in betas:
                                        betas[t] = {}
                                    betas[t][s] = beta_val

                            all_vars.update(handle_edge_stat(edge_stat[start_i : end_i + 1], betas))
                            start_i = end_i + 2
                            continue

                    all_vars.update(handle_edge_stat(edge_stat[start_i : end_i + 1], betas))
                    start_i = end_i

                return all_vars

            right_i = 1 if length == 2 else 2
            left_vars = handle_edge_stat(edge_stat[0], betas)
            right_vars = handle_edge_stat(edge_stat[right_i], betas)
            all_vars = left_vars.union(right_vars)

            if length == 2:
                return all_vars

            token = str(edge_stat[1])
            if token not in EDGE_MAP:
                raise ValueError(f"Malformed or unsupported edge token: {token}")

            mapped_edge = EDGE_MAP[token]

            for l in left_vars:
                for r in right_vars:
                    self.ebunch.append((l, r, mapped_edge))

            return all_vars

        for line in lines:
            line = line.strip()
            if not line:
                continue

            results = dagitty_line.parseString(line, parseAll=True)

            for var_s in results.get("var_stat", []):
                name = str(var_s[0]).strip("\"'").strip("`")
                self.nodes.add(name)
                if len(var_s) == 2:
                    opt = str(var_s[1][0]).rstrip(",").lower()
                    if opt.startswith("latent") or opt == "l":
                        self.roles["latents"].add(name)
                    elif opt.startswith("outcome") or opt.startswith("o"):
                        self.roles["outcomes"].add(name)
                    elif opt.startswith("exposure") or opt.startswith("e"):
                        self.roles["exposures"].add(name)
                    elif opt.startswith("adjusted") or opt.startswith("a"):
                        self.roles["adjusted"].add(name)
                    elif opt.startswith("selected") or opt.startswith("s"):
                        self.roles["selected"].add(name)

            for edge_s in results.get("edge_stat", []):
                handle_edge_stat(edge_s, self.betas)

            # If there's an annotation on a top level edge_relation length 3
            annotation = results.get("annotation")
            if annotation and isinstance(annotation, ParseResults):
                if annotation[0] == "beta":
                    # apply to the last edge processed?
                    pass

        for u, v, _ in self.ebunch:
            self.nodes.add(u)
            self.nodes.add(v)


class DagittyWriter:
    def __init__(self, model):
        self.model = model

    def _quote(self, name):
        name = str(name)
        if re.search(r"\s|-|@|<|>", name):
            return f'"{name}"'
        return name

    def write(self) -> str:
        # Determine header based on class name
        model_type = type(self.model).__name__.lower()
        if model_type not in ("dag", "mag", "pag", "pdag", "admg"):
            model_type = "dag"

        lines = [f"{model_type} {{"]

        # Extract roles
        # Support mixed usage
        node_statements = []
        nodes_written = set()
        for node in self.model.nodes():
            if getattr(self.model, "get_role", None):
                # We can dynamically fetch roles from model
                pass

            # Use data dict
            data = self.model.nodes[node]
            roles = data.get("roles", set())

            opts = []
            if "latents" in roles:
                opts.append("latent")
            if "outcomes" in roles:
                opts.append("outcome")
            if "exposures" in roles:
                opts.append("exposure")
            if "adjusted" in roles:
                opts.append("adjusted")
            if "selected" in roles:
                opts.append("selected")

            if opts:
                node_statements.append(f"{self._quote(node)} [{', '.join(opts)}]")
                nodes_written.add(node)
            elif self.model.degree(node) == 0:
                node_statements.append(f"{self._quote(node)}")
                nodes_written.add(node)

        lines.extend(node_statements)

        # Betas
        betas = {}
        if hasattr(self.model, "cpds"):
            from pgmpy.factors.continuous import LinearGaussianCPD

            for cpd in self.model.cpds:
                if isinstance(cpd, LinearGaussianCPD):
                    tgt = cpd.variable
                    for i, ev in enumerate(cpd.evidence):
                        if tgt not in betas:
                            betas[tgt] = {}
                        betas[tgt][ev] = cpd.beta[i + 1]

        # Edges
        # reverse map
        REVERSE_MAP = {v: k for k, v in EDGE_MAP.items()}

        if type(self.model).__name__ in ("DAG", "LinearGaussianBayesianNetwork"):
            for u, v in sorted(self.model.edges(), key=lambda x: (str(x[0]), str(x[1]))):
                edge_str = f"{self._quote(u)} -> {self._quote(v)}"
                if v in betas and u in betas[v]:
                    edge_str += f" [beta={betas[v][u]}]"
                lines.append(edge_str)
        else:
            for u, v, edge_type in sorted(
                self.model.get_edges(data=True), key=lambda x: (str(x[0]), str(x[1]), str(x[2]))
            ):
                token = REVERSE_MAP.get(edge_type, "->")
                lines.append(f"{self._quote(u)} {token} {self._quote(v)}")

        lines.append("}")
        return "\n".join(lines)
