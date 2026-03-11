def parse_lavaan(lines):
    # Step 0: Check if pyparsing is installed
    try:
        from pyparsing import OneOrMore, Optional, Suppress, Word, alphanums
    except ImportError as e:
        raise ImportError(
            f"{e}. pyparsing is required for using lavaan syntax. Please install using: pip install pyparsing"
        ) from None

    # Step 1: Define the grammar for each type of string.
    var = Word(alphanums)
    reg_gram = (
        OneOrMore(
            var.setResultsName("predictors", listAllMatches=True)
            + Optional(Suppress("+"))
        )
        + "~"
        + OneOrMore(
            var.setResultsName("covariates", listAllMatches=True)
            + Optional(Suppress("+"))
        )
    )
    intercept_gram = var("inter_var") + "~" + Word("1")
    covar_gram = (
        var("covar_var1")
        + "~~"
        + OneOrMore(
            var.setResultsName("covar_var2", listAllMatches=True)
            + Optional(Suppress("+"))
        )
    )
    latent_gram = (
        var("latent")
        + "=~"
        + OneOrMore(
            var.setResultsName("obs", listAllMatches=True) + Optional(Suppress("+"))
        )
    )

    # Step 2: Preprocess string to lines

    # Step 3: Initialize arguments and fill them by parsing each line.
    ebunch = []
    latents = []
    err_corr = []
    err_var = []
    for line in lines:
        line = line.strip()
        if (line != "") and (not line.startswith("#")):
            if intercept_gram.matches(line):
                continue
            elif reg_gram.matches(line):
                results = reg_gram.parseString(line, parseAll=True)
                for pred in results["predictors"]:
                    ebunch.extend(
                        [(covariate, pred) for covariate in results["covariates"]]
                    )
            elif covar_gram.matches(line):
                results = covar_gram.parseString(line, parseAll=True)
                for var in results["covar_var2"]:
                    err_corr.append((results["covar_var1"], var))

            elif latent_gram.matches(line):
                results = latent_gram.parseString(line, parseAll=True)
                latents.append(results["latent"])
                ebunch.extend([(results["latent"], obs) for obs in results["obs"]])
    return ebunch, latents, err_corr, err_var


def parse_dagitty(lines):
    def handle_edge_stat(edge_stat, latents, ebunch, betas):
        all_vars = set()
        # ParseResults type is resolved at call time (ParseResults imported later)
        if not isinstance(edge_stat, ParseResults) and not isinstance(edge_stat, list):
            return {edge_stat.strip('"').strip("'").rstrip(",")}

        length = len(edge_stat)

        # Handle wrapper group { a } or { X -> Y }
        # These parse as ['a'] or [['X', '->', 'Y']], both have length 1.
        if length == 1:
            # This is a group with one item; recurse on the item itself.
            return handle_edge_stat(edge_stat[0], latents, ebunch, betas)

        # If longer than 3, split into smaller edge/stat parts and handle recursively
        if (isinstance(edge_stat, ParseResults) or isinstance(edge_stat, list)) and len(
            edge_stat
        ) > 3:
            length = len(edge_stat)
            start_i = 0
            while start_i < length - 1:
                # Parse {a -> b -> c} or {a <- b <- c} etc
                if edge_stat[start_i + 1] in [
                    "->",
                    "<-",
                    "<->",
                    "o->",
                    "<-o",
                    "o-o",
                    "--",
                ]:
                    end_i = start_i + 2
                else:
                    # subgraph/list of variables {a b c}
                    end_i = start_i + 1

                all_vars.update(
                    handle_edge_stat(
                        edge_stat[start_i : end_i + 1], latents, ebunch, betas
                    )
                )

                # Parse `edge` [beta=float]
                if end_i + 1 < length and isinstance(
                    edge_stat[end_i + 1], ParseResults
                ):
                    if (
                        isinstance(edge_stat[end_i + 1][0], str)
                        and edge_stat[end_i + 1][0] == "beta"
                    ):
                        source = edge_stat[start_i]
                        target = edge_stat[end_i]
                        beta = edge_stat[end_i + 1][1]
                        if target not in betas:
                            betas[target] = {}
                        betas[target][source] = beta
                        break
                else:
                    start_i = end_i

            return all_vars

        # Now edge_stat length is 1 or 2 or 3
        length = len(edge_stat)
        right_i = 1 if length == 2 else 2

        # resolve left and right vars (may be subgraphs)
        left_vars = handle_edge_stat(edge_stat[0], latents, ebunch, betas)
        right_vars = handle_edge_stat(edge_stat[right_i], latents, ebunch, betas)
        all_vars.update(left_vars)
        all_vars.update(right_vars)

        # No edges created for subgraph {X Y}
        if length == 2:
            return all_vars

        # token representing the edge symbol, e.g., "->", "<-",
        token = edge_stat[1]

        # Helper to map visual characters to mark characters used by MAG/PAG
        def char_to_mark(c):
            # '<' means an arrowhead at that endpoint (visual '<' -> mark '>')
            if c == "<":
                return ">"
            # Dagitty can use '@' to represent circle endpoints for PAG; map to 'o'
            if c == "@":
                return "o"
            if c in (">", "o", "-"):
                return c
            # fallback (shouldn't happen)
            return c

        # For MAG/PAG we want to create 4-tuple edges (u, v, tail_mark_on_u, head_mark_on_v)
        # For DAG we keep the old behavior (u, v) and create artificial latent for <->.
        for left_var in sorted(left_vars):
            for right_var in sorted(right_vars):
                if target_type.upper() in ("MAG", "PAG", "PDAG"):
                    t = str(token)
                    left_mark = char_to_mark(t[0])
                    right_mark = char_to_mark(t[-1])

                    # Normalize directed edges so that directed edge is stored as (tail, head, "-", ">")
                    # i.e., if visual marks indicate right-to-left arrow (left_mark==">" and right_mark=="-"),
                    # we reverse the order to keep tail '-' first and head '>' second.
                    if (left_mark, right_mark) == ("-", ">"):
                        ebunch.append((left_var, right_var, "-", ">"))
                    elif (left_mark, right_mark) == (">", "-"):
                        ebunch.append((right_var, left_var, "-", ">"))
                    else:
                        # other cases: bidirected (">", ">"), circle marks ("o", ">"), ("o", "o"), ("-", "-"), etc.
                        ebunch.append((left_var, right_var, left_mark, right_mark))
                else:
                    # DAG behavior (backwards compatible)
                    if token == "->":
                        ebunch.append((left_var, right_var))
                    elif token == "<-":
                        ebunch.append((right_var, left_var))
                    elif token == "<->":
                        # represent bidirected in DAG-as-MAG form using an artificial latent confounder
                        latent_var = (
                            f"u_{left_var}_{right_var}"
                            if left_var < right_var
                            else f"u_{right_var}_{left_var}"
                        )
                        latents.append(latent_var)
                        ebunch.append((latent_var, left_var))
                        ebunch.append((latent_var, right_var))
                        all_vars.add(latent_var)
                    else:
                        # unknown token for DAG; keep graceful fallback
                        ebunch.append((left_var, right_var))

        return all_vars

    def split_at_betas(lines):
        import re

        # allow single or double quoted names before arrow when splitting on beta annotations
        split_regex = r'(?<=\])\s+(?=[\w"\'`]+\s*->)'
        new_dag_lines = []
        for line in lines:
            split_lines = re.split(split_regex, line)
            new_dag_lines.extend(split_lines)
        return new_dag_lines

    # Step 0: Check if pyparsing is installed
    try:
        from pyparsing import (
            Combine,
            Group,
            OneOrMore,
            Optional,
            ParseResults,
            QuotedString,
            Suppress,
            Word,
            ZeroOrMore,
            alphanums,
            nestedExpr,
            pyparsing_common,
        )
    except ImportError as e:
        raise ImportError(
            f"{e}. pyparsing is required for using dagitty syntax. Please install using: pip install pyparsing"
        ) from None

    # Infer graph type from header (e.g. "dag {", "mag {", "pag {") if present.
    # If header is present, override target_type to follow the file.
    import re

    first_nonempty = None
    for ln in lines:
        if isinstance(ln, str) and ln.strip():
            first_nonempty = ln.strip()
            break
    if first_nonempty is not None:
        m = re.match(r"^\s*(\w+)", first_nonempty, flags=re.IGNORECASE)
        if m:
            hdr = m.group(1).lower()
            if hdr in ("dag", "mag", "pag"):
                target_type = hdr.upper()

    # Step 1: DAGitty Grammar in pyparsing
    # Variable name like X.1, a_b, 123. Support single or double quoted names with spaces
    var = Word(alphanums + "_" + ".") ^ QuotedString('"') ^ QuotedString("'")
    option = nestedExpr("[", "]")
    var_stat = var + Optional(option)
    subgraph = nestedExpr("{", "}")
    var_or_subgraph = subgraph ^ var

    # include '@' (Dagitty's circle character) when parsing PAG; map later to 'o'
    edge_chars = "><-@" if target_type.upper() == "PAG" else "><-"
    edge = Word(edge_chars)

    beta = (
        Suppress("[")
        + Group(Word("beta") + Suppress("=") + pyparsing_common.number())
        + Suppress("]")
    )

    edge_relation = (
        var_or_subgraph
        + OneOrMore(edge + var_or_subgraph)
        + Optional(beta.setResultsName("annotation"))
    )

    bb_re = Combine("bb=" + QuotedString('"'))
    pos_re = Combine("[pos=" + QuotedString('"') + "]")

    statement = (
        edge_relation.setResultsName("edge_stat*")
        ^ var_stat.setResultsName("var_stat*")
        ^ subgraph.setResultsName("edge_stat*")  # <-- Add subgraph as a valid statement
        ^ bb_re
        ^ pos_re
    )

    dagitty_line = ZeroOrMore(statement + Optional(";"))

    # Step 2: Preprocess lines and strip outer dag { ... }
    lines = split_at_betas(lines)
    cleaned_dag = False
    while True:
        if not lines:
            break
        first_line = lines.pop(0).strip()
        if first_line:
            if not cleaned_dag:
                # Accept headers "dag", "mag", or "pag" (case insensitive).
                # Remove the header token (whatever it is) instead of assuming "dag".
                m_hdr = re.match(r"^\s*(\w+)", first_line, flags=re.IGNORECASE)
                if m_hdr and m_hdr.group(1).lower() in ("dag", "mag", "pag", "pdag"):
                    cleaned_dag = True
                    # remove the header token from the start so the "{" is handled below
                    first_line = first_line[m_hdr.end() :]
            start_loc = first_line.find("{")
            if start_loc >= 0:
                first_line = first_line[start_loc + 1 :].strip()
                lines.insert(0, first_line)
                break

    while True:
        if not lines:
            break
        last_line = lines.pop().strip()
        if last_line:
            end_loc = last_line.rfind("}")
            if end_loc != -1:
                last_line = last_line[:end_loc]
                lines.append(last_line)
            else:
                lines.append(last_line)
            break

    # Step 3: Parse lines
    ebunch = []
    roles = {"outcomes": [], "exposures": [], "latents": []}
    latents = roles["latents"]
    betas = {}
    nodes = set()
    for line in lines:
        line = line.strip()
        if line != "":
            results = dagitty_line.parseString(line, parseAll=True)

            for var_stat in results.get("var_stat", []):
                name = var_stat[0]
                if isinstance(name, str):
                    name = name.strip("\"'")
                nodes.add(name)
                if len(var_stat) == 2:
                    option = str(var_stat[1][0]).rstrip(",").lower()
                    # latent markers: 'latent', 'latents', 'l'
                    if option.startswith("latent") or option == "l":
                        roles["latents"].append(name)
                    elif option.startswith("outcome") or option.startswith("o"):
                        roles["outcomes"].append(name)
                    elif option.startswith("exposure") or option.startswith("e"):
                        roles["exposures"].append(name)
            for edge_stat in results.get("edge_stat", []):
                handle_edge_stat(edge_stat, latents, ebunch, betas)

    for e in ebunch:
        # ebunch items can be 2-tuples or 4-tuples
        nodes.add(e[0])
        nodes.add(e[1])

    return ebunch, roles, betas, nodes
