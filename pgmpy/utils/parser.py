def parse_lavaan(lines):
    # Step 0: Check if pyparsing is installed
    try:
        from pyparsing import OneOrMore, Optional, Suppress, Word, alphanums, nums
    except ImportError as e:
        raise ImportError(
            e.msg
            + ". pyparsing is required for using lavaan syntax. Please install using: pip install pyparsing"
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

    def handle_edge_stat(edge_stat, latents, ebunch):
        all_vars = set()
        if not isinstance(edge_stat, ParseResults) and not isinstance(edge_stat, list):
            return set([edge_stat.strip('"')])
        # if edge_stat is longer than 3, then treat one edge symbol at once. Call multiple times
        if (isinstance(edge_stat, ParseResults) or isinstance(edge_stat, list)) and len(
            edge_stat
        ) > 3:
            l = len(edge_stat)
            start_i = 0
            while start_i < l - 1:
                # Parse {a -> b -> c}
                if edge_stat[start_i + 1] in ["->", "<-", "<->"]:
                    end_i = start_i + 2
                # Parse {a b c d}
                else:
                    end_i = start_i + 1

                all_vars.update(
                    handle_edge_stat(edge_stat[start_i : end_i + 1], latents, ebunch)
                )
                start_i = end_i

            return all_vars

        l = len(edge_stat)
        right_i = 1 if l == 2 else 2
        # length is three. Now check if any node is a subgraph
        left_vars = handle_edge_stat(edge_stat[0], latents, ebunch)
        right_vars = handle_edge_stat(edge_stat[right_i], latents, ebunch)
        all_vars.update(left_vars)
        all_vars.update(right_vars)

        # No edges created for subgraph {X Y}
        if l == 2:
            return all_vars

        # Now connect the every pair of left and right vars with the given edge for {X <- Y}
        for left_var in list(left_vars):
            for right_var in list(right_vars):
                # connect with the given edge.
                if edge_stat[1] == "->":
                    ebunch.append((left_var, right_var))
                elif edge_stat[1] == "<-":
                    ebunch.append((right_var, left_var))
                elif edge_stat[1] == "<->":
                    # add to latent
                    latent_var = (
                        f"u_{left_var}_{right_var}"
                        if left_var < right_var
                        else f"u_{right_var}_{left_var}"
                    )
                    latents.append(latent_var)
                    ebunch.append((latent_var, left_var))
                    ebunch.append((latent_var, right_var))
                    # latent variable from subgraph
                    all_vars.add(latent_var)
                else:
                    print("unknown edge type")

        return all_vars

    # Step 0: Check if pyparsing is installed
    try:
        from pyparsing import (
            Combine,
            OneOrMore,
            Optional,
            ParseResults,
            QuotedString,
            Word,
            ZeroOrMore,
            alphanums,
            nestedExpr,
        )
    except ImportError as e:
        raise ImportError(
            e.msg
            + ". pyparsing is required for using dagitty syntax. Please install using: pip install pyparsing"
        ) from None

    # Step 1: DAGitty Grammar in pyparsing
    # Reference: https://www.dagitty.net/manual-3.x.pdf#page=3.58
    # Drawing and Analyzing Causal DAGs with DAGitty by Johannes Textor
    # Variable name like X.1, a_b, 123. Double-quote if with special characters
    var = Word(alphanums + "_" + ".") ^ QuotedString('"')
    # Exposure, outcome, latent, adjusted
    option = nestedExpr("[", "]")
    # The variable statements: variable name + list of option(s)
    var_stat = var + Optional(option)
    # { } open a new scope for a subgraph
    subgraph = nestedExpr("{", "}")
    # arrow can point to a variable or subgraph.
    var_or_subgraph = subgraph ^ var
    # edge type (which can be ->, <-, or <->)
    edge = Word("><-")
    # edge chaining
    edge_relation = var_or_subgraph + OneOrMore(edge + var_or_subgraph)

    # Display info bb="1,2,3,4", [pos="1,2"] will be parsed and discarded
    bb_re = Combine("bb=" + QuotedString('"'))
    pos_re = Combine("[pos=" + QuotedString('"') + "]")

    # If possible, try to match with edge_relation with arrow, rather than only reading varnames as var_stat
    statement = (
        edge_relation.setResultsName("edge_stat*")
        ^ var_stat.setResultsName("var_stat*")
        ^ bb_re
        ^ pos_re
    )

    # different statements on the same line without semicolon
    dagitty_line = ZeroOrMore(statement + Optional(";"))

    # Step 2:
    # Clean the opening of the enclosing dag{ .. } or dag Smoking { .. }
    cleaned_dag = False
    while True:
        first_line = lines.pop(0).strip()
        if first_line:
            # Try to find and remove "dag"
            if not cleaned_dag:
                assert first_line[:3] == "dag"
                cleaned_dag = True
                first_line = first_line[3:]
            # Try to find and remove {, either same line as 'dag' or next lines.
            if (
                cleaned_dag
            ):  # Do not change this with else. cleaned_dag could have changed.
                start_loc = first_line.find("{")
                if start_loc >= 0:
                    first_line = first_line[start_loc + 1 :].strip()
                    lines.insert(0, first_line)
                    break

    # Clean the tail of the enclosing dag{ .. } or dag Smoking { .. }
    while True:
        # Search iteratively from the last line
        last_line = lines.pop().strip()
        if last_line:
            assert last_line[-1] == "}", "dag { }"
            lines.append(last_line[:-1])
            break
    # Step 3: Initialize arguments and fill them by parsing each line.
    ebunch = []
    latents = []
    for line in lines:
        line = line.strip()
        if line != "":
            results = dagitty_line.parseString(line, parseAll=True)

            for var_stat in results.get("var_stat", []):
                if len(var_stat) == 2:
                    option = var_stat[1][0].lower()
                    if (
                        option[:6] == "latent"
                        or option == "l"
                        or option.startswith("l,")
                    ):
                        latents.append(var_stat[0].strip('"'))
            for edge_stat in results.get("edge_stat", []):
                handle_edge_stat(edge_stat, latents, ebunch)

    return ebunch, latents
