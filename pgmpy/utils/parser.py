def parse_lavaan(lines):
    # Step 0: Check if pyparsing is installed
    try:
        from pyparsing import (
            OneOrMore,
            Optional,
            Suppress,
            Word,
            alphanums,
            nums,
        )
    except ImportError as e:
        raise ImportError(
            e.msg
            + ". pyparsing is required for using lavaan syntax. Please install using: pip install pyparsing"
        )

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
