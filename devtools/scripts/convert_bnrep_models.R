#!/usr/bin/env Rscript

################################################################################
# Script: convert_bnrep_models.R
# Purpose: Convert a single Bayesian network model from the bnRep repository to
#          BIF format (discrete) or JSON format (continuous/Gaussian)
# Author: pgmpy development team
# License: MIT
#
# Requirements:
#   - R packages: bnRep, bnlearn, jsonlite, jsonvalidate
#   - Install with: install.packages(c("bnRep", "bnlearn", "jsonlite", "jsonvalidate"))
#
# Usage:
#   Rscript convert_bnrep_models.R [--model=MODEL_NAME] [--output_folder=DIR]
#
# Arguments:
#   --model=MODEL_NAME        Name of the model from bnRep to convert
#                              (if omitted, converts all supported models)
#   --output_folder=DIR       Directory for output file (default: current directory)
#
# Examples:
#   # Convert a discrete model (outputs lawschool.bif in current directory)
#   Rscript convert_bnrep_models.R --model=lawschool
#
#   # Convert a Gaussian model to a specific directory
#   Rscript convert_bnrep_models.R --model=ecoli70 --output_folder=output/
#
#   # Convert all supported models
#   Rscript convert_bnrep_models.R --output_folder=output/
#
# Notes:
#   - Discrete models are saved as {model_name}.bif
#   - Gaussian models are saved as {model_name}.json
################################################################################

# Auto-install and load required packages
required_packages <- c("bnlearn", "jsonlite", "jsonvalidate", "bnRep")
missing_packages <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]
if (length(missing_packages) > 0) {
  cat(sprintf("Installing missing packages: %s\n", paste(missing_packages, collapse = ", ")))
  install.packages(missing_packages, repos = "https://cloud.r-project.org", quiet = TRUE)
}
suppressPackageStartupMessages({
  library(bnlearn)
  library(jsonlite)
  library(jsonvalidate)
  library(bnRep)
})

# Resolve schema path relative to this script's location
script_dir <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)))
schema_path <- file.path(script_dir, "..", "schema", "lgbn_schema.json")

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
model_name <- NULL
output_folder <- "."
for (arg in args) {
  if (grepl("^--model=", arg)) {
    model_name <- sub("^--model=", "", arg)
  } else if (grepl("^--output_folder=", arg)) {
    output_folder <- sub("^--output_folder=", "", arg)
  }
}

################################################################################
# Function: convert_gaussian_to_json
# Purpose: Convert a Gaussian bn.fit object to pgmpy JSON format
# Format: Matches the schema expected by pgmpy's JSON reader
#         Reference: https://github.com/pgmpy/example_models/blob/main/continuous/ecoli70.json
################################################################################
convert_gaussian_to_json <- function(bn_fit, output_file) {
  node_names <- names(bn_fit)
  arcs_matrix <- bnlearn::arcs(bn_fit)

  model_data <- list(
    nodes = node_names,
    arcs = list(),
    cpds = list()
  )

  for (i in seq_len(nrow(arcs_matrix))) {
    model_data$arcs[[i]] <- c(arcs_matrix[i, "from"], arcs_matrix[i, "to"])
  }

  for (node in node_names) {
    node_cpd <- bn_fit[[node]]
    parents <- node_cpd$parents
    coefs <- node_cpd$coefficients

    cpd_info <- list(
      coefficients = list("(Intercept)" = coefs[1]),
      variance = node_cpd$sd^2,
      parents = parents
    )

    for (parent in parents) {
      cpd_info$coefficients[[parent]] <- coefs[parent]
    }

    model_data$cpds[[node]] <- cpd_info
  }

  json_str <- jsonlite::toJSON(model_data, pretty = TRUE, auto_unbox = FALSE, digits = NA)

  # Validate against LGBN schema
  result <- jsonvalidate::json_validate(json_str, schema_path, verbose = TRUE, engine = "ajv")
  if (!result) {
    cat("Schema validation errors:\n")
    print(attr(result, "errors"))
    stop("Generated JSON does not conform to the LGBN schema.")
  }

  writeLines(json_str, output_file)
  cat(sprintf("Successfully converted to JSON: %s\n", output_file))
}

################################################################################
# Function: get_actual_model_type
# Purpose: Determine the actual model type from the bn.fit object class,
#          since bnRep summary may mislabel some models (e.g. CLG as Gaussian)
################################################################################
get_actual_model_type <- function(bn_fit) {
  if (inherits(bn_fit, "bn.fit.dnet")) {
    return("Discrete")
  } else if (inherits(bn_fit, "bn.fit.gnet")) {
    return("Gaussian")
  } else if (inherits(bn_fit, "bn.fit.cgnet")) {
    return("CLG")
  } else {
    return("Unknown")
  }
}

################################################################################
# Function: convert_model
# Purpose: Load and convert a single bnRep model
################################################################################
convert_model <- function(model_name, output_folder) {
  cat(sprintf("\nConverting model '%s'\n", model_name))

  data(list = model_name, package = "bnRep", envir = environment())
  bn_fit <- get(model_name)

  model_type <- get_actual_model_type(bn_fit)
  cat(sprintf("  Model type: %s\n", model_type))

  if (model_type == "Discrete") {
    output_file <- file.path(output_folder, paste0(model_name, ".bif"))
    bnlearn::write.bif(bn_fit, file = output_file)
    cat(sprintf("  Successfully converted to BIF: %s\n", output_file))
  } else if (model_type == "Gaussian") {
    output_file <- file.path(output_folder, paste0(model_name, ".json"))
    convert_gaussian_to_json(bn_fit, output_file)
  } else {
    cat(sprintf("  Skipping: model type '%s' is not supported.\n", model_type))
  }
}

################################################################################
# Main execution
################################################################################

data(bnRep_summary, package = "bnRep")

if (is.null(model_name)) {
  model_names <- bnRep_summary$Name
  cat(sprintf("Converting all %d models from bnRep\n", length(model_names)))
} else {
  model_names <- model_name
}

cat(paste(rep("=", 70), collapse = ""), "\n")

for (name in model_names) {
  convert_model(name, output_folder)
}

cat(paste(rep("=", 70), collapse = ""), "\n")
cat("Done!\n\n")
