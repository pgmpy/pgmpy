#!/usr/bin/env Rscript

################################################################################
# Script: convert_bnrep_models.R
# Purpose: Convert a single Bayesian network model from the bnRep repository to
#          BIF format (discrete) or JSON format (continuous/Gaussian)
# Author: pgmpy development team
# License: MIT
#
# Requirements:
#   - R packages: bnRep, bnlearn, jsonlite
#   - Install with: install.packages(c("bnRep", "bnlearn", "jsonlite"))
#
# Usage:
#   Rscript convert_bnrep_models.R --model=MODEL_NAME --output=OUTPUT_FILE
#
# Arguments:
#   --model=MODEL_NAME    Name of the model from bnRep to convert (required)
#   --output=OUTPUT_FILE  Path to output file (required)
#   --help                Show this help message
#
# Examples:
#   # Convert a discrete model to BIF format
#   Rscript convert_bnrep_models.R --model=lawschool --output=lawschool.bif
#
#   # Convert a Gaussian model to JSON format
#   Rscript convert_bnrep_models.R --model=ecoli70 --output=ecoli70.json
#
# Notes:
#   - Discrete models must be saved with .bif extension
#   - Gaussian models must be saved with .json extension
#   - The script will validate the output file extension matches the model type
################################################################################

# Load required libraries
suppressPackageStartupMessages({
  if (!require("bnlearn", quietly = TRUE)) {
    stop("Package 'bnlearn' is required. Install with: install.packages('bnlearn')")
  }
  if (!require("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required. Install with: install.packages('jsonlite')")
  }
  if (!require("bnRep", quietly = TRUE)) {
    stop("Package 'bnRep' is required. Install with: install.packages('bnRep')")
  }
})

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Default values
model_name <- NULL
output_file <- NULL
show_help <- FALSE

# Parse arguments
for (arg in args) {
  if (grepl("^--model=", arg)) {
    model_name <- sub("^--model=", "", arg)
  } else if (grepl("^--output=", arg)) {
    output_file <- sub("^--output=", "", arg)
  } else if (arg == "--help") {
    show_help <- TRUE
  } else {
    warning(paste("Unknown argument:", arg))
  }
}

# Show help message
if (show_help) {
  cat("
convert_bnrep_models.R - Convert bnRep models to BIF/JSON format

Usage:
  Rscript convert_bnrep_models.R --model=MODEL_NAME --output=OUTPUT_FILE

Arguments:
  --model=MODEL_NAME    Name of the model from bnRep to convert (required)
  --output=OUTPUT_FILE  Path to output file (required)
  --help                Show this help message

Examples:
  # Convert a discrete model to BIF format
  Rscript convert_bnrep_models.R --model=lawschool --output=lawschool.bif

  # Convert a Gaussian model to JSON format
  Rscript convert_bnrep_models.R --model=ecoli70 --output=ecoli70.json

Notes:
  - Discrete models must be saved with .bif extension
  - Gaussian models must be saved with .json extension
  - The script validates the output file extension matches the model type

For more information, see: https://github.com/manueleleonelli/bnRep
")
  quit(save = "no", status = 0)
}

# Validate required arguments
if (is.null(model_name)) {
  cat("Error: --model argument is required\n")
  cat("Use --help for usage information\n")
  quit(save = "no", status = 1)
}

if (is.null(output_file)) {
  cat("Error: --output argument is required\n")
  cat("Use --help for usage information\n")
  quit(save = "no", status = 1)
}

################################################################################
# Function: convert_discrete_to_bif
# Purpose: Convert a discrete bn.fit object to BIF format
################################################################################
convert_discrete_to_bif <- function(bn_fit, output_file) {
  tryCatch({
    # Validate file extension
    if (!grepl("\\.bif$", output_file, ignore.case = TRUE)) {
      stop("Output file for discrete models must have .bif extension")
    }

    bnlearn::write.bif(bn_fit, file = output_file)
    cat(sprintf("✓ Successfully converted to BIF: %s\n", output_file))
    return(TRUE)
  }, error = function(e) {
    cat(sprintf("✗ Error converting to BIF: %s\n", e$message))
    return(FALSE)
  })
}

################################################################################
# Function: convert_gaussian_to_json
# Purpose: Convert a Gaussian bn.fit object to pgmpy JSON format
# Format: Matches the schema expected by pgmpy's JSON reader
#         Reference: https://github.com/pgmpy/example_models/blob/main/continuous/ecoli70.json
################################################################################
convert_gaussian_to_json <- function(bn_fit, output_file) {
  tryCatch({
    # Validate file extension
    if (!grepl("\\.json$", output_file, ignore.case = TRUE)) {
      stop("Output file for Gaussian models must have .json extension")
    }

    # Extract network structure
    bn_net <- bnlearn::bn.net(bn_fit)

    # Build JSON structure matching pgmpy's expected format
    model_data <- list(
      nodes = list(),
      arcs = list(),
      cpds = list()
    )

    # Extract nodes
    node_names <- names(bn_fit)
    model_data$nodes <- node_names

    # Extract arcs (edges)
    arcs_matrix <- bnlearn::arcs(bn_net)
    if (nrow(arcs_matrix) > 0) {
      for (i in 1:nrow(arcs_matrix)) {
        model_data$arcs[[i]] <- c(arcs_matrix[i, "from"], arcs_matrix[i, "to"])
      }
    }

    # Extract CPD parameters for each node
    for (node in node_names) {
      node_cpd <- bn_fit[[node]]
      parents <- node_cpd$parents

      # Initialize CPD structure
      cpd_info <- list(
        coefficients = list(),
        variance = numeric(0),
        parents = if (length(parents) > 0) parents else list()
      )

      # Extract coefficients (try both 'coefficients' and 'coef' field names)
      coef_field <- NULL
      if ("coefficients" %in% names(node_cpd)) {
        coef_field <- "coefficients"
      } else if ("coef" %in% names(node_cpd)) {
        coef_field <- "coef"
      }

      if (!is.null(coef_field)) {
        # Get intercept
        cpd_info$coefficients[["(Intercept)"]] <- node_cpd[[coef_field]][1]

        # Get parent coefficients
        if (length(parents) > 0) {
          for (parent in parents) {
            parent_coef <- node_cpd[[coef_field]][parent]
            cpd_info$coefficients[[parent]] <- parent_coef
          }
        }

        # Get variance (stored as standard deviation in bnlearn, need variance)
        cpd_info$variance <- node_cpd$sd^2
      } else {
        stop(sprintf("Node %s does not have coefficient information", node))
      }

      model_data$cpds[[node]] <- cpd_info
    }

    # Write to JSON file with proper formatting
    # Use auto_unbox=FALSE to ensure single values are in arrays (matching pgmpy format)
    json_str <- jsonlite::toJSON(model_data, pretty = TRUE, auto_unbox = FALSE)
    writeLines(json_str, output_file)

    cat(sprintf("✓ Successfully converted to JSON: %s\n", output_file))
    return(TRUE)
  }, error = function(e) {
    cat(sprintf("✗ Error converting to JSON: %s\n", e$message))
    return(FALSE)
  })
}

################################################################################
# Main execution
################################################################################

cat(sprintf("Converting model '%s' from bnRep\n", model_name))
cat(paste(rep("=", 70), collapse = ""), "\n")

# Load bnRep summary to get model information
data(bnRep_summary, package = "bnRep")

# Find model info in summary
model_info <- bnRep_summary[bnRep_summary$Name == model_name, ]

if (nrow(model_info) == 0) {
  cat(sprintf("✗ Error: Model '%s' not found in bnRep repository\n", model_name))
  cat("\nTo see available models, use the bnRep R package:\n")
  cat("  R -e 'data(bnRep_summary, package=\"bnRep\"); print(bnRep_summary$Name)'\n\n")
  quit(save = "no", status = 1)
}

model_type <- model_info$Type[1]
cat(sprintf("Model type: %s\n", model_type))
cat(sprintf("Nodes: %d\n", model_info$Nodes[1]))
cat(sprintf("Arcs: %d\n", model_info$Arcs[1]))
cat(paste(rep("-", 70), collapse = ""), "\n")

# Load the model from bnRep
cat("Loading model from bnRep...\n")
tryCatch({
  data(list = model_name, package = "bnRep", envir = environment())
  bn_fit <- get(model_name)
  cat("✓ Model loaded successfully\n")
}, error = function(e) {
  cat(sprintf("✗ Error loading model: %s\n", e$message))
  quit(save = "no", status = 1)
})

# Convert based on model type
cat(sprintf("Converting to output file: %s\n", output_file))

if (model_type == "Discrete") {
  success <- convert_discrete_to_bif(bn_fit, output_file)
} else if (model_type == "Gaussian") {
  success <- convert_gaussian_to_json(bn_fit, output_file)
} else if (model_type == "CLG") {
  cat("✗ Error: CLG (Conditional Linear Gaussian) models are not yet supported\n")
  cat("Only Discrete and Gaussian models are currently supported.\n")
  quit(save = "no", status = 1)
} else {
  cat(sprintf("✗ Error: Unknown model type '%s'\n", model_type))
  quit(save = "no", status = 1)
}

cat(paste(rep("=", 70), collapse = ""), "\n")

if (success) {
  cat("✓ Conversion completed successfully!\n\n")
  quit(save = "no", status = 0)
} else {
  cat("✗ Conversion failed!\n\n")
  quit(save = "no", status = 1)
}
