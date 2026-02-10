# bnRep Example Models — Design Notes

## Background

The bnRep repository (https://github.com/manueleleonelli/bnRep) provides a collection of Bayesian
network models from the literature. These models are useful for benchmarking, education, and
research, but are not directly loadable in pgmpy in their current form.

This document records the agreed documentation and workflow for supporting bnRep example models
in pgmpy. It is intentionally limited to documentation and clarification of existing behavior,
rather than implementation or architectural changes.

## Comparison with existing example models

pgmpy currently supports loading example models from the bnlearn repository via
`pgmpy.utils.get_example_model`. bnRep differs in that:

- It contains models sourced from published literature
- It includes both discrete and continuous models
- Models are not distributed in formats directly consumable by pgmpy

## Documented workflow

1. **Input**
   - Raw model files from the bnRep repository

2. **Conversion**
   - A dedicated conversion script (implemented in R) reads bnRep models
   - Model type is detected:
   - Discrete models → exported to BIF format
   - Continuous models → exported to JSON format

3. **Output location**
   - Converted model files are placed in the `pgmpy/example_models` repository
   - Each bnRep model corresponds to one converted model file

4. **Model loading**
   - Existing pgmpy model loaders already support BIF and JSON formats
   - No changes to loader logic or public APIs are required

## Notes for contributors

This document is intentionally limited to documentation and workflow clarification.
Implementation of the conversion script and inclusion of converted model files
are expected to be handled in follow-up work.
