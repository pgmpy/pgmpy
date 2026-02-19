## Summary
This PR converts `test_global_vars.py` from unittest to pytest format, following the pattern established in recent commits.

## Changes Made
- Removed `import unittest` (no longer needed)
- Converted `TestDuplicateFilter` from `unittest.TestCase` to plain pytest class
- Replaced `self.assertEqual()` with `assert ... == ...` assertion
- Created `@pytest.fixture` named `reset_config` for resetting configuration after tests
- Added the fixture to `test_duplicate_filter` test method

## Testing
All tests pass successfully:
- 2 passed, 3 skipped (skipped tests require torch/torch.cuda)

## Related
This follows the conversion pattern from:
- `test_NaiveBayes.py`
- `test_BayesianEstimator.py`
- `test_NoisyOR.py`
- `test_BaseEstimator.py`

## Checklist
- [x] Tests pass with pytest
- [x] Code follows project style guidelines
- [x] No breaking changes
