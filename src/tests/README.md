# KKT_MPNN Test Suite

This directory contains unit tests for the KKT_MPNN project.

## Quick Start

```bash
# Run all tests
cd /home/joachim-verschelde/Repos/KKT_MPNN
python -m pytest src/tests/ -v

# Run specific test file
pytest src/tests/test_jepa_phase1.py -v

# Run with detailed output
pytest src/tests/test_jepa_phase1.py -vv --tb=long
```

## Test Files

### test_jepa_phase1.py
**Status:** ✅ 27/27 tests passing

Complete validation suite for JEPA Phase 1 implementation, covering:
- JEPA utility functions (EMA, cosine loss, masking strategies)
- KKTNetMLP JEPA extensions
- GNNPolicy JEPA extensions
- End-to-end integration tests

**See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for detailed results.**

## Test Organization

Tests are organized into four main classes:

1. **TestJEPAUtilities** (13 tests)
   - EMA update functionality
   - Cosine similarity loss
   - LP-aware masking
   - GNN node masking
   - JEPA loss computation

2. **TestKKTNetMLPExtensions** (5 tests)
   - Trunk encoding
   - JEPA embedding normalization
   - JEPA prediction normalization
   - Backward compatibility

3. **TestGNNPolicyExtensions** (5 tests)
   - Node embedding normalization
   - Shape verification
   - Backward compatibility

4. **TestIntegration** (4 tests)
   - Model instantiation
   - End-to-end training loops
   - Gradient flow verification

## Requirements

Tests require the standard project environment:
- Python 3.9
- PyTorch 2.0
- PyTorch Geometric
- pytest

Install with:
```bash
conda env create -f environment.yml
conda activate graph-aug
```

## Writing New Tests

Follow the existing test structure:
1. Use descriptive test names (e.g., `test_feature_does_expected_thing`)
2. Include docstrings explaining what is tested
3. Use clear assertions with helpful error messages
4. Group related tests in classes
5. Use pytest fixtures for shared setup

Example:
```python
class TestNewFeature:
    """Test new feature implementation"""

    def test_feature_shape(self):
        """Test that feature returns correct shape"""
        result = new_feature(input_data)
        assert result.shape == expected_shape, \
            f"Expected {expected_shape}, got {result.shape}"
```

## CI/CD Integration

Tests are designed to be CI/CD friendly:
- Fast execution (< 5 seconds)
- No external dependencies
- Deterministic results
- Clear pass/fail indicators

## Test Coverage Goals

Target coverage by component:
- Core utilities: 100%
- Model extensions: 100%
- Integration: Key workflows
- Edge cases: Critical safety checks

## Troubleshooting

### Tests fail with import errors
Make sure you're in the project root and the conda environment is activated:
```bash
cd /home/joachim-verschelde/Repos/KKT_MPNN
conda activate graph-aug
```

### CUDA/GPU errors
Tests run on CPU by default. If you encounter GPU-specific issues, set:
```bash
export CUDA_VISIBLE_DEVICES=""
pytest src/tests/ -v
```

### Warnings about num_nodes
PyTorch Geometric warnings are informational only and don't affect test results.

## Contact

For questions about tests, see the validation report or project documentation.
