"""
Unit tests for Phase 3: Optional Feature Normalization.

Tests cover:
1. Settings dataclass with normalize_features field
2. Feature normalization enabled (default behavior)
3. Feature normalization disabled
4. get_bipartite_graph function with normalize_features parameter

Run with: pytest src/tests/test_normalization_feature.py -v
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np
from data.common import Settings
from data.generators import get_bipartite_graph


class TestSettingsDataclass:
    """Test Settings dataclass normalize_features field"""

    def test_settings_has_normalize_features_field(self):
        """Test that Settings has normalize_features field"""
        settings = Settings(
            problems=('IS',),
            is_sizes=(10,),
            ca_sizes=(5,),
            sc_sizes=(5,),
            cfl_sizes=(5,),
            rnd_sizes=(5,),
        )

        assert hasattr(settings, 'normalize_features'), \
            "Settings should have normalize_features field"

    def test_settings_normalize_features_default_true(self):
        """Test that normalize_features defaults to True"""
        settings = Settings(
            problems=('IS',),
            is_sizes=(10,),
            ca_sizes=(5,),
            sc_sizes=(5,),
            cfl_sizes=(5,),
            rnd_sizes=(5,),
        )

        assert settings.normalize_features is True, \
            "normalize_features should default to True"

    def test_settings_normalize_features_can_be_false(self):
        """Test that normalize_features can be set to False"""
        settings = Settings(
            problems=('IS',),
            is_sizes=(10,),
            ca_sizes=(5,),
            sc_sizes=(5,),
            cfl_sizes=(5,),
            rnd_sizes=(5,),
            normalize_features=False
        )

        assert settings.normalize_features is False, \
            "normalize_features should be False when explicitly set"

    def test_settings_normalize_features_can_be_true(self):
        """Test that normalize_features can be explicitly set to True"""
        settings = Settings(
            problems=('IS',),
            is_sizes=(10,),
            ca_sizes=(5,),
            sc_sizes=(5,),
            cfl_sizes=(5,),
            rnd_sizes=(5,),
            normalize_features=True
        )

        assert settings.normalize_features is True, \
            "normalize_features should be True when explicitly set"


class TestFeatureNormalizationEnabled:
    """Test feature normalization when enabled (default behavior)"""

    def test_constraint_features_normalized(self):
        """Test that constraint features are normalized when normalize_features=True"""
        # Create simple LP problem with known values
        # A = [[2, 3], [4, 5]]
        # b = [10, 20]
        # c = [1, 2]

        from ecole import scip
        import pyscipopt as scip_py

        # Create a simple SCIP model
        model = scip_py.Model()
        model.hideOutput()

        # Add variables
        x1 = model.addVar(name="x1", vtype="C", lb=0, ub=None)
        x2 = model.addVar(name="x2", vtype="C", lb=0, ub=None)

        # Add constraints with known coefficients
        model.addCons(2*x1 + 3*x2 <= 10, name="c1")
        model.addCons(4*x1 + 5*x2 <= 20, name="c2")

        # Set objective
        model.setObjective(x1 + 2*x2, "minimize")

        # Get bipartite graph with normalization enabled (default)
        graph_data = get_bipartite_graph(model, normalize_features=True)

        # Extract constraint features
        constraint_features = graph_data['constraint_features']

        # Check that features are in normalized range [0, 1]
        # Allow small tolerance for numerical precision
        assert torch.all(constraint_features >= -0.01), \
            f"Normalized constraint features should be >= 0, got min: {constraint_features.min()}"
        assert torch.all(constraint_features <= 1.01), \
            f"Normalized constraint features should be <= 1, got max: {constraint_features.max()}"

    def test_variable_features_normalized(self):
        """Test that variable features are normalized when normalize_features=True"""
        from ecole import scip
        import pyscipopt as scip_py

        # Create a simple SCIP model
        model = scip_py.Model()
        model.hideOutput()

        # Add variables with various bounds and costs
        x1 = model.addVar(name="x1", vtype="C", lb=0, ub=100, obj=10)
        x2 = model.addVar(name="x2", vtype="C", lb=0, ub=200, obj=20)

        # Add a constraint
        model.addCons(x1 + x2 <= 150, name="c1")

        # Get bipartite graph with normalization enabled
        graph_data = get_bipartite_graph(model, normalize_features=True)

        # Extract variable features
        variable_features = graph_data['variable_features']

        # Check that features are in normalized range [0, 1]
        assert torch.all(variable_features >= -0.01), \
            f"Normalized variable features should be >= 0, got min: {variable_features.min()}"
        assert torch.all(variable_features <= 1.01), \
            f"Normalized variable features should be <= 1, got max: {variable_features.max()}"

    def test_edge_features_normalized(self):
        """Test that edge features are normalized when normalize_features=True"""
        from ecole import scip
        import pyscipopt as scip_py

        # Create a simple SCIP model
        model = scip_py.Model()
        model.hideOutput()

        # Add variables
        x1 = model.addVar(name="x1", vtype="C", lb=0, ub=None)
        x2 = model.addVar(name="x2", vtype="C", lb=0, ub=None)

        # Add constraints with various coefficients
        model.addCons(10*x1 + 20*x2 <= 100, name="c1")
        model.addCons(5*x1 + 15*x2 <= 75, name="c2")

        # Set objective
        model.setObjective(x1 + x2, "minimize")

        # Get bipartite graph with normalization enabled
        graph_data = get_bipartite_graph(model, normalize_features=True)

        # Extract edge features
        edge_features = graph_data['edge_features']

        # Check that features are in normalized range [0, 1]
        assert torch.all(edge_features >= -0.01), \
            f"Normalized edge features should be >= 0, got min: {edge_features.min()}"
        assert torch.all(edge_features <= 1.01), \
            f"Normalized edge features should be <= 1, got max: {edge_features.max()}"


class TestFeatureNormalizationDisabled:
    """Test feature normalization when disabled"""

    def test_constraint_features_not_normalized(self):
        """Test that constraint features are NOT normalized when normalize_features=False"""
        from ecole import scip
        import pyscipopt as scip_py

        # Create a simple SCIP model with known large values
        model = scip_py.Model()
        model.hideOutput()

        # Add variables
        x1 = model.addVar(name="x1", vtype="C", lb=0, ub=None)
        x2 = model.addVar(name="x2", vtype="C", lb=0, ub=None)

        # Add constraints with large RHS values (outside [0,1])
        model.addCons(x1 + x2 <= 1000, name="c1")  # Large RHS
        model.addCons(x1 + x2 <= 2000, name="c2")  # Even larger RHS

        # Set objective
        model.setObjective(x1 + x2, "minimize")

        # Get bipartite graph with normalization DISABLED
        graph_data = get_bipartite_graph(model, normalize_features=False)

        # Extract constraint features
        constraint_features = graph_data['constraint_features']

        # With normalization disabled, at least some features should be > 1
        # (indicating raw values are preserved)
        # The RHS values (1000, 2000) should appear in the features
        has_large_values = torch.any(constraint_features.abs() > 10)

        assert has_large_values, \
            f"Unnormalized features should contain large values, got max: {constraint_features.abs().max()}"

    def test_variable_features_not_normalized(self):
        """Test that variable features are NOT normalized when normalize_features=False"""
        from ecole import scip
        import pyscipopt as scip_py

        # Create a simple SCIP model with large objective coefficients
        model = scip_py.Model()
        model.hideOutput()

        # Add variables with large objective coefficients
        x1 = model.addVar(name="x1", vtype="C", lb=0, ub=1000, obj=500)  # Large values
        x2 = model.addVar(name="x2", vtype="C", lb=0, ub=2000, obj=1000)  # Even larger

        # Add a constraint
        model.addCons(x1 + x2 <= 1500, name="c1")

        # Get bipartite graph with normalization DISABLED
        graph_data = get_bipartite_graph(model, normalize_features=False)

        # Extract variable features
        variable_features = graph_data['variable_features']

        # With normalization disabled, features should contain large values
        has_large_values = torch.any(variable_features.abs() > 10)

        assert has_large_values, \
            f"Unnormalized variable features should contain large values, got max: {variable_features.abs().max()}"

    def test_edge_features_not_normalized(self):
        """Test that edge features are NOT normalized when normalize_features=False"""
        from ecole import scip
        import pyscipopt as scip_py

        # Create a simple SCIP model with large coefficients
        model = scip_py.Model()
        model.hideOutput()

        # Add variables
        x1 = model.addVar(name="x1", vtype="C", lb=0, ub=None)
        x2 = model.addVar(name="x2", vtype="C", lb=0, ub=None)

        # Add constraints with large coefficients
        model.addCons(100*x1 + 200*x2 <= 1000, name="c1")
        model.addCons(300*x1 + 400*x2 <= 2000, name="c2")

        # Set objective
        model.setObjective(x1 + x2, "minimize")

        # Get bipartite graph with normalization DISABLED
        graph_data = get_bipartite_graph(model, normalize_features=False)

        # Extract edge features
        edge_features = graph_data['edge_features']

        # With normalization disabled, edge features should contain large values
        # (the coefficients 100, 200, 300, 400)
        has_large_values = torch.any(edge_features.abs() > 10)

        assert has_large_values, \
            f"Unnormalized edge features should contain large values, got max: {edge_features.abs().max()}"


class TestNormalizationComparison:
    """Test that normalized and unnormalized features differ appropriately"""

    def test_normalized_vs_unnormalized_differ(self):
        """Test that normalized and unnormalized features are different"""
        from ecole import scip
        import pyscipopt as scip_py

        # Create a simple SCIP model
        model1 = scip_py.Model()
        model1.hideOutput()
        x1 = model1.addVar(name="x1", vtype="C", lb=0, ub=1000, obj=100)
        x2 = model1.addVar(name="x2", vtype="C", lb=0, ub=2000, obj=200)
        model1.addCons(10*x1 + 20*x2 <= 500, name="c1")
        model1.setObjective(x1 + x2, "minimize")

        # Clone model for second test
        model2 = scip_py.Model()
        model2.hideOutput()
        x1 = model2.addVar(name="x1", vtype="C", lb=0, ub=1000, obj=100)
        x2 = model2.addVar(name="x2", vtype="C", lb=0, ub=2000, obj=200)
        model2.addCons(10*x1 + 20*x2 <= 500, name="c1")
        model2.setObjective(x1 + x2, "minimize")

        # Get with normalization enabled
        graph_norm = get_bipartite_graph(model1, normalize_features=True)

        # Get with normalization disabled
        graph_unnorm = get_bipartite_graph(model2, normalize_features=False)

        # Features should be different (normalized vs raw)
        # Note: They might have same shape, but values should differ
        var_feat_norm = graph_norm['variable_features']
        var_feat_unnorm = graph_unnorm['variable_features']

        # At least some features should differ significantly
        # (normalized should be in [0,1], unnormalized should have larger values)
        max_norm = var_feat_norm.abs().max()
        max_unnorm = var_feat_unnorm.abs().max()

        assert max_unnorm > max_norm * 2, \
            f"Unnormalized max ({max_unnorm}) should be significantly larger than normalized max ({max_norm})"

    def test_graph_structure_same_regardless_of_normalization(self):
        """Test that graph structure (edges, sizes) is unchanged by normalization setting"""
        from ecole import scip
        import pyscipopt as scip_py

        # Create a simple SCIP model
        model1 = scip_py.Model()
        model1.hideOutput()
        x1 = model1.addVar(name="x1", vtype="C", lb=0, ub=None)
        x2 = model1.addVar(name="x2", vtype="C", lb=0, ub=None)
        model1.addCons(x1 + x2 <= 100, name="c1")
        model1.addCons(x1 + x2 <= 200, name="c2")
        model1.setObjective(x1 + x2, "minimize")

        # Clone model
        model2 = scip_py.Model()
        model2.hideOutput()
        x1 = model2.addVar(name="x1", vtype="C", lb=0, ub=None)
        x2 = model2.addVar(name="x2", vtype="C", lb=0, ub=None)
        model2.addCons(x1 + x2 <= 100, name="c1")
        model2.addCons(x1 + x2 <= 200, name="c2")
        model2.setObjective(x1 + x2, "minimize")

        # Get graphs with different normalization settings
        graph_norm = get_bipartite_graph(model1, normalize_features=True)
        graph_unnorm = get_bipartite_graph(model2, normalize_features=False)

        # Check that graph structure is identical
        assert graph_norm['edge_index'].shape == graph_unnorm['edge_index'].shape, \
            "Edge index shape should be same regardless of normalization"
        assert torch.equal(graph_norm['edge_index'], graph_unnorm['edge_index']), \
            "Edge indices should be identical regardless of normalization"

        # Check feature dimensions match (even if values differ)
        assert graph_norm['constraint_features'].shape == graph_unnorm['constraint_features'].shape, \
            "Constraint feature shapes should match"
        assert graph_norm['variable_features'].shape == graph_unnorm['variable_features'].shape, \
            "Variable feature shapes should match"
        assert graph_norm['edge_features'].shape == graph_unnorm['edge_features'].shape, \
            "Edge feature shapes should match"


class TestBackwardCompatibility:
    """Test backward compatibility of normalization feature"""

    def test_get_bipartite_graph_defaults_to_true(self):
        """Test that get_bipartite_graph defaults to normalize_features=True when not specified"""
        from ecole import scip
        import pyscipopt as scip_py

        # Create a simple SCIP model
        model = scip_py.Model()
        model.hideOutput()
        x1 = model.addVar(name="x1", vtype="C", lb=0, ub=1000, obj=100)
        x2 = model.addVar(name="x2", vtype="C", lb=0, ub=2000, obj=200)
        model.addCons(10*x1 + 20*x2 <= 500, name="c1")
        model.setObjective(x1 + x2, "minimize")

        # Call without normalize_features parameter (should default to True)
        graph_default = get_bipartite_graph(model)

        # Call with explicit normalize_features=True
        model2 = scip_py.Model()
        model2.hideOutput()
        x1 = model2.addVar(name="x1", vtype="C", lb=0, ub=1000, obj=100)
        x2 = model2.addVar(name="x2", vtype="C", lb=0, ub=2000, obj=200)
        model2.addCons(10*x1 + 20*x2 <= 500, name="c1")
        model2.setObjective(x1 + x2, "minimize")
        graph_explicit = get_bipartite_graph(model2, normalize_features=True)

        # Results should be identical (default behavior is normalization)
        # Features should be in [0, 1] range for both
        assert torch.all(graph_default['variable_features'] <= 1.01), \
            "Default behavior should normalize features"
        assert torch.all(graph_explicit['variable_features'] <= 1.01), \
            "Explicit True should normalize features"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
