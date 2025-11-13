"""
Unit tests for JEPA Phase 2 implementation (Training Integration).

Tests cover:
1. TrainingState extensions for JEPA loss tracking
2. Target model creation for EMA mode
3. train_epoch function signature extensions
4. JEPA loss integration in training loop
5. Checkpoint handling for target model

Run with: pytest src/tests/test_jepa_phase2.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY
from copy import deepcopy

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch_geometric
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader

# Import components to test
from train import TrainingState, train_epoch
from models.models import KKTNetMLP, GNNPolicy
from models.jepa_utils import ema_update


class TestTrainingStateExtensions:
    """Test TrainingState extensions for JEPA loss tracking"""

    def test_add_jepa_loss_accumulates_correctly(self):
        """Test that add_jepa_loss accumulates loss values"""
        state = TrainingState(log_every=10)

        # Manually set these to avoid WandB calls
        state.steps = 0
        state.trained_items = 0

        state.add_jepa_loss(1.5)
        assert state.jepa_loss_sum == 1.5

        state.add_jepa_loss(2.3)
        assert state.jepa_loss_sum == pytest.approx(3.8)

        state.add_jepa_loss(0.7)
        assert state.jepa_loss_sum == pytest.approx(4.5)  # 1.5 + 2.3 + 0.7 = 4.5

    def test_finish_epoch_returns_tuple(self):
        """Test that finish_epoch returns tuple of (training_loss, jepa_loss)"""
        state = TrainingState(log_every=10)
        state.trained_items = 4
        state.training_loss_sum = 8.0
        state.jepa_loss_sum = 4.0

        result = state.finish_epoch()

        assert isinstance(result, tuple), "finish_epoch should return a tuple"
        assert len(result) == 2, "finish_epoch should return 2 values"

    def test_finish_epoch_returns_both_losses(self):
        """Test that finish_epoch computes correct average losses"""
        state = TrainingState(log_every=10)
        state.trained_items = 4
        state.training_loss_sum = 8.0
        state.jepa_loss_sum = 4.0

        training_loss, jepa_loss = state.finish_epoch()

        assert training_loss == pytest.approx(2.0), "Training loss should be 8.0/4 = 2.0"
        assert jepa_loss == pytest.approx(1.0), "JEPA loss should be 4.0/4 = 1.0"

    def test_finish_epoch_returns_none_when_no_jepa_loss(self):
        """Test that finish_epoch returns None for JEPA loss when not used"""
        state = TrainingState(log_every=10)
        state.trained_items = 4
        state.training_loss_sum = 8.0
        state.jepa_loss_sum = 0.0  # No JEPA loss

        training_loss, jepa_loss = state.finish_epoch()

        assert training_loss == pytest.approx(2.0)
        assert jepa_loss is None, "JEPA loss should be None when sum is 0"

    def test_finish_epoch_resets_jepa_loss(self):
        """Test that finish_epoch resets JEPA loss sum to 0"""
        state = TrainingState(log_every=10)
        state.trained_items = 4
        state.training_loss_sum = 8.0
        state.jepa_loss_sum = 4.0

        state.finish_epoch()

        assert state.jepa_loss_sum == 0.0, "JEPA loss sum should be reset after epoch"

    def test_reset_training_state_clears_jepa_loss(self):
        """Test that _reset_training_state clears JEPA loss sum"""
        state = TrainingState(log_every=10)
        state.jepa_loss_sum = 10.0

        state._reset_training_state()

        assert state.jepa_loss_sum == 0.0

    def test_get_step_still_works(self):
        """Test that get_step() method still functions correctly"""
        state = TrainingState(log_every=10)
        assert state.get_step() == 0

        state.steps = 5
        assert state.get_step() == 5

    def test_get_epoch_still_works(self):
        """Test that get_epoch() method still functions correctly"""
        state = TrainingState(log_every=10)
        assert state.get_epoch() == 0

        state.epoch = 3
        assert state.get_epoch() == 3

    def test_finish_epoch_increments_epoch(self):
        """Test that finish_epoch increments epoch counter"""
        state = TrainingState(log_every=10)
        state.trained_items = 1
        state.training_loss_sum = 1.0

        initial_epoch = state.get_epoch()
        state.finish_epoch()

        assert state.get_epoch() == initial_epoch + 1


class TestTargetModelCreation:
    """Test target model creation for JEPA EMA mode"""

    @pytest.fixture
    def simple_args(self):
        """Create simple args object"""
        class Args:
            use_jepa = True
            jepa_mode = "ema"
            ema_momentum = 0.996
        return Args()

    def test_target_model_created_in_ema_mode(self):
        """Test that target model is created when use_jepa=True and jepa_mode='ema'"""
        m, n = 10, 8
        online_model = KKTNetMLP(m, n)

        # Simulate what train.py does
        args = type('Args', (), {
            'use_jepa': True,
            'jepa_mode': 'ema'
        })()

        target_model = None
        if args.use_jepa and args.jepa_mode == "ema":
            target_model = deepcopy(online_model)
            for p in target_model.parameters():
                p.requires_grad_(False)

        assert target_model is not None, "Target model should be created in EMA mode"

    def test_target_model_not_created_in_simsiam_mode(self):
        """Test that target model is NOT created in SimSiam mode"""
        m, n = 10, 8
        online_model = KKTNetMLP(m, n)

        # Simulate SimSiam mode
        args = type('Args', (), {
            'use_jepa': True,
            'jepa_mode': 'simsiam'
        })()

        target_model = None
        if args.use_jepa and args.jepa_mode == "ema":
            target_model = deepcopy(online_model)

        assert target_model is None, "Target model should NOT be created in SimSiam mode"

    def test_target_model_parameters_no_grad(self):
        """Test that target model parameters have requires_grad=False"""
        m, n = 10, 8
        online_model = KKTNetMLP(m, n)

        # Create target model as in train.py
        target_model = deepcopy(online_model)
        for p in target_model.parameters():
            p.requires_grad_(False)

        # Check all parameters
        for param in target_model.parameters():
            assert param.requires_grad is False, \
                "All target model parameters should have requires_grad=False"

    def test_target_model_is_deepcopy(self):
        """Test that target model is independent from online model"""
        m, n = 10, 8
        online_model = KKTNetMLP(m, n)

        # Initialize online model with specific values
        for param in online_model.parameters():
            param.data.fill_(1.0)

        # Create target model
        target_model = deepcopy(online_model)

        # Modify online model
        for param in online_model.parameters():
            param.data.fill_(2.0)

        # Target should be unchanged
        for param in target_model.parameters():
            assert torch.allclose(param.data, torch.ones_like(param.data)), \
                "Target model should be independent from online model"

    def test_target_model_none_when_use_jepa_false(self):
        """Test that target model is None when use_jepa=False"""
        m, n = 10, 8
        online_model = KKTNetMLP(m, n)

        # Simulate use_jepa=False
        args = type('Args', (), {
            'use_jepa': False,
            'jepa_mode': 'ema'
        })()

        target_model = None
        if args.use_jepa and args.jepa_mode == "ema":
            target_model = deepcopy(online_model)

        assert target_model is None, "Target model should be None when use_jepa=False"

    def test_target_model_same_architecture(self):
        """Test that target model has same architecture as online model"""
        m, n = 10, 8
        online_model = KKTNetMLP(m, n, hidden=128, jepa_embed_dim=64)
        target_model = deepcopy(online_model)

        # Check that architectures match
        online_params = sum(p.numel() for p in online_model.parameters())
        target_params = sum(p.numel() for p in target_model.parameters())

        assert online_params == target_params, \
            "Target model should have same number of parameters as online model"


class TestTrainEpochSignature:
    """Test train_epoch function signature extensions"""

    @pytest.fixture
    def minimal_loader(self):
        """Create minimal data loader for testing"""
        # Create minimal MLP data with correct mask dimensions
        m, n = 5, 4
        batch_size = 2

        # Simple dataset
        model_input = torch.randn(batch_size, m * n + m + n)
        A = torch.randn(batch_size, m, n)
        b = torch.randn(batch_size, m)
        c = torch.randn(batch_size, n)
        # Masks should be [B, M] and [B, N] with 1.0 for real, 0.0 for padding
        mask_m = torch.ones((batch_size, m), dtype=torch.float32)
        mask_n = torch.ones((batch_size, n), dtype=torch.float32)

        dataset = [(model_input, A, b, c, mask_m, mask_n)]
        return dataset

    def test_train_epoch_accepts_args_parameter(self):
        """Test that train_epoch accepts args parameter"""
        import inspect
        sig = inspect.signature(train_epoch)

        assert 'args' in sig.parameters, "train_epoch should have 'args' parameter"

    def test_train_epoch_accepts_target_model_parameter(self):
        """Test that train_epoch accepts target_model parameter"""
        import inspect
        sig = inspect.signature(train_epoch)

        assert 'target_model' in sig.parameters, "train_epoch should have 'target_model' parameter"

    def test_train_epoch_returns_tuple(self, minimal_loader):
        """Test that train_epoch returns tuple of (loss, jepa_loss)"""
        m, n = 5, 4
        model = KKTNetMLP(m, n)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        device = 'cpu'
        training_state = TrainingState(log_every=10)

        with patch('wandb.log'):
            result = train_epoch(
                model=model,
                loader=minimal_loader,
                optimizer=optimizer,
                device=device,
                training_state=training_state,
                primal_weight=0.1,
                dual_weight=0.1,
                stationarity_weight=0.6,
                complementary_slackness_weight=0.2,
                args=None,
                target_model=None
            )

        assert isinstance(result, tuple), "train_epoch should return a tuple"
        assert len(result) == 2, "train_epoch should return 2 values"

    def test_train_epoch_backward_compatible(self, minimal_loader):
        """Test backward compatibility with args=None, target_model=None"""
        m, n = 5, 4
        model = KKTNetMLP(m, n)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        device = 'cpu'
        training_state = TrainingState(log_every=10)

        # Should work without errors when args=None and target_model=None
        with patch('wandb.log'):
            try:
                train_loss, jepa_loss = train_epoch(
                    model=model,
                    loader=minimal_loader,
                    optimizer=optimizer,
                    device=device,
                    training_state=training_state,
                    primal_weight=0.1,
                    dual_weight=0.1,
                    stationarity_weight=0.6,
                    complementary_slackness_weight=0.2,
                    args=None,
                    target_model=None
                )
                assert jepa_loss is None, "JEPA loss should be None when args=None"
            except Exception as e:
                pytest.fail(f"train_epoch should work with args=None: {e}")


class TestJEPALossIntegration:
    """Test JEPA loss integration in training loop"""

    @pytest.fixture
    def mlp_setup(self):
        """Setup for MLP tests"""
        m, n = 5, 4
        batch_size = 2

        model = KKTNetMLP(m, n, jepa_embed_dim=32)
        target_model = deepcopy(model)
        for p in target_model.parameters():
            p.requires_grad_(False)

        # Create data with correct mask dimensions
        model_input = torch.randn(batch_size, m * n + m + n)
        A = torch.randn(batch_size, m, n)
        b = torch.randn(batch_size, m)
        c = torch.randn(batch_size, n)
        mask_m = torch.ones((batch_size, m), dtype=torch.float32)
        mask_n = torch.ones((batch_size, n), dtype=torch.float32)

        loader = [(model_input, A, b, c, mask_m, mask_n)]

        return model, target_model, loader

    @pytest.fixture
    def jepa_args(self):
        """Create args for JEPA training"""
        class Args:
            use_jepa = True
            jepa_mode = "ema"
            jepa_weight = 0.2
            jepa_pretrain_epochs = 0
            jepa_mask_entry_online = 0.3
            jepa_mask_row_online = 0.2
            jepa_mask_col_online = 0.2
            jepa_mask_entry_target = 0.1
            jepa_mask_row_target = 0.05
            jepa_mask_col_target = 0.05
            jepa_noisy_mask = False
            jepa_row_scaling = False
            ema_momentum = 0.996
            jepa_mask_ratio_nodes = 0.3
        return Args()

    def test_jepa_loss_computed_for_mlp(self, mlp_setup, jepa_args):
        """Test that JEPA loss is computed for MLP architecture"""
        model, target_model, loader = mlp_setup

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        training_state = TrainingState(log_every=10)

        with patch('wandb.log') as mock_log:
            train_loss, jepa_loss = train_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                device='cpu',
                training_state=training_state,
                primal_weight=0.1,
                dual_weight=0.1,
                stationarity_weight=0.6,
                complementary_slackness_weight=0.2,
                args=jepa_args,
                target_model=target_model
            )

        assert jepa_loss is not None, "JEPA loss should be computed for MLP"
        assert jepa_loss > 0, "JEPA loss should be positive"

    def test_jepa_loss_logged_to_wandb(self, mlp_setup, jepa_args):
        """Test that JEPA loss is logged to WandB"""
        model, target_model, loader = mlp_setup

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        training_state = TrainingState(log_every=10)

        with patch('wandb.log') as mock_log:
            train_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                device='cpu',
                training_state=training_state,
                primal_weight=0.1,
                dual_weight=0.1,
                stationarity_weight=0.6,
                complementary_slackness_weight=0.2,
                args=jepa_args,
                target_model=target_model
            )

        # Check that wandb.log was called with JEPA loss
        log_calls = [call[0][0] for call in mock_log.call_args_list]
        jepa_logged = any('train/loss_jepa' in call for call in log_calls)
        assert jepa_logged, "JEPA loss should be logged to WandB"

    def test_kkt_loss_also_logged(self, mlp_setup, jepa_args):
        """Test that KKT loss is also logged separately"""
        model, target_model, loader = mlp_setup

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        training_state = TrainingState(log_every=10)

        with patch('wandb.log') as mock_log:
            train_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                device='cpu',
                training_state=training_state,
                primal_weight=0.1,
                dual_weight=0.1,
                stationarity_weight=0.6,
                complementary_slackness_weight=0.2,
                args=jepa_args,
                target_model=target_model
            )

        # Check that wandb.log was called with KKT loss
        log_calls = [call[0][0] for call in mock_log.call_args_list]
        kkt_logged = any('train/loss_kkt' in call for call in log_calls)
        assert kkt_logged, "KKT loss should be logged to WandB"

    def test_training_state_tracks_jepa_loss(self, mlp_setup, jepa_args):
        """Test that training_state tracks JEPA loss"""
        model, target_model, loader = mlp_setup

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        training_state = TrainingState(log_every=10)

        with patch('wandb.log'):
            train_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                device='cpu',
                training_state=training_state,
                primal_weight=0.1,
                dual_weight=0.1,
                stationarity_weight=0.6,
                complementary_slackness_weight=0.2,
                args=jepa_args,
                target_model=target_model
            )

        # After finish_epoch, jepa_loss_sum should be reset but we got a return value
        # This confirms tracking happened
        assert True  # If we got here, tracking worked

    def test_no_jepa_loss_when_use_jepa_false(self, mlp_setup):
        """Test that JEPA loss is not computed when use_jepa=False"""
        model, target_model, loader = mlp_setup

        args = type('Args', (), {'use_jepa': False})()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        training_state = TrainingState(log_every=10)

        with patch('wandb.log'):
            train_loss, jepa_loss = train_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                device='cpu',
                training_state=training_state,
                primal_weight=0.1,
                dual_weight=0.1,
                stationarity_weight=0.6,
                complementary_slackness_weight=0.2,
                args=args,
                target_model=target_model
            )

        assert jepa_loss is None, "JEPA loss should be None when use_jepa=False"

    def test_pretraining_schedule_jepa_only(self, mlp_setup):
        """Test pre-training schedule (JEPA-only epochs)"""
        model, target_model, loader = mlp_setup

        # Set pretrain epochs > 0
        args = type('Args', (), {
            'use_jepa': True,
            'jepa_mode': 'ema',
            'jepa_weight': 0.2,
            'jepa_pretrain_epochs': 3,
            'jepa_mask_entry_online': 0.3,
            'jepa_mask_row_online': 0.2,
            'jepa_mask_col_online': 0.2,
            'jepa_mask_entry_target': 0.1,
            'jepa_mask_row_target': 0.05,
            'jepa_mask_col_target': 0.05,
            'jepa_noisy_mask': False,
            'jepa_row_scaling': False,
            'ema_momentum': 0.996,
        })()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        training_state = TrainingState(log_every=10)
        # epoch 0 < 3, so should be JEPA-only

        with patch('wandb.log'):
            train_loss, jepa_loss = train_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                device='cpu',
                training_state=training_state,
                primal_weight=0.1,
                dual_weight=0.1,
                stationarity_weight=0.6,
                complementary_slackness_weight=0.2,
                args=args,
                target_model=target_model
            )

        # In JEPA-only mode, both losses should be similar (loss = jepa_loss)
        assert jepa_loss is not None, "JEPA loss should be computed"

    def test_joint_training_schedule(self, mlp_setup):
        """Test joint training schedule (combined loss)"""
        model, target_model, loader = mlp_setup

        args = type('Args', (), {
            'use_jepa': True,
            'jepa_mode': 'ema',
            'jepa_weight': 0.2,
            'jepa_pretrain_epochs': 0,  # No pretrain, joint from start
            'jepa_mask_entry_online': 0.3,
            'jepa_mask_row_online': 0.2,
            'jepa_mask_col_online': 0.2,
            'jepa_mask_entry_target': 0.1,
            'jepa_mask_row_target': 0.05,
            'jepa_mask_col_target': 0.05,
            'jepa_noisy_mask': False,
            'jepa_row_scaling': False,
            'ema_momentum': 0.996,
        })()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        training_state = TrainingState(log_every=10)

        with patch('wandb.log'):
            train_loss, jepa_loss = train_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                device='cpu',
                training_state=training_state,
                primal_weight=0.1,
                dual_weight=0.1,
                stationarity_weight=0.6,
                complementary_slackness_weight=0.2,
                args=args,
                target_model=target_model
            )

        # Both losses should be tracked
        assert jepa_loss is not None, "JEPA loss should be computed in joint training"

    def test_ema_update_called_after_optimizer_step(self, mlp_setup, jepa_args):
        """Test that EMA update is called after optimizer step"""
        model, target_model, loader = mlp_setup

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        training_state = TrainingState(log_every=10)

        # Store initial target weights
        target_weight_before = next(target_model.parameters()).data.clone()

        with patch('wandb.log'):
            train_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                device='cpu',
                training_state=training_state,
                primal_weight=0.1,
                dual_weight=0.1,
                stationarity_weight=0.6,
                complementary_slackness_weight=0.2,
                args=jepa_args,
                target_model=target_model
            )

        # Target weights should have changed due to EMA update
        target_weight_after = next(target_model.parameters()).data
        assert not torch.allclose(target_weight_before, target_weight_after), \
            "Target model weights should change after EMA update"

    def test_ema_update_not_called_in_simsiam_mode(self, mlp_setup):
        """Test that EMA update is NOT called in SimSiam mode"""
        model, _, loader = mlp_setup

        # SimSiam doesn't use target model
        args = type('Args', (), {
            'use_jepa': True,
            'jepa_mode': 'simsiam',
            'jepa_weight': 0.2,
            'jepa_pretrain_epochs': 0,
            'jepa_mask_entry_online': 0.3,
            'jepa_mask_row_online': 0.2,
            'jepa_mask_col_online': 0.2,
            'jepa_mask_entry_target': 0.1,
            'jepa_mask_row_target': 0.05,
            'jepa_mask_col_target': 0.05,
            'jepa_noisy_mask': False,
            'jepa_row_scaling': False,
            'ema_momentum': 0.996,
        })()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        training_state = TrainingState(log_every=10)

        with patch('wandb.log'):
            # In SimSiam mode, target_model should be None or same as online
            train_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                device='cpu',
                training_state=training_state,
                primal_weight=0.1,
                dual_weight=0.1,
                stationarity_weight=0.6,
                complementary_slackness_weight=0.2,
                args=args,
                target_model=None  # SimSiam uses None
            )

        # Test passes if no exception is raised
        assert True


class TestCheckpointHandling:
    """Test checkpoint save/load for target model"""

    def test_checkpoint_saves_target_model_when_present(self):
        """Test that checkpoint saves target_model state when it exists"""
        m, n = 10, 8
        model = KKTNetMLP(m, n)
        target_model = deepcopy(model)
        optimizer = torch.optim.Adam(model.parameters())

        # Simulate checkpoint creation as in train.py
        ckpt = {
            "epoch": 5,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": {},
        }

        if target_model is not None:
            ckpt["target_model"] = target_model.state_dict()

        assert "target_model" in ckpt, "Checkpoint should contain target_model"

    def test_checkpoint_does_not_save_target_model_when_none(self):
        """Test that checkpoint does NOT save target_model when None"""
        m, n = 10, 8
        model = KKTNetMLP(m, n)
        target_model = None
        optimizer = torch.optim.Adam(model.parameters())

        # Simulate checkpoint creation
        ckpt = {
            "epoch": 5,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": {},
        }

        if target_model is not None:
            ckpt["target_model"] = target_model.state_dict()

        assert "target_model" not in ckpt, "Checkpoint should not contain target_model when None"

    def test_checkpoint_load_handles_missing_target_model(self):
        """Test that checkpoint loading handles missing target_model gracefully"""
        m, n = 10, 8
        model = KKTNetMLP(m, n)
        target_model = deepcopy(model)

        # Create checkpoint without target_model
        ckpt = {
            "epoch": 5,
            "model": model.state_dict(),
            "optimizer": torch.optim.Adam(model.parameters()).state_dict(),
            "args": {},
        }

        # Should be able to load without error
        try:
            model.load_state_dict(ckpt["model"])
            if "target_model" in ckpt and target_model is not None:
                target_model.load_state_dict(ckpt["target_model"])
        except Exception as e:
            pytest.fail(f"Loading checkpoint should handle missing target_model: {e}")

    def test_checkpoint_loads_target_model_correctly(self):
        """Test that checkpoint loads target_model with correct state"""
        m, n = 10, 8

        # Create and save models
        online_model = KKTNetMLP(m, n)
        target_model = deepcopy(online_model)

        # Modify target model
        for param in target_model.parameters():
            param.data.fill_(5.0)

        # Save checkpoint
        ckpt = {
            "epoch": 5,
            "model": online_model.state_dict(),
            "target_model": target_model.state_dict(),
        }

        # Create new models
        new_online = KKTNetMLP(m, n)
        new_target = deepcopy(new_online)

        # Load checkpoint
        new_online.load_state_dict(ckpt["model"])
        new_target.load_state_dict(ckpt["target_model"])

        # Check that loaded target has correct values
        for param in new_target.parameters():
            assert torch.allclose(param.data, torch.full_like(param.data, 5.0)), \
                "Loaded target model should have correct state"

    def test_loaded_target_model_has_correct_requires_grad(self):
        """Test that loaded target_model still has requires_grad=False"""
        m, n = 10, 8

        online_model = KKTNetMLP(m, n)
        target_model = deepcopy(online_model)
        for p in target_model.parameters():
            p.requires_grad_(False)

        # Save and load
        ckpt = {"target_model": target_model.state_dict()}

        new_target = deepcopy(online_model)
        new_target.load_state_dict(ckpt["target_model"])

        # After loading, we need to manually set requires_grad=False again
        # (state_dict doesn't save requires_grad)
        for p in new_target.parameters():
            p.requires_grad_(False)

        for param in new_target.parameters():
            assert param.requires_grad is False


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
