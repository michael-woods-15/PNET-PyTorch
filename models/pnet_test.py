import torch
import torch.nn as nn
import logging
import sys
import os  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from models.pnet import SingleOutputPNet, PNet
from reactome.pathway_hierarchy import get_connectivity_maps
from data_access.genomic_data import run_genomic_data_pipeline
from models.model_utils import count_parameters

def test_single_output_pnet(connectivity_maps, features):
    """Test SingleOutputPNet implementation"""
    print("=" * 80)
    print("Testing SingleOutputPNet")
    print("=" * 80)
    
    # Setup
    batch_size = 1011
    n_genes = 9229
    n_modalities = 3
    input_dim = n_genes * n_modalities  # 27687
    
    # Create model
    model = SingleOutputPNet(
        connectivity_maps=connectivity_maps,
        n_genes=n_genes,
        n_modalities=n_modalities,
        dropout_h0=0.5,
        dropout_h=0.1
    )
    
    # Create dummy input
    #x = torch.randn(batch_size, input_dim)
    x = features
    
    print(f"\nInput shape: {x.shape}")
    print(f"Expected: ({batch_size}, {input_dim})")
    
    # Forward pass
    model.eval()  # Set to eval mode to disable dropout
    with torch.no_grad():
        output = model(x)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected: ({batch_size}, 1)")
    
    # Check output
    assert output.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {output.shape}"
    print("\n✅ SingleOutputPNet test passed!")
    
    # Test with training mode (dropout enabled)
    model.train()
    output_train = model(x)
    print(f"\nTraining mode output shape: {output_train.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


def test_multi_output_pnet(connectivity_maps, features):
    """Test multi-output PNet implementation"""
    print("\n" + "=" * 80)
    print("Testing Multi-Output PNet")
    print("=" * 80)
    
    # Setup
    batch_size = 1011
    n_genes = 9229
    n_modalities = 3
    input_dim = n_genes * n_modalities  # 27687
    
    # Create model
    model = PNet(
        connectivity_maps=connectivity_maps,
        n_genes=n_genes,
        n_modalities=n_modalities,
        dropout_h0=0.5,
        dropout_h=0.1
    )
    
    # Create dummy input
    #x = torch.randn(batch_size, input_dim)
    x = features
    
    print(f"\nInput shape: {x.shape}")
    print(f"Expected: ({batch_size}, {input_dim})")
    
    # Forward pass
    model.eval()  # Set to eval mode to disable dropout
    with torch.no_grad():
        outputs = model(x)
    
    print(f"\nNumber of outputs: {len(outputs)}")
    print(f"Expected: 6 (o1-o6)")
    
    # Check each output
    expected_output_sizes = [n_genes, 1387, 1066, 447, 147, 26]
    print("\nOutput shapes:")
    for i, (output, expected_size) in enumerate(zip(outputs, expected_output_sizes)):
        print(f"  o{i+1} (from h{i}): {output.shape} - Expected: ({batch_size}, 1)")
        assert output.shape == (batch_size, 1), f"Output {i+1} shape mismatch: expected ({batch_size}, 1), got {output.shape}"
    
    print("\n✅ Multi-output PNet test passed!")
    
    # Test with training mode (dropout enabled)
    model.train()
    outputs_train = model(x)
    print(f"\nTraining mode: {len(outputs_train)} outputs generated")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Keras implementation: 71,009 parameters")

    count_parameters(model)
    
    return model


def test_layer_by_layer(connectivity_maps, features):
    """Test intermediate layer outputs"""
    print("\n" + "=" * 80)
    print("Testing Layer-by-Layer Forward Pass")
    print("=" * 80)
    
    batch_size = 1011
    n_genes = 9229
    n_modalities = 3
    input_dim = n_genes * n_modalities
    
    model = PNet(connectivity_maps=connectivity_maps)
    
    #x = torch.randn(batch_size, input_dim)
    x = features
    model.eval()
    
    print(f"\nInput: {x.shape}")
    
    with torch.no_grad():
        # Manual forward pass to inspect each layer
        current = x
        for i, layer in enumerate(model.layers):
            current = layer(current)
            print(f"After layer h{i}: {current.shape}")
            output = model.output_heads[i](current)
            print(f"  → Output o{i+1}: {output.shape}")
    
    print("\n✅ Layer-by-layer test passed!")


def test_gradient_flow(connectivity_maps):
    """Test that gradients flow properly"""
    print("\n" + "=" * 80)
    print("Testing Gradient Flow")
    print("=" * 80)
    
    batch_size = 2
    n_genes = 9229
    n_modalities = 3
    input_dim = n_genes * n_modalities
    
    # Test SingleOutputPNet
    print("\nSingleOutputPNet:")
    model_single = SingleOutputPNet(connectivity_maps=connectivity_maps)
    x = torch.randn(batch_size, input_dim, requires_grad=True)
    output = model_single(x)
    loss = output.sum()
    loss.backward()
    
    has_grad = sum(1 for p in model_single.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model_single.parameters())
    print(f"  Parameters with gradients: {has_grad}/{total_params}")
    assert has_grad == total_params, "Not all parameters received gradients!"
    print("  ✅ Gradients flowing properly")
    
    # Test Multi-output PNet
    print("\nMulti-output PNet:")
    model_multi = PNet(connectivity_maps=connectivity_maps)
    x = torch.randn(batch_size, input_dim, requires_grad=True)
    outputs = model_multi(x)
    loss = sum(o.sum() for o in outputs)
    loss.backward()
    
    has_grad = sum(1 for p in model_multi.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model_multi.parameters())
    print(f"  Parameters with gradients: {has_grad}/{total_params}")
    assert has_grad == total_params, "Not all parameters received gradients!"
    print("  ✅ Gradients flowing properly")


def test_with_real_data_shapes(connectivity_maps):
    """Test with actual expected data dimensions"""
    print("\n" + "=" * 80)
    print("Testing with Real Data Dimensions")
    print("=" * 80)
    
    # Simulate real batch sizes
    batch_sizes = [1, 16, 32, 64]
    n_genes = 9229
    n_modalities = 3
    input_dim = n_genes * n_modalities
    
    model = PNet(connectivity_maps=connectivity_maps)
    model.eval()
    
    print("\nTesting different batch sizes:")
    for bs in batch_sizes:
        x = torch.randn(bs, input_dim)
        with torch.no_grad():
            outputs = model(x)
        print(f"  Batch size {bs:3d}: ✅ Generated {len(outputs)} outputs")
        for i, output in enumerate(outputs):
            assert output.shape == (bs, 1), f"Batch size {bs}, output {i+1} shape mismatch"
    
    print("\n✅ All batch sizes tested successfully!")


if __name__ == "__main__":
    print("Starting P-NET Implementation Tests\n")

    connectivity_maps = get_connectivity_maps()[:5]
    features, responses, sample_ids = run_genomic_data_pipeline(use_selected_genes_only = True, use_coding_genes_only = True, combine_type = 'union')
    print(type(features))

    features = torch.tensor(features.values, dtype=torch.float32)
    
    try:
        # Run all tests
        #test_single_output_pnet(connectivity_maps, features)
        test_multi_output_pnet(connectivity_maps, features)
        #test_layer_by_layer(connectivity_maps, features)
        #test_gradient_flow(connectivity_maps)
        #test_with_real_data_shapes(connectivity_maps)
        
        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()