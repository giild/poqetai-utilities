import argparse
import sys
from pathlib import Path
from collections import OrderedDict
from safetensors import safe_open


def get_layer_type(param_name: str, shape: tuple) -> str:
    """
    Infer layer type from parameter name and shape.
    """
    name_lower = param_name.lower()
    
    # Common layer type patterns
    if 'embed' in name_lower:
        return 'Embedding'
    elif 'layernorm' in name_lower or 'layer_norm' in name_lower:
        return 'LayerNorm'
    elif 'attention' in name_lower:
        if 'query' in name_lower or 'q_proj' in name_lower:
            return 'Attention Query'
        elif 'key' in name_lower or 'k_proj' in name_lower:
            return 'Attention Key'
        elif 'value' in name_lower or 'v_proj' in name_lower:
            return 'Attention Value'
        elif 'output' in name_lower or 'o_proj' in name_lower:
            return 'Attention Output'
        else:
            return 'Attention'
    elif 'mlp' in name_lower or 'feed_forward' in name_lower:
        if 'gate' in name_lower:
            return 'MLP Gate'
        elif 'up' in name_lower:
            return 'MLP Up'
        elif 'down' in name_lower:
            return 'MLP Down'
        else:
            return 'MLP'
    elif 'linear' in name_lower:
        return 'Linear'
    elif 'conv' in name_lower:
        if len(shape) == 4:
            return 'Conv2D'
        elif len(shape) == 3:
            return 'Conv1D'
        else:
            return 'Convolution'
    elif 'bias' in name_lower:
        return 'Bias'
    elif 'weight' in name_lower:
        if len(shape) == 2:
            return 'Linear Weight'
        elif len(shape) == 1:
            return 'Bias/Scale'
        else:
            return 'Weight'
    elif 'lm_head' in name_lower:
        return 'Language Model Head'
    elif 'classifier' in name_lower:
        return 'Classifier'
    else:
        return 'Unknown'


def format_shape(shape: tuple) -> str:
    """Format shape tuple as string."""
    return f"({', '.join(map(str, shape))})"


def get_parameter_count(shape: tuple) -> int:
    """Calculate number of parameters from shape."""
    count = 1
    for dim in shape:
        count *= dim
    return count


def format_param_count(count: int) -> str:
    """Format parameter count with appropriate units."""
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f}B"
    elif count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.2f}K"
    else:
        return str(count)


def inspect_safetensor_model(model_path: str, show_config: bool = True, max_layers: int = None):
    """
    Inspect a safetensor model and print layer information.
    
    Args:
        model_path: Path to the model directory or safetensor file
        show_config: Whether to show model configuration
        max_layers: Maximum number of layers to show (None for all)
    """
    model_path = Path(model_path)
    
    # Find safetensor files
    if model_path.is_dir():
        safetensor_files = list(model_path.glob("*.safetensors"))
        if not safetensor_files:
            print(f"No .safetensors files found in {model_path}")
            return

    else:
        safetensor_files = [model_path]
    
    print("=" * 80)
    print("MODEL LAYERS")
    print("=" * 80)
    print(f"{'Layer Name':<40} {'Type':<20} {'Shape':<15} {'Parameters':<12}")
    print("-" * 80)
    
    total_params = 0
    layer_count = 0
    all_layers = OrderedDict()
    
    # Collect all parameters from all safetensor files
    for safetensor_file in sorted(safetensor_files):
        try:
            with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    all_layers[key] = {
                        'shape': tensor.shape,
                        'dtype': tensor.dtype,
                        'file': safetensor_file.name
                    }
        except Exception as e:
            print(f"Error reading {safetensor_file}: {e}")
            continue
    
    # Print layer information
    for param_name, info in all_layers.items():
        if max_layers and layer_count >= max_layers:
            remaining = len(all_layers) - layer_count
            print(f"... and {remaining} more layers")
            break
            
        shape = tuple(info['shape'])
        layer_type = get_layer_type(param_name, shape)
        param_count = get_parameter_count(shape)
        
        # Truncate long parameter names
        display_name = param_name if len(param_name) <= 38 else param_name[:35] + "..."
        
        print(f"{display_name:<40} {layer_type:<20} {format_shape(shape):<15} {format_param_count(param_count):<12}")
        
        total_params += param_count
        layer_count += 1
    
    print("-" * 80)
    print(f"Total Parameters: {format_param_count(total_params)} ({total_params:,})")
    print(f"Total Layers: {layer_count}")
    
    if len(safetensor_files) > 1:
        print(f"Safetensor Files: {len(safetensor_files)}")
        for f in safetensor_files:
            print(f"  - {f.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Hugging Face model layers from safetensor format"
    )
    parser.add_argument(
        "model_path",
        help="Path to model directory or safetensor file"
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Skip showing model configuration"
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        help="Maximum number of layers to display"
    )
    
    args = parser.parse_args()
    
    if not Path(args.model_path).exists():
        print(f"Error: Path '{args.model_path}' does not exist")
        sys.exit(1)
    
    try:
        inspect_safetensor_model(
            args.model_path,
            show_config=not args.no_config,
            max_layers=args.max_layers
        )
    except Exception as e:
        print(f"Error inspecting model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
