import argparse
import json
from math import ceil
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def calculate_effective_rank(S: torch.Tensor) -> float:
    """
    Compute the Effective Rank based on the normalized singular values.
    Effective Rank = exp(-sum(p_i * log(p_i))), where p_i = s_i / sum(s_i)
    """
    s_sum = torch.sum(S)
    if s_sum == 0:
        return 0.0

    normalized_S = S / s_sum
    normalized_S_log = torch.log(normalized_S + 1e-9)
    entropy = -torch.sum(normalized_S * normalized_S_log)
    effective_rank = torch.exp(entropy)
    return effective_rank.item()


def get_weight_map_from_files(model_path: Path) -> Dict[str, str]:
    """
    If no model.safetensors.index.json exists, scan all .safetensors files
    and build a mapping: {tensor_name: filename}.
    Raises error if duplicate tensor names are found across files.
    """
    weight_map = {}
    tensor_files = list(model_path.glob("*.safetensors"))
    if not tensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}.")

    for tensor_file in tensor_files:
        with safe_open(tensor_file, framework="pt", device="cpu") as f:
            for tensor_name in f.keys():
                if tensor_name in weight_map:
                    raise ValueError(f"Duplicate tensor name detected: {tensor_name} in multiple files.")
                weight_map[tensor_name] = tensor_file.name
    return weight_map


def get_weight_map(model_path: Path) -> Dict[str, str]:
    """
    Try to load weight map from model.safetensors.index.json.
    If not found, fall back to scanning all .safetensors files.
    """
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        print(f"Found {index_path}, loading weight map from index file...")
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)["weight_map"]
    else:
        print(f"{index_path} not found, scanning all .safetensors files to build weight map...")
        return get_weight_map_from_files(model_path)


def copy_extra_files(source: Path, target: Path):
    """
    Copy all files from source to target, except:
    - .safetensors files
    - model.safetensors.index.json
    """
    extra_files = [
        f for f in source.iterdir()
        if f.is_file() and
        f.suffix != ".safetensors" and
        f.name != "model.safetensors.index.json"
    ]
    for src_file in extra_files:
        dst_file = target / src_file.name
        dst_file.write_bytes(src_file.read_bytes())
        print(f"✅ Copied extra file: {src_file.name} → {dst_file}")


def main(args):
    path_b = Path(args.b_path).resolve()
    path_i = Path(args.i_path).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # SVD cache setup
    svd_cache_path = Path(args.svd_cache_path) if args.svd_cache_path else output_path / "svd_cache.pt"
    svd_cache = {}
    if svd_cache_path.exists():
        print(f"Loading SVD cache from {svd_cache_path}...")
        svd_cache = torch.load(svd_cache_path, map_location="cpu", weights_only=True)
        print(f"Successfully loaded {len(svd_cache)} cached SVD entries.")
    else:
        print("No SVD cache found. Computing SVDs during this run.")

    print("Using CPU for computation.")
    print("Reading model weight maps...")

    # Load weight maps from both models (supports index.json or direct .safetensors scanning)
    weight_map_b = get_weight_map(path_b)
    weight_map_i = get_weight_map(path_i)

    # Identify tensor sets
    all_tensors_b = set(weight_map_b.keys())
    all_tensors_i = set(weight_map_i.keys())
    common_tensors = all_tensors_b & all_tensors_i
    only_in_i = all_tensors_i - all_tensors_b

    print(f"Model B tensors: {len(all_tensors_b)}")
    print(f"Model I tensors: {len(all_tensors_i)}")
    print(f"Common tensors: {len(common_tensors)}")
    print(f"Tensors only in Model I: {len(only_in_i)} → Will be copied directly to output.")

    # Group tensors in Model B by their source file (for efficient I/O)
    tensors_by_file_b = defaultdict(list)
    for name, filename in weight_map_b.items():
        tensors_by_file_b[filename].append(name)

    new_weights = {}
    is_cache_updated = False

    print("Starting SVD-based interpolation (with cross-file tensor alignment)...")

    # Process each file in Model B
    for filename_b, tensor_names_in_file_b in tqdm(tensors_by_file_b.items(), desc="Processing Model B shards"):
        file_b_path = path_b / filename_b
        with safe_open(file_b_path, framework="pt", device="cpu") as f_b:

            # For each tensor in this B-file, find which file in Model I it comes from
            tensors_grouped_by_i_file = defaultdict(list)
            for name in tensor_names_in_file_b:
                if name in weight_map_i:
                    filename_i = weight_map_i[name]
                    tensors_grouped_by_i_file[filename_i].append(name)

            # Open each corresponding Model I file and process overlapping tensors
            for filename_i, tensor_names_to_process in tensors_grouped_by_i_file.items():
                file_i_path = path_i / filename_i
                with safe_open(file_i_path, framework="pt", device="cpu") as f_i:
                    for name in tensor_names_to_process:
                        tensor_b = f_b.get_tensor(name)
                        tensor_i = f_i.get_tensor(name)

                        tensor_b_float = tensor_b.to(dtype=torch.float32)
                        tensor_i_float = tensor_i.to(dtype=torch.float32)
                        diff = tensor_i_float - tensor_b_float

                        print(f"Processing tensor: {name}, shape: {diff.shape}")

                        if diff.ndim >= 2:
                            original_shape = diff.shape
                            try:
                                if name in svd_cache:
                                    U, S, Vh = svd_cache[name]
                                else:
                                    print(f"\nCache miss: Computing SVD for tensor '{name}'...")
                                    # Reshape to 2D: [out_features, -1]
                                    diff_2d = diff.reshape(original_shape[0], -1) if diff.ndim > 2 else diff

                                    # Compute SVD in float64 for numerical stability, then convert back
                                    U_f64, S_f64, Vh_f64 = torch.linalg.svd(diff_2d.to(torch.float64), full_matrices=False)
                                    U, S, Vh = U_f64.to(torch.float32), S_f64.to(torch.float32), Vh_f64.to(torch.float32)

                                    svd_cache[name] = (U, S, Vh)
                                    is_cache_updated = True

                                eff_rank = calculate_effective_rank(S)
                                k = int(ceil(eff_rank))

                                if args.gamma > 0.0:
                                    print(f"Keeping top-{k} components, scaling others by gamma={args.gamma}")
                                    S_copy = S.clone()
                                    S_copy[k:] *= args.gamma
                                    truncated_diff_2d = U @ torch.diag_embed(S_copy) @ Vh
                                else:
                                    print(f"Keeping top-{k} components, zeroing others")
                                    U_k, S_k, Vh_k = U[:, :k], S[:k], Vh[:k, :]
                                    truncated_diff_2d = U_k @ torch.diag(S_k) @ Vh_k

                                truncated_diff = truncated_diff_2d.reshape(original_shape) if diff.ndim > 2 else truncated_diff_2d
                                new_tensor_float = tensor_b_float + truncated_diff

                            except torch.linalg.LinAlgError:
                                print(f"Warning: SVD failed for tensor '{name}'. Using raw difference.")
                                new_tensor_float = tensor_b_float + diff
                        else:
                            print("Keeping original: tensor is 1D or scalar.")
                            new_tensor_float = tensor_b_float + diff

                        # Store result in target dtype
                        new_weights[name] = new_tensor_float.to(tensor_b.dtype)

    # Handle tensors that exist ONLY in Model I (not in Model B)
    print(f"\nProcessing {len(only_in_i)} tensors found only in Model I (copying directly)...")
    for name in tqdm(only_in_i, desc="Copying unique tensors from Model I"):
        filename_i = weight_map_i[name]
        file_i_path = path_i / filename_i
        with safe_open(file_i_path, framework="pt", device="cpu") as f_i:
            tensor_i = f_i.get_tensor(name)
            new_weights[name] = tensor_i  # Direct copy, no interpolation

    # Save updated SVD cache if modified
    if is_cache_updated:
        print(f"\nSaving updated SVD cache to {svd_cache_path}...")
        torch.save(svd_cache, svd_cache_path)
        print("SVD cache saved successfully.")

    # Prepare output shards
    print("\nAll interpolation complete. Organizing output shards...")
    shards = {}
    total_size = 0

    # Process tensors from Model B (interpolated)
    for tensor_name, safetensor_file in tqdm(weight_map_b.items(), desc="Organizing Model B shards"):
        if tensor_name in new_weights:
            if safetensor_file not in shards:
                shards[safetensor_file] = {}
            shards[safetensor_file][tensor_name] = new_weights[tensor_name]
            total_size += new_weights[tensor_name].numel() * new_weights[tensor_name].element_size()

    # Process tensors only in Model I (copied directly)
    for name in only_in_i:
        filename_i = weight_map_i[name]
        if filename_i not in shards:
            shards[filename_i] = {}
        shards[filename_i][name] = new_weights[name]
        total_size += new_weights[name].numel() * new_weights[name].element_size()

    # Save all shards
    print("Saving output shards...")
    for safetensor_file, tensors in tqdm(shards.items(), desc="Writing shards"):
        save_file(tensors, output_path / safetensor_file)

    # Copy index.json from Model B if it exists
    index_path_b = path_b / "model.safetensors.index.json"
    if index_path_b.exists():
        with open(index_path_b, "r", encoding="utf-8") as f:
            new_index = json.load(f)
        new_index["metadata"] = {"total_size": total_size}
        with open(output_path / "model.safetensors.index.json", "w", encoding="utf-8") as f:
            json.dump(new_index, f, indent=2, ensure_ascii=False)
    else:
        print("⚠️ No index.json found in Model B. Output will not include model.safetensors.index.json.")

    # Copy all extra files from Model I (config, tokenizer, etc.)
    print("\nCopying extra files from Model I (config, tokenizer, generation_config, etc.)...")
    copy_extra_files(path_i, output_path)

    print(f"\n✅ Success! New model saved to: {output_path}")
    print(f"  - Total shards saved: {len(shards)}")
    print(f"  - Tensors interpolated: {len(common_tensors)}")
    print(f"  - Tensors copied from Model I only: {len(only_in_i)}")
    print(f"  - Total parameters: {total_size // 1_000_000}M")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform SVD-based interpolation between two models with support for unindexed safetensors and copying extra files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("b_path", type=str, help="Path to base model (B) directory.")
    parser.add_argument("i_path", type=str, help="Path to target model (I) directory.")
    parser.add_argument("output_path", type=str, help="Output directory for the interpolated model.")
    parser.add_argument("gamma", type=float, default=0.0, help="Scaling factor for singular values beyond effective rank. Use 0.0 to zero out, >0.0 to scale down.")
    parser.add_argument(
        "--svd_cache_path",
        type=str,
        default=None,
        help="Path to SVD cache file (default: output_path/svd_cache.pt). Use to skip expensive SVD recomputation."
    )
    args = parser.parse_args()
    main(args)

# python3 -u timber.py weights/Llama-3.1-8B/Llama-3.1-8B  weights/Llama-3.1-8B/Llama-3.1-8B-Instruct/ weights/Llama-3.1-8B/Llama-3.1-8B-B-I-g00 0.0 --svd_cache_path weights/Llama-3.1-8B/svd_cache.pt