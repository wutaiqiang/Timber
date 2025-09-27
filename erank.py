import os
import json
import re
import numpy as np
import torch
from safetensors import safe_open
import argparse


def parse_safetensors_index(json_path):
    with open(json_path, "r") as f:
        index_data = json.load(f)

    weight_map = index_data["weight_map"]
    layer_pattern = re.compile(r"model\.layers\.(\d+)\.")
    layers = set()
    layer_to_keys = {}

    for key in weight_map.keys():
        match = layer_pattern.search(key)
        if match:
            layer_idx = int(match.group(1))
            layers.add(layer_idx)
            if layer_idx not in layer_to_keys:
                layer_to_keys[layer_idx] = []
            layer_to_keys[layer_idx].append(key)

    total_layers = max(layers) + 1 if layers else 0
    return {
        "total_layers": total_layers,
        "layer_to_keys": layer_to_keys
    }



def load_selected_weights_optimized(folder_path, selected_keys, device="cpu"):
    weights = {}
    files = [f for f in os.listdir(folder_path) if f.endswith(".safetensors")]
    requested_keys = set(selected_keys)  
    found_keys = set()



    for filename in files:
        full_path = os.path.join(folder_path, filename)
        try:
            with safe_open(full_path, framework="pt", device=device) as f:
                file_keys = set(f.keys())
                intersect = requested_keys & file_keys
                for key in intersect:
                    tensor = f.get_tensor(key)
                    if tensor.dtype == torch.bfloat16:
                        tensor = tensor.float()
                    weights[key] = tensor.cpu().numpy()
                    found_keys.add(key)

        except Exception as e:
            print(f"[Error] Failed to read {full_path}: {e}")

    not_found = requested_keys - found_keys
    if not_found:
        for key in sorted(not_found):

            pass
    else:

        pass

    return weights


def calculate_sigma(w_A, w_B):
    diff = np.abs(w_A - w_B).sum()
    total = np.abs(w_A).sum() + np.abs(w_B).sum()
    return diff / total 

def calculate_sparsity_ratio(w_A, w_B, threshold=1e-5):
    return np.mean(np.abs(w_A - w_B) < threshold)


def _get_erank(matrix: np.ndarray) -> float:

    s = np.linalg.svd(matrix, compute_uv=False)


    s = s[s > 1e-9]
    if s.size == 0:
        return 0.0


    normalized_s = s / np.sum(s)


    # H(p) = -Σ(p_i * log(p_i))
    entropy = -np.sum(normalized_s * np.log(normalized_s))


    # eRank = exp(H)
    erank = np.exp(entropy)

    return erank


def calculate_erank(w_A: np.ndarray, w_B: np.ndarray) -> dict:

    if w_A.shape != w_B.shape:
        raise ValueError(" w_A and w_B must in same shape")
    print("matrix shape:", w_A.shape)
    w_diff = w_B - w_A

    erank_A = _get_erank(w_A)
    erank_B = _get_erank(w_B)
    erank_diff = _get_erank(w_diff)


    # results = {
    #     'eRank(w_A)': erank_A,
    #     'eRank(w_B)': erank_B,
    #     'eRank(w_B - w_A)': erank_diff
    # }
    print(f"A/B/B-A erank: {erank_A}, {erank_B}, {erank_diff}")

    cut_rank = int(np.ceil(erank_diff))
    # print(f"Cut rank: {cut_rank}")

    U, S, V = np.linalg.svd(w_diff)
    U_truncated = U[:, :cut_rank]
    S_truncated = S[:cut_rank]
    V_truncated = V[:cut_rank, :]

    # Reconstruct the truncated matrix (optional)
    w_diff_reconstructed = U_truncated @ np.diag(S_truncated) @ V_truncated
    erank_diff_new = _get_erank(w_diff_reconstructed)
    print(f"New diff erank: {erank_diff_new}")

    w_new = w_A + w_diff_reconstructed
    erank_new = _get_erank(w_new)

    print(f"New erank: {erank_new}")

    return erank_diff



def main(base_model, instruct_model):
    folder_A = base_model
    folder_B = instruct_model
    index_json_name = "model.safetensors.index.json"

    json_path_A = os.path.join(folder_A, index_json_name)
    json_info = parse_safetensors_index(json_path_A)
    num_layers = json_info["total_layers"]
    layer_to_keys = json_info["layer_to_keys"]

    all_sigmas = []  

    print(f"Total layers: {num_layers}")

    for layer_idx in range(num_layers):
        print(f"Processing Layer {layer_idx} ...")

        keys_in_layer = layer_to_keys.get(layer_idx, [])
        if not keys_in_layer:
            continue



        A_weights = load_selected_weights_optimized(folder_A, keys_in_layer.copy(), device="cpu")
        B_weights = load_selected_weights_optimized(folder_B, keys_in_layer.copy(), device="cpu")


        sigmas = []
       
        for key in keys_in_layer:
            if key in A_weights and key in B_weights and "bias" not in key and "norm" not in key:

                print(f"key: {key}")
                sigma = calculate_erank(A_weights[key], B_weights[key])
                sigmas.append(sigma)

                if "layers.2" in key:
                    assert 1==2, "stop"
                



        all_sigmas.extend(sigmas)

        del A_weights, B_weights
        torch.cuda.empty_cache()

    

    print("Average sigma across all tensors:\n", np.mean(all_sigmas))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two LLaMA models' weights using sigma metric.")
    parser.add_argument("--B", type=str, required=True, help="Path to base model (e.g., llama-3.1-8B)")
    parser.add_argument("--I", type=str, required=True, help="Path to instruct model (e.g., llama-3.1-8B-Instruct)")

    args = parser.parse_args()

    main(args.B, args.I)


    # python3 erank.py --B Qwen3-4B-Base/ --I Qwen3-4B/
