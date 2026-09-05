import argparse
import os
import requests
import json
import numpy as np
from tqdm import tqdm

try:
    import tensorflow as tf
except ImportError:
    tf = None

def download_and_load_gpt2(model_size, models_dir, load_checkpoint=True):
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.meta",
        "model.ckpt.index",
        "model.ckpt.data-00000-of-00001",
        "vocab.bpe",
    ]

    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = f"{base_url}/{model_size}/{filename}"
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    tf_ckpt_path = None
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = None

    if load_checkpoint:
        if tf is not None:
            tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
            params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
        else:
            print("TensorFlow is not installed. Only download functionality is available.")

    return settings, params


def download_file(url, destination):
    try:
        response = requests.get(url, stream=True, verify=False)
        file_size = int(response.headers.get("content-length", 0))

        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return

        block_size = 1024
        progress_bar_description = url.split("/")[-1]
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
            with open(destination, "wb") as file:
                for chunk in response.iter_content(block_size):
                    if not chunk:
                        continue
                    progress_bar.update(len(chunk))
                    file.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        print(f"Please check the URL: {url}")


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    if tf is None:
        raise RuntimeError("TensorFlow is required to load GPT-2 checkpoint parameters.")

    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        variable_name_parts = name.split("/")[1:]

        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GPT-2 model files into a local directory")
    parser.add_argument("model_size", nargs="?", default="124M", help="GPT-2 model size to download (124M, 355M, 774M, 1558M)")
    parser.add_argument("models_dir", nargs="?", default="gpt2", help="Directory to store downloaded GPT-2 models")
    parser.add_argument("--no-checkpoint-load", action="store_true", help="Download files only without attempting to load TensorFlow checkpoint parameters")
    args = parser.parse_args()

    settings, params = download_and_load_gpt2(args.model_size, args.models_dir, load_checkpoint=not args.no_checkpoint_load)
    print(f"Downloaded GPT-2 model {args.model_size} into {os.path.join(args.models_dir, args.model_size)}")
    if params is None:
        if args.no_checkpoint_load:
            print("Model files downloaded without checkpoint loading.")
        else:
            print("Model files downloaded, but TensorFlow is not installed so checkpoint parameters were not loaded.")
    else:
        print("Model files downloaded and checkpoint parameters were loaded.")