import argparse
import concurrent.futures
import glob
import os
import time
from io import BytesIO


DATASET_PREFIX = {
    "aircraft": "air",
    "cub200": "cub",
}


def load_genai():
    from google import genai
    return genai


def get_api_key(cli_api_key):
    api_key = cli_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Gemini API key is required. Set GEMINI_API_KEY or GOOGLE_API_KEY, "
        )
    return api_key


def encode_image(image_path, image_size):
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return buffered.getvalue()


def process_single_image_task(args):
    client, image_path, prompt, seed, image_id, max_retries, image_size, model_name = args
    genai = load_genai()

    try:
        image_bytes = encode_image(image_path, image_size)
    except Exception as e:
        print(f"[Image {image_id}] Error processing image: {e}")
        return "Error_Image_Process"

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    genai.types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    f"{prompt}\n[Seed: {seed}]",
                ],
                config=genai.types.GenerateContentConfig(
                    temperature=0.0,
                    seed=seed,
                ),
            )
            label = response.text.strip() if response and hasattr(response, "text") else None
            return label if label else "Unknown"
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "quota" in err_msg.lower():
                wait_time = 10 + (attempt * 5)
                print(f"Quota exceeded (image {image_id}). Waiting {wait_time}s...")
            elif "503" in err_msg or "502" in err_msg or "Bad Gateway" in err_msg:
                wait_time = 5
            else:
                wait_time = 2
                print(f"[Seed {seed} | Image {image_id}] Error {attempt}/{max_retries}: {err_msg[:100]}...")

            time.sleep(wait_time)

    return "API_Overloaded"


def load_aircraft_labeling_metadata(label_type):
    from aircraft import DATA_DIR, download_and_extract, read_metadata_df

    download_and_extract()
    root_path = os.path.join(DATA_DIR, "fgvc-aircraft-2013b")
    train_df = read_metadata_df(root_path, "trainval")

    image_paths = train_df["filepath"].tolist()
    if label_type == "weak":
        class_names = sorted(train_df["weak_label_str"].unique().tolist())
    else:
        class_names = sorted(train_df["full_label_str"].unique().tolist())

    return image_paths, class_names


def load_cub_labeling_metadata(label_type):
    import pandas as pd

    from cub200 import download_cub_2011, root as CUB_ROOT

    download_cub_2011()

    base_folder = "CUB_200_2011"
    images_file = os.path.join(CUB_ROOT, base_folder, "images.txt")
    labels_file = os.path.join(CUB_ROOT, base_folder, "image_class_labels.txt")
    split_file = os.path.join(CUB_ROOT, base_folder, "train_test_split.txt")
    classes_file = os.path.join(CUB_ROOT, base_folder, "classes.txt")
    images_folder = os.path.join(CUB_ROOT, base_folder, "images")

    images_df = pd.read_csv(images_file, sep=" ", names=["img_id", "filepath"])
    labels_df = pd.read_csv(labels_file, sep=" ", names=["img_id", "target"])
    split_df = pd.read_csv(split_file, sep=" ", names=["img_id", "is_train"])
    data_df = images_df.merge(labels_df, on="img_id").merge(split_df, on="img_id")
    train_df = data_df[data_df["is_train"] == 1].copy()

    image_paths = [os.path.join(images_folder, filepath) for filepath in train_df["filepath"]]

    if label_type == "weak":
        images_df["class"] = images_df["filepath"].apply(lambda x: x.split("/")[0].split(".")[-1])
        images_df["suffix"] = images_df["class"].apply(lambda x: x.split("_")[-1])
        class_names = pd.unique(images_df["suffix"]).tolist()
    else:
        class_names = []
        with open(classes_file, "r") as f:
            for line in f:
                _, raw_name = line.strip().split(" ", 1)
                class_names.append(raw_name.split(".", 1)[1].replace("_", " "))

    return image_paths, class_names


def load_labeling_metadata(dataset, label_type):
    if dataset == "aircraft":
        return load_aircraft_labeling_metadata(label_type)
    if dataset == "cub200":
        return load_cub_labeling_metadata(label_type)
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_prompt(dataset, label_type, class_names):
    class_list = " / ".join(class_names)

    if dataset == "aircraft" and label_type == "weak":
        task = "Identify the manufacturer of the aircraft in this image."
    elif dataset == "aircraft" and label_type == "full":
        task = "Classify the aircraft variant in this image."
    elif dataset == "cub200" and label_type == "weak":
        task = "Identify the coarse bird group in this image."
    elif dataset == "cub200" and label_type == "full":
        task = "Classify the bird species in this image."
    else:
        raise ValueError(f"Unsupported labeling task: {dataset}/{label_type}")

    return (
        f"{task} Classify the image into one of the following {len(class_names)} classes:\n"
        f"({class_list})\n"
        "Output only the class name."
    )


def default_output_filename(dataset, label_type):
    prefix = DATASET_PREFIX[dataset]
    if label_type == "weak":
        return f"{prefix}_predicted_weak_labels.npy"
    return f"{prefix}_predicted_full_labels.npy"


def sorted_batch_files(batch_dir, seed):
    batch_pattern = os.path.join(batch_dir, f"seed{seed}_batch*.npy")
    return sorted(
        [f for f in glob.glob(batch_pattern) if not f.endswith(".tmp.npy")],
        key=lambda x: int(os.path.basename(x).split("_batch")[1].split(".npy")[0]),
    )


def merge_seed_batches(seed, batch_dir, output_path):
    import numpy as np

    all_labels = []
    batch_files = sorted_batch_files(batch_dir, seed)
    print(f"[Seed {seed}] Found {len(batch_files)} batch files in {batch_dir}")

    for batch_file in batch_files:
        labels = np.load(batch_file, allow_pickle=True).tolist()
        all_labels.extend(labels)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.save(output_path, np.array(all_labels, dtype=object))
    print(f"[Seed {seed}] Saved merged labels to {output_path} (total: {len(all_labels)})")
    return all_labels


def run_labeling(args):
    import numpy as np
    from tqdm import tqdm

    api_key = get_api_key(args.api_key)
    genai = load_genai()
    client = genai.Client(api_key=api_key)

    image_paths, class_names = load_labeling_metadata(args.dataset, args.label_type)
    prompt = build_prompt(args.dataset, args.label_type, class_names)

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = DATASET_PREFIX[args.dataset]
    class_names_path = os.path.join(args.output_dir, f"{prefix}_{args.label_type}_class_names.npy")
    np.save(class_names_path, np.array(class_names, dtype=object))

    output_path = args.output_path or os.path.join(
        args.output_dir,
        default_output_filename(args.dataset, args.label_type),
    )

    batch_dir = os.path.join(
        args.output_dir,
        f"{args.dataset}_{args.label_type}_batches",
        f"seed{args.seed}",
    )
    os.makedirs(batch_dir, exist_ok=True)

    batch_size = args.batch_size
    num_batches = (len(image_paths) + batch_size - 1) // batch_size
    print(f"Dataset: {args.dataset}")
    print(f"Label type: {args.label_type}")
    print(f"Images: {len(image_paths)}")
    print(f"Classes: {len(class_names)}")
    print(f"Batch size: {batch_size}")
    print(f"Workers: {args.max_workers}")
    print(f"Output: {output_path}")

    for batch_idx in range(num_batches):
        batch_file_path = os.path.join(batch_dir, f"seed{args.seed}_batch{batch_idx + 1}.npy")
        if os.path.exists(batch_file_path):
            print(f"Seed {args.seed} batch {batch_idx + 1} already done, skipping...")
            continue

        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(image_paths))
        tasks = [
            (
                client,
                image_paths[i],
                prompt,
                args.seed,
                i,
                args.max_retries,
                args.image_size,
                args.model,
            )
            for i in range(start, end)
        ]

        print(f"Processing batch {batch_idx + 1}/{num_batches} (images {start}-{end - 1})")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            labels = list(tqdm(executor.map(process_single_image_task, tasks), total=len(tasks)))

        np.save(batch_file_path, np.array(labels, dtype=object))

    merge_seed_batches(args.seed, batch_dir, output_path)


def build_parser(default_dataset=None, default_label_type=None):
    parser = argparse.ArgumentParser(description="Generate VLM labels for CUB200 or FGVC-Aircraft.")
    parser.add_argument("--dataset", default=default_dataset or "aircraft", choices=["aircraft", "cub200"])
    parser.add_argument("--label_type", default=default_label_type or "weak", choices=["weak", "full"])
    parser.add_argument("--output_dir", default="result")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--model", default="gemini-2.0-flash")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--max_workers", default=10, type=int)
    parser.add_argument("--max_retries", default=5, type=int)
    parser.add_argument("--image_size", default=224, type=int)
    return parser


def main(default_dataset=None, default_label_type=None):
    parser = build_parser(default_dataset=default_dataset, default_label_type=default_label_type)
    args = parser.parse_args()
    run_labeling(args)


if __name__ == "__main__":
    main()
