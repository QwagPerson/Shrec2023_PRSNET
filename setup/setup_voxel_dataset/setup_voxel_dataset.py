import os
import pickle
from multiprocessing import Process, Lock
import torch.multiprocessing as mp
import torch
from torch.utils.data import random_split, Dataset
import time
from typing import List, Callable
import argparse
import pathlib
from dataset.simple_points_dataset import SimplePointsDataset
from setup.setup_voxel_dataset.voxel import Voxel, normalize_pcd


def split_dataset(n_files: int, n_workers: int) -> List:
    """
    Calculates how many files every worker has to process.
    :param n_files: amount of files
    :param n_workers: amount of workers
    :return: a list of n_workers elements where the i-th element is a tuple with
    the start and end of the i-th worker.
    """
    step = n_files // n_workers
    split_jobs = []
    split_jobs += [(
        step * k, step * (k + 1)
    ) for k in range(0, n_workers - 1)]
    split_jobs += [(
        step * (n_workers - 1), n_files
    )]
    return split_jobs


def fork_safe_print(worker_name: str, msg: str, lock: Lock) -> None:
    """
    A simple safe version of print ensuring that only one process
    has access to stdout at the time.
    :param worker_name: The name of the worker which prints
    :param msg: The message to be printed in the stdout
    :param lock: The lock used to lock stdout. It has to be consistently used
    between the child processes
    :return: None
    """
    lock.acquire()
    try:
        print(f"{worker_name}: {msg}")
    finally:
        lock.release()


def transform_dataset(
        worker_name: str,
        output_path: str,
        transform_function: Callable[[int, torch.Tensor, torch.Tensor, str, dict], None],
        print_lock: Lock,
        param_dict: dict,
        dataset: Dataset,
) -> None:
    """
    Apply a transformation to a part of the dataset applying the transformation
    function to each element between start and end.
    :param worker_name: ID of the worker. Necessary for safe prints.
    :param output_path: Path leading to output data
    :param transform_function: Function applied to read and transform the old data into new data
    :param print_lock: Helper lock to use safe prints
    :param param_dict: Dict used to pass kwargs to the transform function
    :param dataset: Torch dataset containing the points and symmetries.
    :return: None
    """
    for idx, (f_idx, points, sym) in enumerate(dataset):
        start = time.time()
        transform_function(f_idx, points, sym, output_path, param_dict)
        end = time.time()
        fork_safe_print(worker_name=worker_name, msg=f"{idx} ({f_idx}) done! time spent: {end - start}",
                        lock=print_lock)


def create_folder_structure(voxel_dataset_root_path):
    os.mkdir(voxel_dataset_root_path)
    sub_folders = [
        "transformation_params",
        "points",
        "voxel_grid",
        "closest_point_voxel_grid",
        "symmetry_planes"
    ]
    for sub_folder in sub_folders:
        os.mkdir(
            os.path.join(voxel_dataset_root_path, sub_folder)
        )


def print_about_dataset(root, arguments):
    with open(os.path.join(root, "about.txt"), "w") as f:
        f.write(
            "About this dataset:\n" + str(arguments)
        )


def create_voxel_dataset(
        idx: int,
        points: torch.Tensor,
        syms: torch.Tensor,
        output_path: str,
        param_dict: dict,
) -> None:
    voxel_obj = Voxel(
        pcd=points,
        sym_planes=syms,
        env=param_dict["AMBIENTE"],
        resolution=param_dict["RESOLUTION"]
    )

    with open(os.path.join(output_path, f"transformation_params/trans_params_{idx}.pkl"), "wb") as f:
        pickle.dump({"max_norm": voxel_obj.norm, "min": voxel_obj.min}, f)

    torch.save(voxel_obj.points, os.path.join(output_path, f"points/points_{idx}.pt"))
    torch.save(voxel_obj.grid, os.path.join(output_path, f"voxel_grid/voxel_grid_{idx}.pt"))
    torch.save(voxel_obj.closest_point_grid,
               os.path.join(output_path, f"closest_point_voxel_grid/closest_point_voxel_grid_{idx}.pt"))
    torch.save(voxel_obj.symmetries_tensor, os.path.join(output_path, f"symmetry_planes/symmetry_planes_{idx}.pt"))


def complete_voxel_dataset(
        idx: int,
        points: torch.Tensor,
        syms: torch.Tensor,
        output_path: str,
        param_dict: dict,
) -> None:
    pcd, pcd_min, pcd_max_norm = normalize_pcd(points)
    with open(os.path.join(output_path, f"transformation_params/trans_params_{idx}.pkl"), "wb") as f:
        pickle.dump({"max_norm": pcd_max_norm, "min": pcd_min}, f)


def create_parser():
    new_parser = argparse.ArgumentParser(description='Create a Voxel Dataset.')
    new_parser.add_argument('--env', choices=["remote", "local"],
                            required=True, type=str,
                            help="Enviroment used to execute this script. Visualizations are skipped in remote.")

    new_parser.add_argument("--source_path",
                            required=True, type=pathlib.Path,
                            help="Path to original dataset.")

    new_parser.add_argument("--target_path",
                            required=True, type=pathlib.Path,
                            help="Path to original dataset.")

    new_parser.add_argument("--res",
                            required=True, type=int,
                            help="Voxel resolution used.")

    new_parser.add_argument("--n_workers",
                            required=False, type=int,
                            default=1,
                            help="Amount of workers transforming the dataset.")

    new_parser.add_argument("--seed",
                            required=False, type=int,
                            default=0,
                            help="Seed used.")

    new_parser.add_argument("--sample_size",
                            required=False, type=float,
                            default=1.0,
                            help="Number between 0 and 1. Percentage of used data in transformation. Is random sampled.")

    new_parser.add_argument("--device",
                            required=False, type=str,
                            default="cpu",
                            help="Device used for tensor computations.")
    return new_parser


if __name__ == '__main__':
    parser = create_parser()
    args = vars(parser.parse_args())

    AMBIENTE = args["env"]
    ORIGINAL_DATASET_PATH = args["source_path"]
    VOXEL_DATASET_ROOT = args["target_path"]
    AMOUNT_OF_WORKERS = args["n_workers"]
    RESOLUTION = args["res"]
    PERCENTAGE_USED = args["sample_size"]
    SEED = args["seed"]
    DEVICE = args["device"]

    torch.manual_seed(SEED)
    dataset_generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    # Changing how child process are spawned
    mp.set_start_method('spawn')

    if not os.path.exists(ORIGINAL_DATASET_PATH):
        raise FileNotFoundError("Original dataset not found.")

    if os.path.exists(VOXEL_DATASET_ROOT):
        raise FileExistsError(f"Root path for dataset already exists! Path used: {VOXEL_DATASET_ROOT}")

    create_folder_structure(VOXEL_DATASET_ROOT)
    print_about_dataset(VOXEL_DATASET_ROOT, args)

    workers = []
    original_dataset = SimplePointsDataset(ORIGINAL_DATASET_PATH)

    proportions = [PERCENTAGE_USED, 1 - PERCENTAGE_USED]
    lengths = [int(p * len(original_dataset)) for p in proportions]
    lengths[-1] = len(original_dataset) - sum(lengths[:-1])

    sampled_dataset = torch.utils.data.Subset(
        original_dataset,
        range(int(PERCENTAGE_USED * len(original_dataset)))
    )

    print("Sampled dataset size=", len(sampled_dataset))

    proportions = [1 / AMOUNT_OF_WORKERS for i in range(AMOUNT_OF_WORKERS)]
    lengths = [int(p * len(sampled_dataset)) for p in proportions]
    lengths[-1] = len(sampled_dataset) - sum(lengths[:-1])


    splitted_datasets = []
    init_idx = 0
    for delta in lengths:
        splitted_datasets.append(
            torch.utils.data.Subset(
                original_dataset,
                range(init_idx, init_idx+delta)
            )
        )
        init_idx+=delta

    safe_print_lock = Lock()
    for i in range(AMOUNT_OF_WORKERS):
        transform_dataset_kwargs = {
            "worker_name": f"worker_{i}",
            "output_path": VOXEL_DATASET_ROOT,
            "transform_function": create_voxel_dataset,
            "print_lock": safe_print_lock,
            "param_dict": {
                "AMBIENTE": AMBIENTE,
                "RESOLUTION": RESOLUTION,
            },
            "dataset": splitted_datasets[i],
        }

        worker = Process(target=transform_dataset, kwargs=transform_dataset_kwargs)
        worker.start()
        workers.append(worker)

    for worker in workers:
        worker.join()

    print("Global done!")
