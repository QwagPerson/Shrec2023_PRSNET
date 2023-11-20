import os
from multiprocessing import Process, Lock

import torch
from dotenv import load_dotenv
from typing import List, Callable
from dataset.simple_points_dataset import SimplePointsDataset
from setup.setup_voxel_dataset.voxel import Voxel

AMBIENTE = "local"
ENV_PATH = f"envs/.{AMBIENTE}.env"


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
        data_path: str,
        output_path: str,
        transform_function: Callable[[int, str, str, dict], None],
        print_lock: Lock,
        param_dict: dict,
        start: int,
        end: int,
) -> None:
    """
    Apply a transformation to a part of the dataset applying the transformation
    function to each element between start and end.
    :param worker_name: ID of the worker. Necessary for safe prints.
    :param data_path: Path leading to original data
    :param output_path: Path leading to output data
    :param transform_function: Function applied to read and transform the old data into new data
    :param print_lock: Helper lock to use safe prints
    :param param_dict: Dict used to pass kwargs to the transform function
    :param start: Idx of the starting point in the dataset of the worker
    :param end: Idx of the end point in the dataset for this worker
    :return: None
    """
    for idx in range(start, end):
        transform_function(idx, data_path, output_path, param_dict)
        fork_safe_print(worker_name, f"{idx} done!", print_lock)


def create_folder_structure(voxel_dataset_root_path):
    os.mkdir(voxel_dataset_root_path)
    sub_folders = ["points", "voxel_grid", "closest_point_voxel_grid", "symmetry_planes"]
    for sub_folder in sub_folders:
        os.mkdir(
            os.path.join(voxel_dataset_root_path, sub_folder)
        )


def create_voxel_dataset(
        idx: int,
        data_path: str,
        output_path: str,
        param_dict: dict,
) -> None:
    dataset = SimplePointsDataset(data_path)
    points, symmetries = dataset[idx]
    voxel_obj = Voxel(
        pcd=points,
        sym_planes=symmetries,
        env=param_dict["AMBIENTE"],
        resolution=param_dict["RESOLUTION"]
    )
    torch.save(voxel_obj.points, os.path.join(output_path, f"points/points_{idx}.pt"))
    torch.save(voxel_obj.grid, os.path.join(output_path, f"voxel_grid/voxel_grid_{idx}.pt"))
    torch.save(voxel_obj.closest_point_grid,
               os.path.join(output_path, f"closest_point_voxel_grid/closest_point_voxel_grid_{idx}.pt"))
    torch.save(voxel_obj.symmetries_tensor, os.path.join(output_path, f"symmetry_planes/symmetry_planes_{idx}.pt"))


if __name__ == '__main__':
    env = load_dotenv(ENV_PATH)
    if not env:
        raise ValueError(f"Failed to load env, path used is {ENV_PATH}")

    ORIGINAL_DATASET_PATH = os.environ.get("ORIGINAL_DATASET_PATH")
    ORIGINAL_DATASET_LENGTH = int(os.environ.get("ORIGINAL_DATASET_LENGTH"))
    VOXEL_DATASET_ROOT = os.environ.get("VOXEL_DATASET_ROOT")
    AMOUNT_OF_WORKERS = int(os.environ.get("AMOUNT_OF_WORKERS"))
    RESOLUTION = int(os.environ.get("VOXEL_RESOLUTION"))

    if not os.path.exists(ORIGINAL_DATASET_PATH):
        raise FileNotFoundError("Original dataset not found.")

    if os.path.exists(VOXEL_DATASET_ROOT):
        raise FileExistsError(f"Root path for dataset already exists! Path used: {VOXEL_DATASET_ROOT}")

    create_folder_structure(VOXEL_DATASET_ROOT)

    workers = []
    splitted_dataset = split_dataset(ORIGINAL_DATASET_LENGTH, AMOUNT_OF_WORKERS)

    safe_print_lock = Lock()
    for i in range(AMOUNT_OF_WORKERS):
        start_idx, end_idx = splitted_dataset[i]
        transform_dataset_kwargs = {
            "worker_name": f"worker_{i}",
            "data_path": ORIGINAL_DATASET_PATH,
            "output_path": VOXEL_DATASET_ROOT,
            "transform_function": create_voxel_dataset,
            "print_lock": safe_print_lock,
            "param_dict": {
                "AMBIENTE": AMBIENTE,
                "RESOLUTION": RESOLUTION,
            },
            "start": start_idx,
            "end": end_idx,
        }

        worker = Process(target=transform_dataset, kwargs=transform_dataset_kwargs)
        worker.start()
        workers.append(worker)

    for worker in workers:
        worker.join()

    print("Global done!")
