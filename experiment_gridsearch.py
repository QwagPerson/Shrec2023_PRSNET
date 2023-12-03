import lightning as L
from dataset.voxel_dataset import VoxelDataset, collate_fn
from torch.utils.data import random_split, DataLoader, ConcatDataset
from model.prsnet.lightning_prsnet import LightingPRSNet
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import os
import pickle


def create_or_get(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)
    return folder


###################################
# 1. Constants
###################################

PATIENCE = 3
N_FOLDS = 5
MAX_EPOCHS = 100
DATA_PATH = "/data/gsanteli/voxel_dataset_32_full"
CARPETA_RESUMEN_RESULTAD0S = "resultados_remotos_3_dic"
DEF_DIR = "/remote_test_3_dic"

###################################
# 1. Grid Search Parameters ranges
###################################

BATCH_SIZE_RANGE = [32]
SAMPLE_SIZE_RANGE = [1024]
SDE_FN_RANGE = ["symloss"]
MAX_SDE_RANGE = [0.3]
ANGLE_THRESHOLD_RANGE = [30]
LOSS_USED_RANGE = ["symloss"]
REG_COEF_RANGE = [0.1, 1, 5, 10, 25]
AMOUNT_OF_HEADS_RANGE = [3, 8, 16, 32]

total_experiments = (
        len(BATCH_SIZE_RANGE) *
        len(SAMPLE_SIZE_RANGE) *
        len(SDE_FN_RANGE) *
        len(MAX_SDE_RANGE) *
        len(ANGLE_THRESHOLD_RANGE) *
        len(REG_COEF_RANGE) *
        len(LOSS_USED_RANGE) *
        len(AMOUNT_OF_HEADS_RANGE)
)


############################################
# 1. Definimos funcion de correr experimento
############################################

def run_experiment(
        experiment_name: str,
        batch_size: int,
        sample_size: int,
        sde_fn: str,
        max_sde: float,
        angle_threshold: float,
        ref_coef: float,
        loss_used: str,
        amount_of_heads: int,
        dataset_root: str,
        n_folds: int,
        patience: int,
        max_epochs: int,
        default_dir: str,
):
    # Definimos dataset para hacer el cross val
    dataset = VoxelDataset(dataset_root=dataset_root, sample_size=sample_size)

    proportions = [1 / n_folds] * n_folds
    lengths = [int(p * len(dataset)) for p in proportions]
    lengths[-1] = len(dataset) - sum(lengths[:-1])

    folds = random_split(dataset, lengths)
    results = []

    for idx_fold_val in range(len(folds)):
        train_folds = folds.copy()

        val_dataset = train_folds.pop(idx_fold_val)
        train_dataset = ConcatDataset(train_folds)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=7)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=7)

        model = LightingPRSNet(
            name=experiment_name,
            amount_of_heads=amount_of_heads,
            sde_fn=sde_fn,
            max_sde=max_sde,
            angle_threshold=angle_threshold,
            reg_coef=ref_coef,
            loss_used=loss_used,
        )

        # train with both splits
        trainer = L.Trainer(
            callbacks=[
                EarlyStopping("val_loss", patience=patience),
                ModelCheckpoint(save_top_k=3, monitor="val_loss")
            ],
            max_epochs=max_epochs,
            logger=CSVLogger(default_dir),
            default_root_dir=default_dir,

        )

        trainer.fit(model, train_dataloader, val_dataloader)

        result = trainer.test(model, val_dataloader)
        results.append(result)

    return results


############################################
# 2. Corremos los experimentos en la grilla
############################################
folder_path = create_or_get(CARPETA_RESUMEN_RESULTAD0S)

for reg_coef in REG_COEF_RANGE:
    for n_heads in AMOUNT_OF_HEADS_RANGE:
        exp_name = f"{reg_coef}_{n_heads}__prsnet_simple"
        experiment_result = run_experiment(
            experiment_name=exp_name,
            batch_size=BATCH_SIZE_RANGE[0],
            sample_size=SAMPLE_SIZE_RANGE[0],
            sde_fn=SDE_FN_RANGE[0],
            max_sde=MAX_SDE_RANGE[0],
            angle_threshold=ANGLE_THRESHOLD_RANGE[0],
            ref_coef=reg_coef,
            loss_used=LOSS_USED_RANGE[0],
            amount_of_heads=n_heads,
            dataset_root=DATA_PATH,
            n_folds=N_FOLDS,
            patience=PATIENCE,
            max_epochs=MAX_EPOCHS,
            default_dir=DEF_DIR
        )
        with open(os.path.join(folder_path, f"{exp_name}.txt"), "wb") as f:
            pickle.dump(experiment_result, f)
