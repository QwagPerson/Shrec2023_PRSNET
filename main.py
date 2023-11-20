import os
import torch
from dotenv import load_dotenv
import datetime
from dataset.voxel_dataset import VoxelDataset
from model.prsnet.prs_net import PRSNet
from model.prsnet.sym_loss import SymLoss
from torch.utils.data import DataLoader
import time
from torch.nn import MSELoss

AMBIENTE = "local"
ENV_PATH = f"envs/.{AMBIENTE}.env"

env_loaded = load_dotenv(dotenv_path=ENV_PATH)
if not env_loaded:
    raise Exception(f"Environment not loaded! Using path: {ENV_PATH}\n"
                    f"{os.listdir(os.getcwd())}")

TODAY_DATE = datetime.date.today()
VOXEL_DATA_PATH = "/data/voxel_dataset/"

# Get cpu, gpu or mps device for training.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")

#################################
# Training loop
#################################

# Parameters
EPOCHS = 500
REPORTS_EVERY_N_EPOCHS = 10
BATCH_SIZE = 1
AMOUNT_OF_HEADS = 3
REG_COEF = 0
SAMPLE_SIZE = 1024
PATIENCE = 25

# Dataloader

dataset = VoxelDataset(dataset_root=VOXEL_DATA_PATH, sample_size=SAMPLE_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset.collate_fn)

epochs_without_improvement = 0
best_known_loss = 1e6

model = PRSNet(amount_of_heads=AMOUNT_OF_HEADS, out_features=4)
sym_loss = SymLoss(reg_coef=REG_COEF)  # MSELoss()
optimizer = torch.optim.Adam(model.parameters())

model.train()
for epoch in range(EPOCHS):

    # Measuring time per epoch
    t0 = time.time()

    # Training pass
    epoch_total_train_loss = 0.0
    for samples, voxel_grids, voxel_grids_cp, y_true in dataloader:
        voxel_grids = voxel_grids.to(DEVICE).float()
        voxel_grids_cp = voxel_grids_cp.to(DEVICE).float()
        samples = samples.to(DEVICE).float()
        y_true = y_true.float()

        y_pred = model.forward(voxel_grids.float())
        loss = sym_loss(y_pred, samples, voxel_grids, voxel_grids_cp, DEVICE)
        # loss = sym_loss(y_pred, y_true)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_total_train_loss += loss.item()

    if epoch_total_train_loss > best_known_loss:
        epochs_without_improvement += 1
    else:
        best_known_loss = epoch_total_train_loss
        epochs_without_improvement = 0

    if epochs_without_improvement > PATIENCE:
        # Printing epoch results
        print(f"Last epoch results N°{epoch}: "
              f"\tTotal train loss: {epoch_total_train_loss}"
              f"\tAverage train loss {epoch_total_train_loss / len(dataset)}")
        break

    # Printing epoch results
    print(f"Results of epoch n°{epoch}: "
          f"\tTotal train loss: {epoch_total_train_loss}"
          f"\tAverage train loss {epoch_total_train_loss / len(dataset)}")

model.eval()
points, voxel, voxel_cp, sym = dataset[0]
preds = model.forward(voxel.unsqueeze(0).unsqueeze(0))
print(preds)
print("======")
print(sym)
