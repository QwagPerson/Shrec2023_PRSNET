import os

from src.metrics.eval_script import calculate_metrics_from_predictions, get_match_sequence_plane_symmetry
from src.model.prsnet.prs_net import PRSNet
from src.dataset.SymDataset.SymDataModule import SymDataModule
from src.model.prsnet.losses import SymLoss
from src.utils.voxel import transform_representation

import torch
import datetime as dt
import pandas as pd

if __name__ == "__main__":
    DATAPATH = '/data/ShapeNet_Symmetry/'
    LOGPATH = './logs'
    HEADS = 3
    REG_COEF = 25
    LOG_EVERY_N_STEPS = 5
    PRINT_EVERY_N_STEPS = 100
    RESOLUTION = 32
    NUM_STEPS = 500

    pdict = {
        "eps": 0.01,
        "theta": 0.00015230484,
        "confidence_threshold": 0.1,
    }

    # Log stuff
    current_time = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"results_{current_time}.csv"
    sde_file_name = f"sde_{current_time}.csv"

    metrics = []
    sdes = []

    if not os.path.exists(LOGPATH):
        os.mkdir(LOGPATH)

    # data
    module = SymDataModule(DATAPATH, DATAPATH, batch_size=1, resolution=RESOLUTION)
    module.setup("fit")
    dataloader = iter(module.train_dataloader())

    for batch_idx, batch in enumerate(dataloader):
        model = PRSNet(input_resolution=RESOLUTION, amount_of_heads=HEADS, use_bn=False)
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = SymLoss(REG_COEF)

        for step_num in range(NUM_STEPS):
            # Forward pass
            y_pred = model.forward(batch.get_voxel_grid_stacked())
            loss = loss_fn.forward(batch, y_pred)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step_num + 1) % PRINT_EVERY_N_STEPS == 0:
                print(f'Step [{step_num + 1}], Loss: {loss.item():.4f}')

            if (step_num + 1) % LOG_EVERY_N_STEPS == 0:
                sdes.append((batch.get_item(0).filename, step_num, loss.item()))

        y_pred = model.forward(batch.get_voxel_grid_stacked())

        in_voxel_pred = [(batch.get_voxel_points(), transform_representation(y_pred), batch.get_voxel_plane_syms())]
        in_map, in_phc, _ = calculate_metrics_from_predictions(in_voxel_pred, get_match_sequence_plane_symmetry,
                                                               pdict)

        out_voxel_pred = [(batch.get_points(), batch.get_y_pred_unscaled(y_pred), batch.get_plane_syms())]
        out_map, out_phc, _ = calculate_metrics_from_predictions(out_voxel_pred, get_match_sequence_plane_symmetry,
                                                                 pdict)
        metrics.append([batch.get_item(0).filename, in_map.item(), in_phc.item(), out_map.item(), out_phc.item()])
        print(
            f'file: ({batch.get_item(0).filename}) | in-map:  {in_map} | in-phc: {in_phc} | out-map: {out_map} | out-phc: {out_phc}')

    columms = ["filename", "in-map", "in-phc", "out-map", "out-phc"]
    pd.DataFrame(metrics, columns=columms).to_csv(os.path.join(LOGPATH, file_name), index=False)
    pd.DataFrame(sdes, columns=['filename', 'step', 'loss']).to_csv(os.path.join(LOGPATH, sde_file_name), index=False)
