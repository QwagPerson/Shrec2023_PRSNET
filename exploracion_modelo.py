import polyscope as ps
from lightning import Trainer

from dataset.lightning_voxel_dataset import VoxelDataModule
from model.prsnet.lightning_prsnet import LightingPRSNet
from model.prsnet.losses import apply_symmetry
from model.prsnet.metrics import undo_transform_representation
from setup.setup_voxel_dataset.symmetry_plane import SymmetryPlane


def visualize_prediction(pred_planes, points, real_planes):
    """
    :param pred_planes: N x 7
    :param points: S x 3
    :param real_planes: M x 6
    """
    # Create symmetryPlane Objs
    original_symmetries = [
        SymmetryPlane(
            normal=real_planes[idx, 0:3].detach().numpy(),
            point=real_planes[idx, 3::].detach().numpy(),
        )
        for idx in range(real_planes.shape[0])
    ]

    predicted_symmetries = [
        SymmetryPlane(
            normal=pred_planes[idx, 0:3].detach().numpy(),
            point=pred_planes[idx, 3::].detach().numpy(),
        )
        for idx in range(pred_planes.shape[0])
    ]

    other_rep_pred_planes = undo_transform_representation(pred_planes.unsqueeze(0)).squeeze(dim=0)
    # Reflect points
    reflected_points = [
        apply_symmetry(points, other_rep_pred_planes[idx, 0:3], other_rep_pred_planes[idx, 3].unsqueeze(dim=0))
        for idx in range(other_rep_pred_planes.shape[0])
    ]

    # Visualize
    ps.init()
    ps.remove_all_structures()

    ps.register_point_cloud("original pcd", points.detach().numpy())

    for idx, sym_plane in enumerate(original_symmetries):
        ps.register_surface_mesh(
            f"original_sym_plane_{idx}",
            sym_plane.coords,
            sym_plane.trianglesBase,
            enabled=False,
        )

    for idx, sym_plane in enumerate(predicted_symmetries):
        ps.register_surface_mesh(
            f"predicted_sym_plane_{idx}",
            sym_plane.coords,
            sym_plane.trianglesBase,
            enabled=True,
            transparency=0.65
        )

    for idx, ref_points in enumerate(reflected_points):
        ps.register_point_cloud(f"reflected_points_{idx}", ref_points.detach().numpy(), enabled=False, )
    ps.show()


def visualize_prediction_results(prediction, visualize_unscaled=True):
    """
    :param prediction:
    :param visualize_unscaled:
    :return:
    """
    prediction = [x.float() for x in prediction]
    fig_idx, y_out, sample_points_out, y_pred, sample_points, y_true, y_true_out = prediction

    batch_size = sample_points_out.shape[0]

    for batch_idx in range(batch_size):
        print(y_out[batch_idx, :, -1])
        if visualize_unscaled:
            visualize_prediction(
                pred_planes=y_out[batch_idx, :, :],
                real_planes=y_true_out[batch_idx, :, :],
                points=sample_points_out[batch_idx, :, :]
            )
        else:
            visualize_prediction(
                pred_planes=y_pred[batch_idx, :, :],
                real_planes=y_true[batch_idx, :, :],
                points=sample_points[batch_idx, :, :]
            )


if __name__ == "__main__":
    #  max_sde=0.023, angle_threshold=10, phc_angle=1, phc_dist_percent=0.01
    MODEL_PATH = "modelos_interesantes/version_13_so_many_heads_omaigai/checkpoints/epoch=14-step=12660.ckpt"
    model = LightingPRSNet.load_from_checkpoint(MODEL_PATH,
                                                max_sde=0.023, angle_threshold=10,
                                                phc_angle=1, phc_dist_percent=0.01
                                                )
    data_module = VoxelDataModule(
        test_data_path="/data/voxel_dataset_v2",
        predict_data_path="/data/voxel_dataset_v2",
        train_val_split=1,
        batch_size=1,
        sample_size=2048,
        shuffle=False
    )
    trainer = Trainer(enable_progress_bar=True)

    predictions_results = trainer.predict(model, data_module)

    for pred in predictions_results:
        visualize_prediction_results(pred, visualize_unscaled=True)

    """    for pred in predictions_results:
        visualize_prediction_results(pred, visualize_unscaled=False)
        break"""

