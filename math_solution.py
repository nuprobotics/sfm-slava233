import numpy as np


def triangulation(
        camera_matrix: np.ndarray,
        camera_position1: np.ndarray,
        camera_rotation1: np.ndarray,
        camera_position2: np.ndarray,
        camera_rotation2: np.ndarray,
        image_points1: np.ndarray,
        image_points2: np.ndarray
) -> np.ndarray:

    def world_to_camera(camera_position, camera_rotation):
        transformed_position = -camera_rotation.T @ camera_position
        extrinsic_matrix = np.hstack((camera_rotation.T, transformed_position.reshape(-1, 1)))
        return camera_matrix @ extrinsic_matrix

    P1 = world_to_camera(camera_position1, camera_rotation1)
    P2 = world_to_camera(camera_position2, camera_rotation2)

    points = []

    for i in range(image_points1.shape[0]):
        x1, y1 = image_points1[i]
        x2, y2 = image_points2[i]

        A = np.array([
            (x1 * P1[2] - P1[0]),
            (y1 * P1[2] - P1[1]),
            (x2 * P2[2] - P2[0]),
            (y2 * P2[2] - P2[1])
        ])

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X /= X[3]

        points.append(X[:3])

    return np.array(points)
