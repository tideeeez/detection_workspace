"""Calibrated camera rays and planar image-to-road projection."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    matrix: np.ndarray
    distortion: np.ndarray | None

    @classmethod
    def from_config(cls, config):
        fx, fy, cx, cy = map(float, config["fx_fy_cx_cy"])
        distortion = np.asarray(config.get("distortion", []), dtype=np.float64)
        if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
            raise ValueError("Camera intrinsics require finite values and positive focal lengths.")
        if distortion.shape not in {(0,), (4,), (5,), (8,), (12,), (14,)} or not np.isfinite(distortion).all():
            raise ValueError("Invalid OpenCV distortion coefficients.")
        matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        return cls(matrix, distortion if distortion.size else None)

    def rays(self, points, *, pixel_coordinates=False):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        projection = self.matrix if pixel_coordinates else None
        return cv2.undistortPoints(points, self.matrix, self.distortion, P=projection).reshape(-1, 2)


class RoadProjection:
    def __init__(self, config, intrinsics=None):
        self.intrinsics = intrinsics
        pixels, meters = self._read_points(config["points"])
        threshold = float(config.get("ransac_threshold_m", 0.40))
        if len(pixels) < 4 or not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("Road calibration requires at least four non-collinear image/road points.")
        geometry = self.rectify(pixels)
        self.matrix, inliers = cv2.findHomography(geometry, meters, cv2.RANSAC, threshold)
        if (self.matrix is None or inliers is None or not np.isfinite(self.matrix).all()
                or np.linalg.matrix_rank(self.matrix) != 3):
            raise ValueError("Could not fit a road homography. Check point order and geometry.")
        accepted = inliers.ravel().astype(bool)
        self.hull = cv2.convexHull(pixels[accepted].astype(np.float32))
        if accepted.sum() < 4 or abs(cv2.contourArea(self.hull)) < 1.0:
            raise ValueError("Road calibration points must cover an area of the road.")
        checks = config.get("check_points", [])
        self.check_rmse_m = None
        if checks:
            check_pixels, check_meters = self._read_points(checks)
            if any(cv2.pointPolygonTest(self.hull, tuple(point), False) < 0 for point in check_pixels):
                raise ValueError("Road check points must fall inside the fitted calibration region.")
            projected = cv2.perspectiveTransform(self.rectify(check_pixels)[:, None], self.matrix)[:, 0]
            self.check_rmse_m = float(np.sqrt(np.mean(np.sum((projected - check_meters) ** 2, axis=1))))

    @staticmethod
    def _read_points(points):
        pixels = np.asarray([point["image"] for point in points], dtype=np.float64)
        meters = np.asarray([point["road_m"] for point in points], dtype=np.float64)
        if pixels.ndim != 2 or pixels.shape[1] != 2 or pixels.shape != meters.shape:
            raise ValueError("Each calibration point needs [u, v] and [x, y] coordinates.")
        if not np.isfinite(pixels).all() or not np.isfinite(meters).all():
            raise ValueError("Calibration point coordinates must be finite.")
        return pixels, meters

    def rectify(self, pixels):
        if self.intrinsics is not None and self.intrinsics.distortion is not None:
            return self.intrinsics.rays(pixels, pixel_coordinates=True)
        return pixels

    def project(self, point):
        if cv2.pointPolygonTest(self.hull, point, False) < 0:
            return None
        rectified = self.rectify(np.asarray([point], dtype=np.float64))
        predicted = self.matrix @ np.array([*rectified[0], 1])
        if abs(predicted[2]) < 1e-9:
            return None
        return predicted[:2] / predicted[2]
