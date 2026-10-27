#!/usr/bin/env python3
"""Selectable larva association backends (SPECIFICATION.md 26.1).

Three trackers can drive the per-dish larva stream:

    bytetrack   locked v1 default; runs inside ultralytics via model.track()
    botsort     ultralytics' BoT-SORT, with global motion compensation off
    ocsort      OC-SORT, implemented here and run on model.predict() output

The split exists because ultralytics (8.4.33) ships only ByteTrack and BoT-SORT
and raises an AssertionError for any other ``tracker_type``
(``ultralytics/trackers/track.py``), so OC-SORT cannot be selected through the
library's tracker= argument. Rather than patching ultralytics internals, this
module implements OC-SORT directly over detection boxes: it depends only on
numpy and scipy, imports no ultralytics or torch, and is therefore unit-testable
without a model.

Tracker parameters live in ``config/trackers/*.yaml`` so they are versioned with
the code and recorded in provenance (SPECIFICATION.md 21) instead of being whatever the
installed ultralytics happens to default to.

OC-SORT reference: Cao et al., "Observation-Centric SORT: Rethinking SORT for
Robust Multi-Object Tracking" (CVPR 2023). The three additions over SORT are
implemented below and marked OCM / OCR / ORU.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
TRACKERS = ("bytetrack", "botsort", "ocsort")
#: the subset ultralytics can run itself via model.track(tracker=...)
ULTRALYTICS_TRACKERS = ("bytetrack", "botsort")

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "trackers"


def normalize_tracker_name(value: str) -> str:
    """Accept a bare name, a bare YAML filename or a path; return the name.

    Used as an argparse ``type=`` so the historical spelling
    ``--tracker bytetrack.yaml`` (the pre-v1.1 default, and what old provenance
    records contain) still parses. argparse applies ``type`` before checking
    ``choices``, so normalization happens first and an unknown name is still
    rejected by ``choices``.
    """
    return Path(str(value)).stem


def resolve_tracker_config(name: str, override: Optional[Path] = None) -> Path:
    """Path to the YAML config for `name`, or `override` when given."""
    if override is not None:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"tracker config not found: {path}")
        return path
    name = normalize_tracker_name(name)
    if name not in TRACKERS:
        raise ValueError(f"unknown tracker '{name}'; expected one of {TRACKERS}")
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing shipped tracker config: {path}")
    return path


def load_tracker_config(name: str, override: Optional[Path] = None) -> dict:
    """Parsed tracker config, for provenance and for constructing OCSort."""
    import yaml
    return yaml.safe_load(resolve_tracker_config(name, override).read_text())


def ocsort_from_config(cfg: dict) -> "OCSort":
    """Build an OCSort from a parsed config/trackers/ocsort.yaml."""
    return OCSort(
        det_thresh=float(cfg.get("det_thresh", 0.25)),
        max_age=int(cfg.get("max_age", 30)),
        min_hits=int(cfg.get("min_hits", 1)),
        iou_threshold=float(cfg.get("iou_threshold", 0.3)),
        delta_t=int(cfg.get("delta_t", 3)),
        inertia=float(cfg.get("inertia", 0.2)),
        use_byte=bool(cfg.get("use_byte", False)),
    )


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes -> (len(test), len(gt))."""
    if len(bb_test) == 0 or len(bb_gt) == 0:
        return np.zeros((len(bb_test), len(bb_gt)), dtype=float)
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area_t = ((bb_test[..., 2] - bb_test[..., 0])
              * (bb_test[..., 3] - bb_test[..., 1]))
    area_g = ((bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]))
    union = area_t + area_g - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def speed_direction(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """Unit direction (dy, dx) from bbox1's center to bbox2's center."""
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1], dtype=float)
    norm = float(np.sqrt(speed[0] ** 2 + speed[1] ** 2)) + 1e-6
    return speed / norm


def _speed_direction_batch(dets: np.ndarray,
                           tracks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit directions from each track's past observation to each detection.

    Returns (dy, dx), both shaped (n_tracks, n_dets).
    """
    tracks = tracks[..., np.newaxis]
    cx1 = (dets[:, 0] + dets[:, 2]) / 2.0
    cy1 = (dets[:, 1] + dets[:, 3]) / 2.0
    cx2 = (tracks[:, 0] + tracks[:, 2]) / 2.0
    cy2 = (tracks[:, 1] + tracks[:, 3]) / 2.0
    dx = cx1 - cx2
    dy = cy1 - cy2
    norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
    return dy / norm, dx / norm


def _linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(rows, cols))).reshape(-1, 2)


def k_previous_obs(observations: dict, cur_age: int, k: int) -> np.ndarray:
    """The observation ~k frames back, used to estimate a track's direction.

    OC-SORT deliberately measures direction from *observations* rather than
    from the Kalman velocity, which is what makes it robust to the filter
    drifting during a detection gap.
    """
    if not observations:
        return np.array([-1.0, -1.0, -1.0, -1.0, -1.0])
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    return observations[max(observations.keys())]


def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """xyxy -> [cx, cy, area, aspect] measurement vector."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return np.array([bbox[0] + w / 2.0, bbox[1] + h / 2.0,
                     w * h, w / h if h > 0 else 1.0]).reshape((4, 1))


def _z_to_bbox(x: np.ndarray) -> np.ndarray:
    """[cx, cy, area, aspect, ...] -> xyxy."""
    x = np.asarray(x, dtype=float).reshape(-1)
    area = max(float(x[2]), 1e-6)
    ratio = max(float(x[3]), 1e-6)
    w = np.sqrt(area * ratio)
    h = area / w if w > 0 else 0.0
    return np.array([x[0] - w / 2.0, x[1] - h / 2.0,
                     x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))


# --------------------------------------------------------------------------- #
# Constant-velocity Kalman box tracker
# --------------------------------------------------------------------------- #
class _KalmanBoxTracker:
    """SORT's 7-state constant-velocity box filter plus OC-SORT's state.

    State is [cx, cy, area, aspect, d_cx, d_cy, d_area]. Written out in numpy
    rather than pulled from `filterpy` so the pinned environment gains no new
    dependency (SPECIFICATION.md 19).
    """

    def __init__(self, bbox: np.ndarray, track_id: int, delta_t: int = 3):
        self.F = np.eye(7)
        for i in range(3):
            self.F[i, i + 4] = 1.0
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)

        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0
        self.P = np.eye(7)
        self.P[4:, 4:] *= 1000.0   # high uncertainty on unobservable velocities
        self.P *= 10.0
        self.Q = np.eye(7)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01

        self.x = np.zeros((7, 1))
        self.x[:4] = _bbox_to_z(bbox)

        self.id = track_id
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        # --- OC-SORT state ---
        self.delta_t = delta_t
        self.last_observation = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])
        self.observations: dict[int, np.ndarray] = {}
        self.velocity: Optional[np.ndarray] = None
        # snapshot of (x, P) as of the last real observation, for ORU
        self._frozen: Optional[tuple[np.ndarray, np.ndarray]] = None

    # -- filter primitives ------------------------------------------------- #
    def _kf_predict(self) -> None:
        if float(self.x[6, 0] + self.x[2, 0]) <= 0:  # area would go negative
            self.x[6] *= 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def _kf_update(self, z: np.ndarray) -> None:
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    # -- tracker interface ------------------------------------------------- #
    def predict(self) -> np.ndarray:
        self._kf_predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return _z_to_bbox(self.x)

    def update(self, bbox: Optional[np.ndarray]) -> None:
        if bbox is None:
            return

        if self.last_observation.sum() >= 0:
            # OCM: direction is measured between two real observations, using
            # one from ~delta_t frames back so a single noisy box cannot flip it
            previous_box = None
            for i in range(self.delta_t):
                dt = self.delta_t - i
                if self.age - dt in self.observations:
                    previous_box = self.observations[self.age - dt]
                    break
            if previous_box is None:
                previous_box = self.last_observation
            self.velocity = speed_direction(previous_box, bbox)

            # ORU: the track was recovered after a gap, so the filter has been
            # coasting on prediction alone and has drifted. Rewind to the last
            # real observation and replay the gap along the virtual trajectory
            # interpolated between that observation and this one, so the
            # re-activated track carries a sane state instead of the drift.
            gap = self.time_since_update - 1
            if gap > 0 and self._frozen is not None:
                self.x, self.P = (self._frozen[0].copy(),
                                  self._frozen[1].copy())
                prev = self.last_observation[:4].astype(float)
                curr = np.asarray(bbox, dtype=float)[:4]
                for step in range(1, gap + 1):
                    virtual = prev + (curr - prev) * (step / (gap + 1.0))
                    self._kf_predict()
                    self._kf_update(_bbox_to_z(virtual))
                self._kf_predict()   # the predict paired with this observation

        self.last_observation = np.asarray(bbox, dtype=float)
        self.observations[self.age] = self.last_observation
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self._kf_update(_bbox_to_z(self.last_observation))
        self._frozen = (self.x.copy(), self.P.copy())

    def get_state(self) -> np.ndarray:
        return _z_to_bbox(self.x)


# --------------------------------------------------------------------------- #
# Association (OCM)
# --------------------------------------------------------------------------- #
def associate(detections: np.ndarray, trackers: np.ndarray,
              iou_threshold: float, velocities: np.ndarray,
              previous_obs: np.ndarray, vdc_weight: float):
    """IoU association augmented with OC-SORT's momentum term.

    Returns (matches, unmatched_detections, unmatched_trackers), where matches
    is an (M, 2) array of (detection index, tracker index).
    """
    if len(trackers) == 0 or len(detections) == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(len(detections)),
                np.arange(len(trackers)))

    dy, dx = _speed_direction_batch(detections, previous_obs)
    inertia_y = np.repeat(velocities[:, 0][:, np.newaxis], dy.shape[1], axis=1)
    inertia_x = np.repeat(velocities[:, 1][:, np.newaxis], dx.shape[1], axis=1)
    diff_angle_cos = np.clip(inertia_x * dx + inertia_y * dy, -1.0, 1.0)
    diff_angle = (np.pi / 2.0 - np.abs(np.arccos(diff_angle_cos))) / np.pi

    # tracks with no usable past observation contribute no direction prior
    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0.0
    valid_mask = np.repeat(valid_mask[:, np.newaxis], dx.shape[1], axis=1)

    iou_matrix = iou_batch(detections, trackers)
    scores = np.repeat(detections[:, -1][:, np.newaxis],
                       trackers.shape[0], axis=1)
    angle_cost = ((valid_mask * diff_angle) * vdc_weight).T * scores

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = _linear_assignment(-(iou_matrix + angle_cost))
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    unmatched_dets = [d for d in range(len(detections))
                      if d not in matched_indices[:, 0]]
    unmatched_trks = [t for t in range(len(trackers))
                      if t not in matched_indices[:, 1]]

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    matches = (np.concatenate(matches, axis=0) if matches
               else np.empty((0, 2), dtype=int))
    return matches, np.array(unmatched_dets), np.array(unmatched_trks)


# --------------------------------------------------------------------------- #
# OC-SORT
# --------------------------------------------------------------------------- #
class OCSort:
    """Observation-Centric SORT over one detection stream (one dish crop).

    One instance per dish keeps identities independent across dishes, matching
    the per-dish-crop-stream policy of SPECIFICATION.md 26.1.
    """

    def __init__(self, det_thresh: float = 0.25, max_age: int = 30,
                 min_hits: int = 1, iou_threshold: float = 0.3,
                 delta_t: int = 3, inertia: float = 0.2,
                 use_byte: bool = False):
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.delta_t = delta_t
        self.inertia = inertia
        self.use_byte = use_byte
        self.trackers: list[_KalmanBoxTracker] = []
        self.frame_count = 0
        # ids are numbered per instance, so a dish's track ids do not depend on
        # how many other dishes are being tracked in the same process; the
        # dish namespacing is applied downstream by canonical_track_id
        self._next_id = 0

    def update(self, dets: Optional[np.ndarray]) -> list[Optional[int]]:
        """Advance one frame and return a track id per input detection row.

        `dets` is an (N, 5) array of [x1, y1, x2, y2, conf] in crop pixels. The
        return list is aligned with the input rows and holds None where a row
        did not join an activated track.

        This MUST be called once per frame even when there are no detections
        (pass an empty array): the aging of lost tracks and the gap accounting
        that OC-SORT's recovery depends on both happen here.
        """
        self.frame_count += 1
        dets = (np.empty((0, 5)) if dets is None or len(dets) == 0
                else np.asarray(dets, dtype=float))
        assigned: list[Optional[int]] = [None] * len(dets)

        high_idx = np.where(dets[:, 4] > self.det_thresh)[0]
        dets_high = dets[high_idx]

        # advance every existing track one frame
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t in range(len(self.trackers)):
            pos = self.trackers[t].predict()[0]
            trks[t, :4] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)
            trks = np.delete(trks, t, axis=0)

        velocities = np.array(
            [t.velocity if t.velocity is not None else np.zeros(2)
             for t in self.trackers]).reshape(-1, 2)
        last_boxes = np.array(
            [t.last_observation for t in self.trackers]).reshape(-1, 5)
        k_obs = np.array(
            [k_previous_obs(t.observations, t.age, self.delta_t)
             for t in self.trackers]).reshape(-1, 5)

        matched, unmatched_dets, unmatched_trks = associate(
            dets_high, trks, self.iou_threshold, velocities, k_obs,
            self.inertia)
        for m in matched:
            self.trackers[m[1]].update(dets_high[m[0], :])

        # BYTE-style second pass over low-score detections (off by default;
        # run_larva_inference already gates detections at --larva-conf, so
        # there is usually nothing down here to recover).
        if self.use_byte and len(unmatched_trks) > 0:
            low_idx = np.where((dets[:, 4] <= self.det_thresh)
                               & (dets[:, 4] > 0.1))[0]
            if len(low_idx) > 0:
                iou_left = iou_batch(dets[low_idx], trks[unmatched_trks])
                if iou_left.size and iou_left.max() > self.iou_threshold:
                    rematch = _linear_assignment(-iou_left)
                    keep_trk = []
                    for m in rematch:
                        det_ind, trk_ind = low_idx[m[0]], unmatched_trks[m[1]]
                        if iou_left[m[0], m[1]] < self.iou_threshold:
                            continue
                        self.trackers[trk_ind].update(dets[det_ind, :])
                        keep_trk.append(trk_ind)
                    unmatched_trks = np.setdiff1d(unmatched_trks,
                                                  np.array(keep_trk, dtype=int))

        # OCR: retry the leftovers against each lost track's *last observation*
        # rather than its drifted prediction. This is what lets a track survive
        # a multi-frame detection gap with the same id.
        if len(unmatched_dets) > 0 and len(unmatched_trks) > 0:
            left_dets = dets_high[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = iou_batch(left_dets, left_trks)
            if iou_left.size and iou_left.max() > self.iou_threshold:
                rematch = _linear_assignment(-iou_left)
                drop_det, drop_trk = [], []
                for m in rematch:
                    det_ind = unmatched_dets[m[0]]
                    trk_ind = unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets_high[det_ind, :])
                    drop_det.append(det_ind)
                    drop_trk.append(trk_ind)
                    matched = np.vstack([matched,
                                         np.array([[det_ind, trk_ind]])])
                unmatched_dets = np.setdiff1d(unmatched_dets,
                                              np.array(drop_det, dtype=int))
                unmatched_trks = np.setdiff1d(unmatched_trks,
                                              np.array(drop_trk, dtype=int))

        for i in unmatched_trks:
            self.trackers[i].update(None)

        for i in unmatched_dets:
            self._next_id += 1
            trk = _KalmanBoxTracker(dets_high[i, :], self._next_id,
                                    delta_t=self.delta_t)
            self.trackers.append(trk)
            matched = np.vstack([matched,
                                 np.array([[i, len(self.trackers) - 1]])])

        for det_i, trk_i in matched:
            trk = self.trackers[int(trk_i)]
            if trk.time_since_update < 1 and (
                    trk.hit_streak >= self.min_hits
                    or self.frame_count <= self.min_hits):
                assigned[int(high_idx[int(det_i)])] = trk.id

        self.trackers = [t for t in self.trackers
                         if t.time_since_update <= self.max_age]
        return assigned
