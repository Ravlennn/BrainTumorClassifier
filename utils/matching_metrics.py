import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial.distance import cdist

from utils.components import get_component_properties


def compute_dice(mask_a, mask_b):
    '''
    Calculates the Dice similarity coefficient between two binary masks.
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    - 1.0 — complete match;
    - 0.0 — no overlap.
    '''
    mask_a = mask_a > 0
    mask_b = mask_b > 0

    intersection = np.logical_and(mask_a, mask_b).sum()
    size_sum = mask_a.sum() + mask_b.sum()

    if size_sum == 0:
        return 0.0

    return float(2 * intersection / size_sum)


def compute_overlap_stats(mask_a, mask_b):
    '''
    Calculates the main metrics of the intersection of two lesions:
    - number of intersection voxels;
    - Dice coefficient;
    - proportion of the first lesion covered by the second;
    - proportion of the second lesion covered by the first.
    '''
    mask_a = mask_a > 0
    mask_b = mask_b > 0

    volume_a = int(mask_a.sum())
    volume_b = int(mask_b.sum())

    intersection = int(np.logical_and(mask_a, mask_b).sum())

    dice = compute_dice(mask_a, mask_b)

    overlap_ratio_a = intersection / volume_a if volume_a > 0 else 0.0
    overlap_ratio_b = intersection / volume_b if volume_b > 0 else 0.0

    return {
        "intersection_voxels": intersection,
        "dice": dice,
        "overlap_ratio_a": float(overlap_ratio_a),
        "overlap_ratio_b": float(overlap_ratio_b),
    }


def compute_centroid_distance(row_a, row_b):
    '''
    Calculates the Euclidean distance between the centroids of two lesions.
    '''
    centroid_a = np.array([
        row_a["centroid_z_mm"],
        row_a["centroid_y_mm"],
        row_a["centroid_x_mm"],
    ])

    centroid_b = np.array([
        row_b["centroid_z_mm"],
        row_b["centroid_y_mm"],
        row_b["centroid_x_mm"],
    ])

    return float(np.linalg.norm(centroid_a - centroid_b))


def get_surface_mask(mask):
    '''
    Highlights the surface of a binary lesion.
    '''
    mask = mask > 0

    if mask.sum() == 0:
        return mask

    eroded = ndimage.binary_erosion(mask)
    surface = mask & ~eroded

    return surface


def compute_surface_distance(mask_a, mask_b, spacing=(1.0, 1.0, 1.0)):
    '''
    Calculates the minimum distance between the surfaces of two lesions.
    '''
    mask_a = mask_a > 0
    mask_b = mask_b > 0

    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return np.inf

    surface_a = get_surface_mask(mask_a)
    surface_b = get_surface_mask(mask_b)

    coords_a = np.argwhere(surface_a)
    coords_b = np.argwhere(surface_b)

    if len(coords_a) == 0 or len(coords_b) == 0:
        return np.inf

    spacing = np.array(spacing)

    coords_a_mm = coords_a * spacing
    coords_b_mm = coords_b * spacing

    distances = cdist(coords_a_mm, coords_b_mm)

    return float(distances.min())


def build_pairwise_lesion_table(
    lbl_a,
    lbl_b,
    spacing=(1.0, 1.0, 1.0),
):
    '''
    Creates a table of pairwise comparisons of all lesions between two timepoints.

    For each component from lbl_a and each component from lbl_b, the following are calculated:
    - volumes;
    - relative volume change;
    - intersection;
    - dice;
    - overlap ratios;
    - distance between centroids;
    - minimum distance between surfaces.
    '''
    props_a = get_component_properties(lbl_a, spacing=spacing)
    props_b = get_component_properties(lbl_b, spacing=spacing)

    rows = []

    if props_a.empty or props_b.empty:
        return pd.DataFrame()

    for _, row_a in props_a.iterrows():
        component_id_a = int(row_a["component_id"])
        mask_a = lbl_a == component_id_a

        for _, row_b in props_b.iterrows():
            component_id_b = int(row_b["component_id"])
            mask_b = lbl_b == component_id_b

            overlap_stats = compute_overlap_stats(mask_a, mask_b)

            volume_a = float(row_a["volume_mm3"])
            volume_b = float(row_b["volume_mm3"])

            if volume_a > 0:
                volume_change_ratio = (volume_b - volume_a) / volume_a
            else:
                volume_change_ratio = np.nan

            centroid_distance_mm = compute_centroid_distance(row_a, row_b)

            surface_distance_mm = compute_surface_distance(
                mask_a,
                mask_b,
                spacing=spacing,
            )

            rows.append({
                "component_id_a": component_id_a,
                "component_id_b": component_id_b,

                "volume_a_mm3": volume_a,
                "volume_b_mm3": volume_b,
                "volume_change_ratio": float(volume_change_ratio),

                "intersection_voxels": overlap_stats["intersection_voxels"],
                "dice": overlap_stats["dice"],
                "overlap_ratio_a": overlap_stats["overlap_ratio_a"],
                "overlap_ratio_b": overlap_stats["overlap_ratio_b"],

                "centroid_distance_mm": centroid_distance_mm,
                "surface_distance_mm": surface_distance_mm,
            })

    return pd.DataFrame(rows)


def add_candidate_match(
    pairwise_df,
    centroid_threshold_mm=3.0,
    surface_threshold_mm=2.0,
):
    '''
    Adds candidate-match features to the pairwise table.

    A pair of lesions is considered candidate if at least one condition is met:
    - there is an intersection;
    - the centroids are close;
    - the surfaces are close.
    '''
    if pairwise_df.empty:
        return pairwise_df.copy()

    df = pairwise_df.copy()

    df["has_overlap"] = df["intersection_voxels"] > 0
    df["close_centroid"] = df["centroid_distance_mm"] <= centroid_threshold_mm
    df["close_surface"] = df["surface_distance_mm"] <= surface_threshold_mm

    df["is_candidate_match"] = (
        df["has_overlap"]
        | df["close_centroid"]
        | df["close_surface"]
    )

    return df


def compute_match_score(row):
    '''
    Calculates the final score for a candidate pair of lesion.

    Score logic:
    - high Dice greatly improves the score;
    - high overlap in at least one direction also improves the score;
    - the presence of intersection provides an additional bonus;
    - large distances between centroids are penalized;
    - large distances between surfaces are penalized;
    - sharp volume changes are slightly penalized.

    The score is used in matching: the higher the score, the better the match.
    The score is later converted to a cost for the Hungarian algorithm:
        cost = -score
    '''
    dice = float(row["dice"])
    overlap_a = float(row["overlap_ratio_a"])
    overlap_b = float(row["overlap_ratio_b"])
    intersection = float(row["intersection_voxels"])

    centroid_distance = float(row["centroid_distance_mm"])
    surface_distance = float(row["surface_distance_mm"])

    volume_a = float(row["volume_a_mm3"])
    volume_b = float(row["volume_b_mm3"])

    if volume_a > 0 and volume_b > 0:
        volume_ratio_penalty = abs(np.log(volume_b / volume_a))
    else:
        volume_ratio_penalty = 0.0

    has_overlap_bonus = 1.0 if intersection > 0 else 0.0

    score = (
        5.0 * dice
        + 2.0 * max(overlap_a, overlap_b)
        + has_overlap_bonus
        - 0.05 * centroid_distance
        - 0.02 * surface_distance
        - 0.10 * volume_ratio_penalty
    )

    return float(score)