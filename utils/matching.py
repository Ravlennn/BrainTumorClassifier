import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.components import get_component_ids
from utils.matching_metrics import build_pairwise_lesion_table, add_candidate_match, compute_match_score


def get_lesion_track(
    timepoints,
    labeled,
    n_components,
    spacing=(1.0, 1.0, 1.0),
    centroid_threshold_mm=3.0,
    surface_threshold_mm=2.0,
    min_score=-0.5,
    return_tables=False,
):
    '''
    Constructs lesion trajectories across all patient time points.

    The function sequentially compares neighbor time points:
    T1 -> T2,
    T2 -> T3,
    T3 -> T4.
    '''
    if len(timepoints) == 0:
        if return_tables:
            return {}, {}
        return {}

    tracks = {}
    current_ids = {}

    next_track_id = 1

    first_tp = timepoints[0]

    for comp_id in range(1, n_components[first_tp] + 1):
        tracks[next_track_id] = [comp_id]
        current_ids[next_track_id] = comp_id
        next_track_id += 1

    all_pair_tables = {}

    for tp_idx in range(len(timepoints) - 1):
        tp_a = timepoints[tp_idx]
        tp_b = timepoints[tp_idx + 1]

        lbl_a = labeled[tp_a]
        lbl_b = labeled[tp_b]

        mapping, candidate_df, matches_df = match_components(
            lbl_a,
            lbl_b,
            n_components[tp_a],
            n_components[tp_b],
            spacing=spacing,
            centroid_threshold_mm=centroid_threshold_mm,
            surface_threshold_mm=surface_threshold_mm,
            min_score=min_score,
            return_tables=True,
        )

        all_pair_tables[f"{tp_a}_to_{tp_b}"] = {
            "candidates": candidate_df,
            "matches": matches_df,
        }

        matched_b_ids = set()

        # continue already exist tracks
        for track_id in list(tracks.keys()):
            current_comp = current_ids.get(track_id)

            if current_comp is None:
                tracks[track_id].append(None)
                continue

            result = mapping.get(current_comp)

            if result is None:
                tracks[track_id].append(None)
                current_ids[track_id] = None
            else:
                next_comp = int(result[0])
                tracks[track_id].append(next_comp)
                current_ids[track_id] = next_comp
                matched_b_ids.add(next_comp)

        # add new lesions from next timepoint
        all_b_ids = set(range(1, n_components[tp_b] + 1))
        new_b_ids = sorted(all_b_ids - matched_b_ids)

        for comp_id in new_b_ids:
            new_track = [None] * (tp_idx + 1)
            new_track.append(comp_id)

            tracks[next_track_id] = new_track
            current_ids[next_track_id] = comp_id

            next_track_id += 1

    if return_tables:
        return tracks, all_pair_tables

    return tracks


def match_components(
    lbl_a,
    lbl_b,
    n_a=None,
    n_b=None,
    spacing=(1.0, 1.0, 1.0),
    centroid_threshold_mm=3.0,
    surface_threshold_mm=2.0,
    min_score=-0.5,
    return_tables=False,
    ):
    '''
    Matches lesions between two neighbor time points.

    General logic:
    1. Get component IDs in lbl_a and lbl_b.
    2. Build a pairwise table of metrics:
    Dice, overlap, centroid distance, surface distance, volume ratio.
    3. Keep only candidate pairs:
    there is an intersection OR close centroids OR close surfaces.
    4. Calculate the match_score for each candidate pair.
    5. Build a cost matrix:
    cost = -match_score.
    6. Apply linear_sum_assignment.
    7. Discard matches with scores below min_score.
    '''
    ids_a = get_component_ids(lbl_a)
    ids_b = get_component_ids(lbl_b)

    mapping = {comp_id: None for comp_id in ids_a}

    if len(ids_a) == 0 or len(ids_b) == 0:
        if return_tables:
            return mapping, pd.DataFrame(), pd.DataFrame()
        return mapping

    pairwise_df = build_pairwise_lesion_table(
        lbl_a,
        lbl_b,
        spacing=spacing,
    )

    candidate_df = add_candidate_match(
        pairwise_df,
        centroid_threshold_mm=centroid_threshold_mm,
        surface_threshold_mm=surface_threshold_mm,
    )

    candidate_df = candidate_df[candidate_df["is_candidate_match"]].copy()

    if candidate_df.empty:
        if return_tables:
            return mapping, candidate_df, pd.DataFrame()
        return mapping

    candidate_df["match_score"] = candidate_df.apply(
        compute_match_score,
        axis=1,
    )

    a_to_idx = {comp_id: idx for idx, comp_id in enumerate(ids_a)}
    b_to_idx = {comp_id: idx for idx, comp_id in enumerate(ids_b)}

    large_cost = 1e9
    cost_matrix = np.full((len(ids_a), len(ids_b)), large_cost)

    row_lookup = {}

    for _, row in candidate_df.iterrows():
        comp_a = int(row["component_id_a"])
        comp_b = int(row["component_id_b"])

        i = a_to_idx[comp_a]
        j = b_to_idx[comp_b]

        score = float(row["match_score"])

        cost_matrix[i, j] = -score
        row_lookup[(comp_a, comp_b)] = row

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_rows = []

    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] >= large_cost:
            continue

        comp_a = ids_a[i]
        comp_b = ids_b[j]

        row = row_lookup[(comp_a, comp_b)]
        score = float(row["match_score"])

        if score < min_score:
            continue

        dice = float(row["dice"])

        mapping[comp_a] = (comp_b, round(dice, 3))
        matched_rows.append(row)

    matches_df = pd.DataFrame(matched_rows)

    if return_tables:
        return mapping, candidate_df, matches_df

    return mapping


