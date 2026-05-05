import numpy as np
import pandas as pd

from utils.components import get_component_properties, build_lesion_path, find_match_row


def build_components(timepoints, labeled, n_components=None, spacing=(1.0, 1.0, 1.0)):
    '''
    Builds a general table of properties for all patient lesions.

    For each timepoint, the function takes a labeled mask and calculates the properties of each individual component using get_component_properties:
    - component_id;
    - voxel count;
    - volume;
    - centroid;
    - bounding box.
    '''
    tables = []

    for tp in timepoints:
        df = get_component_properties(
            labeled[tp],
            timepoint=tp,
            spacing=spacing,
        )

        if n_components is not None:
            df["n_components_at_timepoint"] = n_components[tp]

        tables.append(df)

    if len(tables) == 0:
        return pd.DataFrame()

    return pd.concat(tables, ignore_index=True)


def build_tracks(timepoints, tracks):
    '''
    Builds a table of trajectories of lesions.
    Each row in the table corresponds to one track.
    '''
    rows = []

    for track in tracks.values():
        row = {}

        for tp, comp_id in zip(timepoints, track):
            row[f"{tp}_component_id"] = comp_id

        row["lesion_path"] = build_lesion_path(timepoints, track)

        rows.append(row)

    return pd.DataFrame(rows)


def build_unmatched(timepoints, tracks, n_components):
    '''
    Builds a table of new and lost lesion between neighbor timepoints.

    For each pair of assigned timepoints function set:
    - any components from the first time point were not continued;
    - which components appeared as new at the second timepoint.
    '''
    rows = []

    for i in range(len(timepoints) - 1):
        tp_a = timepoints[i]
        tp_b = timepoints[i + 1]

        all_a_ids = set(range(1, n_components[tp_a] + 1))
        all_b_ids = set(range(1, n_components[tp_b] + 1))

        matched_a_ids = set()
        matched_b_ids = set()

        for track in tracks.values():
            comp_a = track[i]
            comp_b = track[i + 1]

            if comp_a is not None and comp_b is not None:
                matched_a_ids.add(comp_a)
                matched_b_ids.add(comp_b)

        disappeared_ids = sorted(all_a_ids - matched_a_ids)
        new_ids = sorted(all_b_ids - matched_b_ids)

        rows.append({
            "from_timepoint": tp_a,
            "to_timepoint": tp_b,
            "status": "disappeared",
            "component_ids": disappeared_ids,
            "count": len(disappeared_ids),
        })

        rows.append({
            "from_timepoint": tp_a,
            "to_timepoint": tp_b,
            "status": "new",
            "component_ids": new_ids,
            "count": len(new_ids),
        })

    return pd.DataFrame(rows)


def build_volume_change(
    timepoints,
    tracks,
    labeled,
    relative_threshold=0.25,
    absolute_threshold_voxels=5,
):
    '''
    Builds a volume change table for the identified lesions.

    For each pair, the following is calculated:
    - Volume at the first timepoint;
    - Volume at the second timepoint;
    - Absolute volume change;
    - Relative change;
    - Percentage change;
    - Change status: increased, decreased, or stable.
    '''
    rows = []

    for i in range(len(timepoints) - 1):
        tp_a = timepoints[i]
        tp_b = timepoints[i + 1]

        lbl_a = labeled[tp_a]
        lbl_b = labeled[tp_b]

        for track in tracks.values():
            comp_a = track[i]
            comp_b = track[i + 1]

            # нас интересуют только реально смэтченные очаги
            if comp_a is None or comp_b is None:
                continue

            mask_a = lbl_a == comp_a
            mask_b = lbl_b == comp_b

            volume_a = int(mask_a.sum())
            volume_b = int(mask_b.sum())

            volume_change = volume_b - volume_a

            if volume_a > 0:
                volume_change_ratio = volume_change / volume_a
                change_percent = volume_change_ratio * 100
            else:
                volume_change_ratio = np.nan
                change_percent = np.nan

            if (
                volume_change >= absolute_threshold_voxels
                and volume_change_ratio >= relative_threshold
            ):
                change_status = "increased"

            elif (
                volume_change <= -absolute_threshold_voxels
                and volume_change_ratio <= -relative_threshold
            ):
                change_status = "decreased"

            else:
                change_status = "stable"

            rows.append({
                "from_timepoint": tp_a,
                "to_timepoint": tp_b,

                "component_id_a": comp_a,
                "component_id_b": comp_b,

                "volume_a_voxels": volume_a,
                "volume_b_voxels": volume_b,
                "volume_change_voxels": volume_change,

                "volume_change_ratio": volume_change_ratio,
                "change_percent": change_percent,

                "change_status": change_status,
            })

    return pd.DataFrame(rows)


def build_summary(
    timepoints,
    tracks,
    labeled,
    matching_debug_tables=None,
    spacing=(1.0, 1.0, 1.0),
    growth_relative_threshold=0.25,
    growth_absolute_threshold_mm3=5.0,
    attention_growth_threshold_percent=40.0,
    new_lesion_attention_threshold_voxels=300,
):
    '''
    Builds a summary table of lesions for manual verification.

    Each row corresponds to one lesion track. The table summarizes:
    - lesion path by timepoints;
    - first and last timepoint where the lesion was found;
    - general lesion status: new, disappeared, increased, decreased, stable, mixed;
    - volume change;
    - matching quality;
    - final needs_attention flag;
    - attention_reason reason.
    '''
    rows = []
    voxel_volume = float(np.prod(spacing))

    for track in tracks.values():
        lesion_path = build_lesion_path(timepoints, track)

        existing = [
            (tp, comp_id)
            for tp, comp_id in zip(timepoints, track)
            if comp_id is not None
        ]

        if len(existing) == 0:
            continue

        first_seen = existing[0][0]
        last_seen = existing[-1][0]

        first_tp, first_comp = existing[0]
        last_tp, last_comp = existing[-1]

        initial_volume_voxels = int((labeled[first_tp] == first_comp).sum())
        last_volume_voxels = int((labeled[last_tp] == last_comp).sum())

        initial_volume_mm3 = initial_volume_voxels * voxel_volume
        last_volume_mm3 = last_volume_voxels * voxel_volume

        if initial_volume_mm3 > 0:
            total_change_percent = (
                (last_volume_mm3 - initial_volume_mm3)
                / initial_volume_mm3
                * 100
            )
        else:
            total_change_percent = np.nan

        pair_change_percents = []
        dice_values = []
        centroid_distances = []
        surface_distances = []
        match_scores = []

        increased_count = 0
        decreased_count = 0
        stable_count = 0

        attention_reasons = []

        exists_flags = [comp_id is not None for comp_id in track]

        is_new = first_seen != timepoints[0]
        is_disappeared = last_seen != timepoints[-1]
        has_tracking_gap = False

        if any(exists_flags) and not all(exists_flags):
            first_idx = exists_flags.index(True)
            last_idx = len(exists_flags) - 1 - exists_flags[::-1].index(True)

            middle_flags = exists_flags[first_idx:last_idx + 1]

            if not all(middle_flags):
                has_tracking_gap = True

        for i in range(len(timepoints) - 1):
            tp_a = timepoints[i]
            tp_b = timepoints[i + 1]

            comp_a = track[i]
            comp_b = track[i + 1]

            if comp_a is None or comp_b is None:
                continue

            vol_a = int((labeled[tp_a] == comp_a).sum()) * voxel_volume
            vol_b = int((labeled[tp_b] == comp_b).sum()) * voxel_volume

            vol_change = vol_b - vol_a

            if vol_a > 0:
                pair_change_percent = vol_change / vol_a * 100
            else:
                pair_change_percent = np.nan

            pair_change_percents.append(pair_change_percent)

            if (
                vol_change >= growth_absolute_threshold_mm3
                and pair_change_percent >= growth_relative_threshold * 100
            ):
                increased_count += 1

            elif (
                vol_change <= -growth_absolute_threshold_mm3
                and pair_change_percent <= -growth_relative_threshold * 100
            ):
                decreased_count += 1

            else:
                stable_count += 1

            if matching_debug_tables is not None:
                pair_key = f"{tp_a}_to_{tp_b}"

                if pair_key in matching_debug_tables:
                    matches_df = matching_debug_tables[pair_key].get("matches")

                    match_row = find_match_row(
                        matches_df=matches_df,
                        comp_a=comp_a,
                        comp_b=comp_b,
                    )

                    if match_row is not None:
                        if "dice" in match_row:
                            dice_values.append(float(match_row["dice"]))

                        if "centroid_distance_mm" in match_row:
                            centroid_distances.append(
                                float(match_row["centroid_distance_mm"])
                            )

                        if "surface_distance_mm" in match_row:
                            surface_distances.append(
                                float(match_row["surface_distance_mm"])
                            )

                        if "match_score" in match_row:
                            match_scores.append(float(match_row["match_score"]))

        if len(pair_change_percents) > 0:
            max_pair_change_percent = float(
                max(pair_change_percents, key=lambda x: abs(x))
            )
        else:
            max_pair_change_percent = np.nan

        min_dice = min(dice_values) if len(dice_values) > 0 else np.nan

        max_centroid_distance_mm = (
            max(centroid_distances)
            if len(centroid_distances) > 0
            else np.nan
        )

        max_surface_distance_mm = (
            max(surface_distances)
            if len(surface_distances) > 0
            else np.nan
        )

        min_match_score = (
            min(match_scores)
            if len(match_scores) > 0
            else np.nan
        )

        if is_new:
            lesion_status = "new"
        elif is_disappeared:
            lesion_status = "disappeared"
        elif increased_count > 0 and decreased_count > 0:
            lesion_status = "mixed"
        elif increased_count > 0:
            lesion_status = "increased"
        elif decreased_count > 0:
            lesion_status = "decreased"
        else:
            lesion_status = "stable"

        if np.isnan(min_dice):
            matching_quality = "not_applicable"
        elif (
            min_dice >= 0.5
            and max_centroid_distance_mm <= 3.0
            and max_surface_distance_mm <= 2.0
        ):
            matching_quality = "good"
        elif (
            min_dice >= 0.2
            and max_centroid_distance_mm <= 6.0
            and max_surface_distance_mm <= 4.0
        ):
            matching_quality = "medium"
        else:
            matching_quality = "poor"

        # attention rules:
        # 1. new lesion only if it is large enough in voxels
        # 2. increased only if total growth >= attention_growth_threshold_percent
        # 3. mixed only if max positive pair growth >= attention_growth_threshold_percent
        # 4. tracking gap is suspicious
        # 5. poor matching quality is suspicious

        if is_new:
            if initial_volume_voxels >= new_lesion_attention_threshold_voxels:
                attention_reasons.append(
                    f"new lesion at {first_seen}"
                )

        if (
            lesion_status == "increased"
            and not np.isnan(total_change_percent)
            and total_change_percent >= attention_growth_threshold_percent
        ):
            attention_reasons.append(
                f"total volume increased by {total_change_percent:.1f}%"
            )

        if lesion_status == "mixed" and len(pair_change_percents) > 0:
            valid_pair_changes = [
                x for x in pair_change_percents
                if not np.isnan(x)
            ]

            if len(valid_pair_changes) > 0:
                max_growth_percent = max(valid_pair_changes)

                if max_growth_percent >= attention_growth_threshold_percent:
                    attention_reasons.append(
                        f"mixed dynamics with growth up to {max_growth_percent:.1f}%"
                    )

        if has_tracking_gap:
            attention_reasons.append("tracking gap / intermittent lesion")

        if matching_quality == "poor":
            attention_reasons.append("poor matching quality")

        attention_reasons = list(dict.fromkeys(attention_reasons))

        needs_attention = len(attention_reasons) > 0

        rows.append({
            "lesion_path": lesion_path,

            "first_seen": first_seen,
            "last_seen": last_seen,
            "lesion_status": lesion_status,

            "initial_volume_mm3": initial_volume_mm3,
            "last_volume_mm3": last_volume_mm3,
            "total_change_percent": total_change_percent,

            "max_pair_change_percent": max_pair_change_percent,

            "min_dice": min_dice,
            "max_centroid_distance_mm": max_centroid_distance_mm,
            "max_surface_distance_mm": max_surface_distance_mm,
            "min_match_score": min_match_score,
            "matching_quality": matching_quality,

            "needs_attention": needs_attention,
            "attention_reason": "; ".join(attention_reasons),
        })

    return pd.DataFrame(rows)

