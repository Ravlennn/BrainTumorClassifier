import numpy as np
import pandas as pd


def get_component_ids(lbl):
    '''
    Return a list of ids of all components found in the labeled mask.
    '''
    ids = np.unique(lbl)
    ids = ids[ids != 0]

    return [int(x) for x in ids]


def get_component_properties(lbl, timepoint=None, spacing=(1.0, 1.0, 1.0)):
    '''
    Calculates the main properties of each focus in the labeled mask:
    - number of voxels;
    - volume in mm3;
    - centroid coordinates in voxels;
    - centroid coordinates in mm;
    - component bounding box.
    '''
    rows = []

    spacing = np.array(spacing)
    voxel_volume = float(np.prod(spacing))

    for comp_id in get_component_ids(lbl):
        mask = lbl == comp_id
        coords = np.argwhere(mask)

        if coords.size == 0:
            continue

        voxel_count = int(mask.sum())
        volume_mm3 = voxel_count * voxel_volume

        centroid_voxel = coords.mean(axis=0)
        centroid_mm = centroid_voxel * spacing

        rows.append({
            "timepoint": timepoint,
            "component_id": comp_id,

            "voxel_count": voxel_count,
            "volume_mm3": volume_mm3,

            "centroid_z": float(centroid_voxel[0]),
            "centroid_y": float(centroid_voxel[1]),
            "centroid_x": float(centroid_voxel[2]),

            "centroid_z_mm": float(centroid_mm[0]),
            "centroid_y_mm": float(centroid_mm[1]),
            "centroid_x_mm": float(centroid_mm[2]),

            "bbox_z_min": int(coords[:, 0].min()),
            "bbox_y_min": int(coords[:, 1].min()),
            "bbox_x_min": int(coords[:, 2].min()),

            "bbox_z_max": int(coords[:, 0].max()),
            "bbox_y_max": int(coords[:, 1].max()),
            "bbox_x_max": int(coords[:, 2].max()),
        })

    return pd.DataFrame(rows)


def build_lesion_path(timepoints, track):
    '''
    Constructs path of the lesions based on time points.
    '''
    parts = []

    for tp, comp_id in zip(timepoints, track):
        if comp_id is None:
            parts.append("None")
        else:
            parts.append(f"{tp}_{int(comp_id)}")

    return " -> ".join(parts)


def find_match_row(matches_df, comp_a, comp_b):
    '''
    Finds a string with a specific match comp_a -> comp_b.
    '''
    if matches_df is None or matches_df.empty:
        return None

    row = matches_df[
        (matches_df["component_id_a"] == comp_a)
        & (matches_df["component_id_b"] == comp_b)
    ]

    if row.empty:
        return None

    return row.iloc[0]
