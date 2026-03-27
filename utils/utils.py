import numpy as np


def initialization_dict(base_path):
    dataset_dict = {}

    def get_patient_number(p_id):
        num_str = ''.join(filter(str.isdigit, p_id))
        return int(num_str) if num_str else 0

    for patient_dir in base_path.iterdir():
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name
        p_num = get_patient_number(patient_id)
        dataset_dict[patient_id] = {}

        if p_num <= 53:
            for timeline_dir in patient_dir.iterdir():
                if not timeline_dir.is_dir():
                    continue

                timeline_id = timeline_dir.name
                dataset_dict[patient_id][timeline_id] = {}
                
                for file_path in timeline_dir.glob("*.nii.gz"):
                    filename = file_path.name.replace('.nii.gz', '')
                    modality = filename.split('_')[-1]
                    dataset_dict[patient_id][timeline_id][modality] = file_path

        else:
            for file_path in patient_dir.glob("*.nii.gz"):
                filename = file_path.name.replace('.nii.gz', '')
                modality = filename.split('_')[-1]
                dataset_dict[patient_id][modality] = file_path

    return dataset_dict


def match_components(lbl_a, lbl_b, n_a, n_b):
    """
    return dict {comp_id_in_A: best_matching_comp_id_in_B}
    based on IoU.
    """
    mapping = {}
    for i in range(1, n_a + 1):
        mask_a = (lbl_a == i)

        overlapping_ids = np.unique(lbl_b[mask_a])
        overlapping_ids = overlapping_ids[overlapping_ids > 0]
        if len(overlapping_ids) == 0:
            mapping[i] = None
            continue

        best_j, best_iou = None, 0.0
        for j in overlapping_ids:
            mask_b = (lbl_b == j)
            intersection = np.logical_and(mask_a, mask_b).sum()
            union = np.logical_or(mask_a, mask_b).sum()
            iou = intersection / union if union > 0 else 0.0                  

            if iou > best_iou:
                best_iou = iou
                best_j = j
        mapping[i] = (best_j, round(best_iou, 3))

    return mapping


def get_lesion_track(timepoints, labeled, n_components):
    tracks = {i: [i] for i in range(1, n_components[timepoints[0]] + 1)}

    current_ids = {i: i for i in tracks}

    for tp_idx in range(len(timepoints) - 1):
        tp_a, tp_b = timepoints[tp_idx], timepoints[tp_idx + 1]
        lbl_a, n_a = labeled[tp_a], n_components[tp_a]
        lbl_b, n_b = labeled[tp_b], n_components[tp_b]

        mapping = match_components(lbl_a, lbl_b, n_a, n_b)

        for track_id, current_comp in current_ids.items():
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

    return tracks
