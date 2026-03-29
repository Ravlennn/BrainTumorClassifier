import numpy as np
from scipy.ndimage import label, maximum_filter, distance_transform_edt
from scipy.ndimage import binary_fill_holes, binary_closing
from skimage.segmentation import random_walker
from scipy.optimize import linear_sum_assignment
from scipy import ndimage


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


def match_components(lbl_a, lbl_b, n_a, n_b, iou_threshold=0.05):
    """
    return dict {comp_id_in_A: best_matching_comp_id_in_B}
    based on IoU.
    """
    iou_matrix = np.zeros((n_a, n_b))
    for i in range(1, n_a + 1):
        mask_a = (lbl_a == i)
        overlapping = np.unique(lbl_b[mask_a])
        overlapping = overlapping[overlapping > 0]
        for j in overlapping:
            mask_b = (lbl_b == j)
            inter = np.logical_and(mask_a, mask_b).sum()
            union = np.logical_or(mask_a, mask_b).sum()
            iou_matrix[i-1, j-1] = inter / union if union > 0 else 0.0

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    mapping = {i: None for i in range(1, n_a + 1)}
    for i, j in zip(row_ind, col_ind):
        iou = iou_matrix[i, j]
        if iou >= iou_threshold:
            mapping[i + 1] = (j + 1, round(iou, 3))

    return mapping


def get_lesion_track(timepoints, labeled, n_components):
    tracks = {i: [i] for i in range(1, n_components[timepoints[0]] + 1)}

    current_ids = {i: i for i in tracks}

    for tp_idx in range(len(timepoints) - 1):
        tp_a, tp_b = timepoints[tp_idx], timepoints[tp_idx + 1]
        lbl_a, n_a = labeled[tp_a], n_components[tp_a]
        lbl_b, n_b = labeled[tp_b], n_components[tp_b]

        mapping = new_match_components(lbl_a, lbl_b, n_a, n_b)

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


def improved_cca(mask: np.ndarray, max_filter_size: int = 20, beta: int = 10):
    """
    distance transform + random walker.
    return (labeled_array, n_components)
    """
    dist = distance_transform_edt(mask)

    local_max = (dist == maximum_filter(dist, size=max_filter_size)) & (mask > 0)
    seeds, n_seeds = label(local_max)

    if n_seeds == 0:
        return np.zeros_like(mask), 0

    markers = seeds.copy().astype(int)
    markers[mask == 0] = -1

    labeled = random_walker(dist, markers, beta=beta, mode='bf')
    labeled[mask == 0] = 0

    n = labeled.max()
    return labeled, n


def preprocess_mask(mask: np.ndarray):

    filled = binary_fill_holes(mask).astype(np.uint8)

    preprocessed = binary_closing(filled, iterations=1).astype(np.uint8)

    # delete noise
    lbl, n = ndimage.label(preprocessed)
    for i in range(1, n + 1):
        if (lbl == i).sum() < 10:
            preprocessed[lbl == i] = 0
            
    return preprocessed


def new_match_components(lbl_a, lbl_b, n_a, n_b, iou_threshold=0.05, alpha=0.5, d_max=15.0):
    """
    score(i,j) = alpha * IoU(i,j) + (1-alpha) * exp(-d(i,j) / d_max)
    alpha=1.0 - only IoU
    alpha=0.0 - only centroids
    d_max - distance in mm at which the centroid part = exp(-1) = 0.37
    """
    centers_a = {i: np.array(ndimage.center_of_mass(lbl_a == i)) for i in range(1, n_a + 1)}
    centers_b = {j: np.array(ndimage.center_of_mass(lbl_b == j)) for j in range(1, n_b + 1)}

    iou_matrix = np.zeros((n_a, n_b))
    score_matrix = np.zeros((n_a, n_b))

    for i in range(1, n_a + 1):
        mask_a = (lbl_a == i)
        overlapping = np.unique(lbl_b[mask_a])
        overlapping = overlapping[overlapping > 0]

        for j in range(1, n_b + 1):
            # IoU
            if j in overlapping:
                mask_b = (lbl_b == j)
                inter = np.logical_and(mask_a, mask_b).sum()
                union = np.logical_or(mask_a, mask_b).sum()
                iou = inter / union if union > 0 else 0.0
            else:
                iou = 0.0

            d = np.linalg.norm(centers_a[i] - centers_b[j])
            centroid_score = np.exp(-d / d_max)

            iou_matrix[i-1, j-1]   = iou
            score_matrix[i-1, j-1] = alpha * iou + (1 - alpha) * centroid_score

    row_ind, col_ind = linear_sum_assignment(-score_matrix)

    mapping = {i: None for i in range(1, n_a + 1)}
    for i, j in zip(row_ind, col_ind):
        iou = iou_matrix[i, j]
        score = score_matrix[i, j]
        if iou >= iou_threshold or (iou == 0.0 and score >= (1 - alpha) * np.exp(-1)):
            mapping[i + 1] = (j + 1, round(iou, 3))

    return mapping
