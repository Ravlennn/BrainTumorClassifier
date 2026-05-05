from ipywidgets import interact, IntSlider  # type: ignore
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import SimpleITK as sitk  # type: ignore


def explore_3D_array(arr: np.ndarray, cmap: str = "gray"):
    '''
    Interactively views a 3D array by sections
    '''
    def fn(SLICE: int):
        plt.figure(figsize=(7, 7))
        plt.imshow(arr[SLICE, :, :], cmap=cmap)
        plt.axis("off")
        plt.show()

    interact(fn, SLICE=(0, arr.shape[0] - 1))


def explore_patient_timepoints(dataset: dict, patient_id: str,
                               modality: str = "mask", cmap: str = "hot"):
    """
    Draw masks for all timepoints for 1 Patient
    """

    patient_data = dataset[patient_id]
    timepoints = sorted(patient_data.keys())

    arrays = {}
    for tp in timepoints:
        path = patient_data[tp].get(modality)
        if path is None:
            continue
        arrays[tp] = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))

    max_slice = min(arr.shape[0] for arr in arrays.values()) - 1
    n_cols = len(arrays)

    def fn(SLICE: int):
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
        if n_cols == 1:
            axes = [axes]

        for ax, (tp, arr) in zip(axes, arrays.items()):
            ax.imshow(arr[SLICE, :, :], cmap=cmap)
            ax.set_title(f"{patient_id} / {tp}\nslice {SLICE}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    slider = IntSlider(value=max_slice // 2, min=0,
                       max=max_slice, description="SLICE")
    interact(fn, SLICE=slider)


def visualize_matching(labeled_dict, tracks_list):
    '''
    Provides an overview visualization of the tracking result for all timepoints.
    - green: the leasion is associated with a neighboring timepoint;
    - orange: a new leasion that did not appear at the first timepoint;
    - red: the leasion disappeared before the last timepoint;
    - purple: an isolated leasion that exists only at one intermediate timepoint;
    - gray: other components not included in the main categories.
    '''
    timepoints = list(labeled_dict.keys())
    labeled_list = [labeled_dict[tp] for tp in timepoints]
    tracks_list = list(tracks_list)

    linked_ids = [set() for _ in range(len(timepoints))]
    new_ids = [set() for _ in range(len(timepoints))]
    disappeared_ids = [set() for _ in range(len(timepoints))]
    isolated_ids = [set() for _ in range(len(timepoints))]

    for track in tracks_list:
        existing_indices = [
            i for i, comp_id in enumerate(track)
            if comp_id is not None
        ]

        if len(existing_indices) == 0:
            continue

        first_idx = existing_indices[0]
        last_idx = existing_indices[-1]


        if len(existing_indices) == 1:
            i = existing_indices[0]
            comp_id = track[i]

            if i == 0 and len(timepoints) > 1:
                disappeared_ids[i].add(comp_id)
            elif i == len(timepoints) - 1 and len(timepoints) > 1:
                new_ids[i].add(comp_id)
            else:
                isolated_ids[i].add(comp_id)

            continue

        for i in existing_indices:
            comp_id = track[i]

            has_prev = i > 0 and track[i - 1] is not None
            has_next = i < len(track) - 1 and track[i + 1] is not None

            if i == first_idx and first_idx > 0:
                new_ids[i].add(comp_id)

            elif i == last_idx and last_idx < len(timepoints) - 1:
                disappeared_ids[i].add(comp_id)

            elif has_prev or has_next:
                linked_ids[i].add(comp_id)

            else:
                isolated_ids[i].add(comp_id)

    max_slice = min(arr.shape[0] for arr in labeled_list) - 1
    n_cols = len(timepoints)

    def fn(SLICE: int):
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
        if n_cols == 1:
            axes = [axes]

        for idx, (ax, arr) in enumerate(zip(axes, labeled_list)):
            sl = arr[SLICE, :, :]
            rgb = np.zeros((*sl.shape, 3), dtype=float)

            comp_ids = np.unique(sl)
            comp_ids = comp_ids[comp_ids != 0]

            for comp_id in comp_ids:
                mask = sl == comp_id

                if comp_id in new_ids[idx]:
                    rgb[mask] = [1.0, 0.5, 0.0]   # orange
                elif comp_id in disappeared_ids[idx]:
                    rgb[mask] = [1.0, 0.0, 0.0]   # red
                elif comp_id in linked_ids[idx]:
                    rgb[mask] = [0.0, 1.0, 0.0]   # green
                elif comp_id in isolated_ids[idx]:
                    rgb[mask] = [0.7, 0.0, 0.7]   # purple
                else:
                    rgb[mask] = [0.5, 0.5, 0.5]   # gray

            ax.imshow(rgb)
            ax.set_title(f"{timepoints[idx]} | slice {SLICE}")
            ax.axis("off")

        patches = [
            mpatches.Patch(color='green', label='Linked to neighbour'),
            mpatches.Patch(color='orange', label='New lesion'),
            mpatches.Patch(color='red', label='Disappeared lesion'),
            mpatches.Patch(color='purple', label='Isolated lesion'),
            mpatches.Patch(color='gray', label='Other'),
        ]

        fig.legend(handles=patches, loc='upper right')
        plt.tight_layout()
        plt.show()

    slider = IntSlider(
        value=max_slice // 2,
        min=0,
        max=max_slice,
        description="SLICE",
    )

    interact(fn, SLICE=slider)


def visualize_track_overlap(labeled_dict, tracks):
    '''
    Interactively shows the overlap of one track between neighbor timepoints.
    - green: the lesion area at the first timepoint;
    - red: the lesion area at the second timepoint;
    - yellow: the intersection of the two masks.
    '''
    timepoints = list(labeled_dict.keys())
    labeled_list = [labeled_dict[tp] for tp in timepoints]
    tracks_list = list(tracks.values()) if isinstance(tracks, dict) else list(tracks)

    max_slice = min(arr.shape[0] for arr in labeled_list) - 1
    n_pairs = len(timepoints) - 1

    if len(tracks_list) == 0:
        raise ValueError("No tracks to visualize")

    def fn(TRACK: int, SLICE: int):
        track = tracks_list[TRACK]

        lesion_path = " -> ".join(
            "None" if comp_id is None else f"{tp}_{int(comp_id)}"
            for tp, comp_id in zip(timepoints, track)
        )

        fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 6))
        if n_pairs == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            tp_a = timepoints[i]
            tp_b = timepoints[i + 1]

            comp_a = track[i]
            comp_b = track[i + 1]

            arr_a = labeled_dict[tp_a]
            arr_b = labeled_dict[tp_b]

            shape_2d = arr_a[SLICE, :, :].shape

            if comp_a is None:
                mask_a = np.zeros(shape_2d, dtype=bool)
            else:
                mask_a = arr_a[SLICE, :, :] == comp_a

            if comp_b is None:
                mask_b = np.zeros(shape_2d, dtype=bool)
            else:
                mask_b = arr_b[SLICE, :, :] == comp_b

            overlap = mask_a & mask_b
            only_a = mask_a & ~mask_b
            only_b = mask_b & ~mask_a

            rgb = np.zeros((*shape_2d, 3), dtype=float)
            rgb[only_a] = [0.0, 1.0, 0.0]
            rgb[only_b] = [1.0, 0.0, 0.0]
            rgb[overlap] = [1.0, 1.0, 0.0]

            label_a = "None" if comp_a is None else f"{tp_a}_{int(comp_a)}"
            label_b = "None" if comp_b is None else f"{tp_b}_{int(comp_b)}"

            ax.imshow(rgb)
            ax.set_title(f"{label_a} vs {label_b}\nslice {SLICE}")
            ax.axis("off")

        patches = [
            mpatches.Patch(color='green', label='Lesion in first timepoint'),
            mpatches.Patch(color='red', label='Lesion in second timepoint'),
            mpatches.Patch(color='yellow', label='Overlap'),
        ]
        fig.legend(handles=patches, loc='upper right')
        fig.suptitle(f"Track #{TRACK}: {lesion_path}", fontsize=14)
        plt.tight_layout()
        plt.show()

    interact(
        fn,
        TRACK=IntSlider(value=0, min=0, max=len(tracks_list) - 1, description="TRACK"),
        SLICE=IntSlider(value=max_slice // 2, min=0, max=max_slice, description="SLICE"),
    )


