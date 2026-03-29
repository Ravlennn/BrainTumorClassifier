from ipywidgets import interact, IntSlider  # type: ignore
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import SimpleITK as sitk  # type: ignore


def explore_3D_array(arr: np.ndarray, cmap: str = "gray"):
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
    timepoints = list(labeled_dict.keys())
    labeled_list = [labeled_dict[tp] for tp in timepoints]
    tracks_list = list(tracks_list)

    matched_ids = [set() for _ in range(len(timepoints))]
    for track in tracks_list:
        for tp_idx, comp_id in enumerate(track):
            if comp_id is not None:
                matched_ids[tp_idx].add(comp_id)

    max_slice = min(arr.shape[0] for arr in labeled_list) - 1
    n_cols = len(timepoints)

    def fn(SLICE: int):
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
        if n_cols == 1:
            axes = [axes]

        for idx, (ax, arr) in enumerate(zip(axes, labeled_list)):
            sl = arr[SLICE, :, :]
            rgb = np.zeros((*sl.shape, 3))

            for comp_id in range(1, arr.max() + 1):
                mask = (sl == comp_id)
                if comp_id in matched_ids[idx]:
                    rgb[mask] = [0, 1, 0]
                else:
                    rgb[mask] = [1, 0, 0]

            ax.imshow(rgb)
            ax.set_title(f"{timepoints[idx]} | slice {SLICE}")
            ax.axis("off")

        patches = [
            mpatches.Patch(color='green', label='Matched'),
            mpatches.Patch(color='red',    label='No Matched'),
        ]
        fig.legend(handles=patches, loc='upper right')
        plt.tight_layout()
        plt.show()

    slider = IntSlider(value=max_slice // 2, min=0,
                       max=max_slice, description="SLICE")
    interact(fn, SLICE=slider)
