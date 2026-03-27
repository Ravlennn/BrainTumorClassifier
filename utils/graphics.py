from ipywidgets import interact, IntSlider  # type: ignore
import matplotlib.pyplot as plt
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
