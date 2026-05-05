import pandas as pd
import numpy as np


def initialization_dict(base_path):
    '''
    Creates a dictionary for further work
    '''
    
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