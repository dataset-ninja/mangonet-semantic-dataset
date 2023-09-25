# https://github.com/avadesh02/MangoNet-Semantic-Dataset

import os
import shutil
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from cv2 import connectedComponents
from dataset_tools.convert import unpack_if_archive
from dotenv import load_dotenv
from supervisely.io.fs import get_file_name, get_file_size
from tqdm import tqdm

import src.settings as s


def fix_masks(image_np: np.ndarray) -> np.ndarray:
    lower_bound = np.array([70, 110, 0])
    upper_bound = np.array([255, 255, 255])
    condition_white = np.logical_and(
        np.all(image_np >= lower_bound, axis=2), np.all(image_np <= upper_bound, axis=2)
    )

    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([20, 20, 20])
    condition_black = np.logical_and(
        np.all(image_np >= lower_bound, axis=2), np.all(image_np <= upper_bound, axis=2)
    )

    image_np[np.where(condition_white)] = (255, 255, 255)
    image_np[np.where(condition_black)] = (0, 0, 0)

    return image_np

def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count
    
def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:

    # project_name = "MangoNet"
    dataset_path = "/home/grokhi/rawdata/MangoNet-Semantic-Dataset/MangoNet Dataset"
    ds_name = "ds"
    batch_size = 3
    images_folder_name = "original images"
    masks_folder_name = "annotated images"
    masks_ext = ".jpg"
    masks_prefix = "Class_"


    def create_ann(image_path):
        labels = []

        image_name = get_file_name(image_path)[5:]
        mask_name = masks_prefix + image_name + masks_ext
        mask_path = os.path.join(masks_pathes, mask_name)
        ann_np = sly.imaging.image.read(mask_path)[:, :, :]
        ann_np = fix_masks(ann_np)[:, :, 0]
        img_height = ann_np.shape[0]
        img_wight = ann_np.shape[1]
        mask = ann_np != 0
        ret, curr_mask = connectedComponents(mask.astype("uint8"), connectivity=8)
        for i in range(1, ret):
            obj_mask = curr_mask == i
            curr_bitmap = sly.Bitmap(obj_mask)
            if curr_bitmap.area > 100:
                curr_label = sly.Label(curr_bitmap, obj_class)
                labels.append(curr_label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels)


    obj_class = sly.ObjClass("mango", sly.Bitmap)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(obj_classes=[obj_class])
    api.project.update_meta(project.id, meta.to_json())

    for ds_name in os.listdir(dataset_path):
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        curr_ds_path = os.path.join(dataset_path, ds_name)

        images_pathes = os.path.join(curr_ds_path, images_folder_name)
        masks_pathes = os.path.join(curr_ds_path, masks_folder_name)
        images_names = os.listdir(images_pathes)

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for img_names_batch in sly.batched(images_names, batch_size=batch_size):
            images_pathes_batch = [
                os.path.join(images_pathes, image_path) for image_path in img_names_batch
            ]

            anns_batch = [create_ann(image_path) for image_path in images_pathes_batch]

            img_infos = api.image.upload_paths(dataset.id, img_names_batch, images_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            api.annotation.upload_anns(img_ids, anns_batch)

            progress.iters_done_report(len(img_names_batch))
    return project


