import numpy as np
from sklearn.cluster import KMeans
import mmcv
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from projects.mmdet3d_plugin.core.box2d import *


def get_kmeans_anchor(
    ann_file,
    num_anchor=900,
    detection_range=55,
    output_file_name="nuscenes_kmeans900.npy",
    verbose=False,
):
    data = mmcv.load(ann_file, file_format="pkl")
    gt_boxes = np.concatenate([x["gt_boxes"] for x in data["infos"]], axis=0)
    # distance = np.linalg.norm(gt_boxes[:, :2], axis=-1, ord=2)
    # mask = distance <= detection_range
    # gt_boxes = gt_boxes[mask]
    clf = KMeans(n_clusters=num_anchor, verbose=verbose)
    print("===========Starting kmeans, please wait.===========")
    clf.fit(gt_boxes[:, [X_2D, Y_2D]])
    anchor = np.zeros((num_anchor, 4))
    anchor[:, [X_2D, Y_2D]] = clf.cluster_centers_
    anchor[:, [W_2D, H_2D]] = np.log(gt_boxes[:, [W_2D, H_2D]].mean(axis=0))
    save_root = os.path.dirname(ann_file)
    output_file_name = os.path.join(save_root, output_file_name)
    np.save(output_file_name, anchor)
    print(f"===========Done! Save results to {output_file_name}.===========")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="anchor kmeans")
    parser.add_argument("--ann_file", type=str, required=True)
    parser.add_argument("--num_anchor", type=int, default=900)
    parser.add_argument("--detection_range", type=float, default=55)
    parser.add_argument(
        "--output_file_name", type=str, default="kmeans900.npy"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    get_kmeans_anchor(
        args.ann_file,
        args.num_anchor,
        args.detection_range,
        args.output_file_name,
        args.verbose,
    )
