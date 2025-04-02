import pdb
import argparse
import os
import shutil
from glob import glob
import json
import pandas as pd
import numpy as np
import open3d as o3d
import yaml

#truncated track, occlusion

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test", default="test_dataset", help="Directory of testing dataset"
    )
    parser.add_argument(
        "--train", default="train_dataset", help="Directory of training dataset"
    )
    parser.add_argument("--gt", default=True, help="Is converting gt or prediction")
    parser.add_argument(
        "-o", "--outdir", default="jrdb", help="Directory of KITTI output"
    )
    opt = parser.parse_args()
    return opt


def get_train_test_names():
    train = [
        "bytes-cafe-2019-02-07_0",
        "clark-center-2019-02-28_0",
        "clark-center-2019-02-28_1",
        "clark-center-intersection-2019-02-28_0",
        "cubberly-auditorium-2019-04-22_0",
        "forbes-cafe-2019-01-22_0",
        "gates-159-group-meeting-2019-04-03_0",
        "gates-ai-lab-2019-02-08_0",
        "gates-basement-elevators-2019-01-17_1",
        "gates-to-clark-2019-02-28_1",
        "hewlett-packard-intersection-2019-01-24_0",
        "huang-2-2019-01-25_0",
        "huang-basement-2019-01-25_0",
        "huang-lane-2019-02-12_0",
        "jordan-hall-2019-04-22_0",
        "memorial-court-2019-03-16_0",
        "meyer-green-2019-03-16_0",
        "nvidia-aud-2019-04-18_0",
        "packard-poster-session-2019-03-20_0",
        "packard-poster-session-2019-03-20_1",
        "packard-poster-session-2019-03-20_2",
        "stlc-111-2019-04-19_0",
        "svl-meeting-gates-2-2019-04-08_0",
        "svl-meeting-gates-2-2019-04-08_1",
        "tressider-2019-03-16_0",
        "tressider-2019-03-16_1",
        "tressider-2019-04-26_2",
    ]
    test = [
        "cubberly-auditorium-2019-04-22_1",
        "discovery-walk-2019-02-28_0",
        "discovery-walk-2019-02-28_1",
        "food-trucks-2019-02-12_0",
        "gates-ai-lab-2019-04-17_0",
        "gates-basement-elevators-2019-01-17_0",
        "gates-foyer-2019-01-17_0",
        "gates-to-clark-2019-02-28_0",
        "hewlett-class-2019-01-23_0",
        "hewlett-class-2019-01-23_1",
        "huang-2-2019-01-25_1",
        "huang-intersection-2019-01-22_0",
        "indoor-coupa-cafe-2019-02-06_0",
        "lomita-serra-intersection-2019-01-30_0",
        "meyer-green-2019-03-16_1",
        "nvidia-aud-2019-01-25_0",
        "nvidia-aud-2019-04-18_1",
        "nvidia-aud-2019-04-18_2",
        "outdoor-coupa-cafe-2019-02-06_0",
        "quarry-road-2019-02-28_0",
        "serra-street-2019-01-30_0",
        "stlc-111-2019-04-19_1",
        "stlc-111-2019-04-19_2",
        "tressider-2019-03-16_2",
        "tressider-2019-04-26_0",
        "tressider-2019-04-26_1",
        "tressider-2019-04-26_3",
    ]
    return train, test


def create_base_dirs(opt):
    os.makedirs(os.path.join(opt.outdir, "sequences"), exist_ok=True, mode=0o777)
    os.makedirs(os.path.join(opt.outdir, "test_sequences"), exist_ok=True, mode=0o777)


def get_sequence_names(folder):
    path_to_seqs = os.path.join(folder, "images/image_0/*")
    return sorted(glob(path_to_seqs))


def copy_images_and_pointclouds(opt):
    # create dataset directories
    create_base_dirs(opt)
    camera_names = [
        "image_0",
        "image_2",
        "image_4",
        "image_6",
        "image_8",
        "image_stitched",
    ]
    train_seqs, test_seqs = get_train_test_names()
    sequence_folders_mot = ["sequences", "test_sequences"]
    sequence_paths_all = [get_sequence_names(opt.train), get_sequence_names(opt.test)]
    indirs = [opt.train, opt.test]
    global_config = os.path.join(opt.train, "calib", "defaults.yaml")
    with open(global_config) as f:
        global_config_dict = yaml.safe_load(f)
    print("Total number of train sequences: %d" % len(sequence_paths_all[0]))
    print("Total number of test sequences: %d" % len(sequence_paths_all[1]))
    # loop over data and save images
    for seq_folder, sequence_paths, indir in zip(
        sequence_folders_mot, sequence_paths_all, indirs
    ):
        for i, seq_path in enumerate(sequence_paths):
            seq_name = seq_path.split("/")[-1]
            if seq_name not in train_seqs and seq_name not in test_seqs:
                print("%s not in train or test" % seq_name)
                continue
            # copy images
            path_to_imgs = os.path.join(("/").join(seq_path.split("/")[:-1]), seq_name)
            for cam in camera_names:
                if cam == "image_stitched":
                    folder_name = os.path.join(seq_name, "imgs")
                else:
                    folder_name = os.path.join(seq_name, cam)
                folder_path_imgs = os.path.join(opt.outdir, seq_folder, folder_name)
                img_path_in = path_to_imgs.replace("image_0", cam)
                img_paths = glob(os.path.join(img_path_in, "*.jpg"))
                os.makedirs(folder_path_imgs, exist_ok=True, mode=0o777)
                for img_src in img_paths:
                    img_dst = os.path.join(folder_path_imgs, img_src.split("/")[-1])
                    shutil.copyfile(img_src, img_dst)
            # copy pointcloads
            folder_path_pcs = os.path.join(opt.outdir, seq_folder, seq_name, "depth")
            os.makedirs(folder_path_pcs, exist_ok=True, mode=0o777)
            path_to_pcs_lower = os.path.join(
                indir, "pointclouds/lower_velodyne", seq_name
            )
            lower_pc_paths = sorted(glob(os.path.join(path_to_pcs_lower, "*.pcd")))
            path_to_pcs_upper = os.path.join(
                indir, "pointclouds/upper_velodyne", seq_name
            )
            upper_pc_paths = sorted(glob(os.path.join(path_to_pcs_upper, "*.pcd")))
            for pc_src_lower, pc_src_upper in zip(lower_pc_paths, upper_pc_paths):
                pc_dst = os.path.join(folder_path_pcs, pc_src_lower.split("/")[-1])
                pc_dst = os.path.splitext(pc_dst)[0] + ".bin"
                upper_pc = o3d.io.read_point_cloud(pc_src_upper)
                lower_pc = o3d.io.read_point_cloud(pc_src_lower)
                upper_pc = np.asarray(upper_pc.points)
                lower_pc = np.asarray(lower_pc.points)
                upper_pc -= np.expand_dims(
                    np.array(
                        global_config_dict["calibrated"]["lidar_upper_to_rgb"][
                            "translation"
                        ]
                    ),
                    0,
                )
                upper_theta = float(
                    global_config_dict["calibrated"]["lidar_upper_to_rgb"]["rotation"][
                        -1
                    ]
                )
                upper_rotation_matrix = np.array(
                    [
                        [np.cos(upper_theta), -np.sin(upper_theta)],
                        [np.sin(upper_theta), np.cos(upper_theta)],
                    ]
                )
                upper_pc[:, :2] = np.squeeze(
                    np.matmul(upper_rotation_matrix, np.expand_dims(upper_pc[:, :2], 2))
                )
                lower_pc -= np.expand_dims(
                    np.array(
                        global_config_dict["calibrated"]["lidar_lower_to_rgb"][
                            "translation"
                        ]
                    ),
                    0,
                )
                lower_theta = float(
                    global_config_dict["calibrated"]["lidar_lower_to_rgb"]["rotation"][
                        -1
                    ]
                )
                lower_rotation_matrix = np.array(
                    [
                        [np.cos(lower_theta), -np.sin(lower_theta)],
                        [np.sin(lower_theta), np.cos(lower_theta)],
                    ]
                )
                lower_pc[:, :2] = np.squeeze(
                    np.matmul(lower_rotation_matrix, np.expand_dims(lower_pc[:, :2], 2))
                )
                pointcloud = np.vstack([upper_pc, lower_pc])
                pointcloud = np.hstack([pointcloud, np.ones((pointcloud.shape[0], 1))])
                with open(pc_dst, 'w') as f:
                    pointcloud.astype(np.float32).tofile(f)
            print(
                "Completed copying images and pointclouds for sequence [%d/%d] in %s"
                % (i + 1, len(sequence_paths), indir)
            )


def convert_2d_gt(opt, train=True):
    train_seqs, test_seqs = get_train_test_names()
    # load the files in the directory
    if train:
        files = glob(os.path.join(opt.train, "labels", "labels_2d", "*.json"))
        seq_folder = "sequences"
        indir = opt.train
    else:
        files = glob(os.path.join(opt.test, "labels", "labels_2d", "*.json"))
        seq_folder = "test_sequences"
        indir = opt.test
    data_columns = [
        "frame",
        "id",
        "bb_left",
        "bb_top",
        "bb_width",
        "bb_height",
        "x",
        "y",
        "z",
        "l",
        "h",
        "w",
        "theta",
        "conf",
    ]
    # loop over each file
    for i, fname in enumerate(sorted(files)):
        data = []
        video = ("_".join(fname.split("_")[:-1])).split("/")[-1]
        cam = (fname.split("_")[-1]).split(".")[0]
        cam = cam[:-1] + "_" + cam[-1]
        if video not in train_seqs and video not in test_seqs:
            print("%s not in train or test" % video)
            continue
        # actually load a single file
        with open(fname) as json_file:
            orig_labels = json.load(json_file)
        # loop over each frame
        for frame_name in sorted(orig_labels["labels"].keys()):
            frame_info = orig_labels["labels"][frame_name]
            # loop over targets
            for target in frame_info:
                id_num = int(target["label_id"].split(":")[-1])
                box_attrs = target["box"]
                frame_num = int(frame_name.split(".")[0])
                conf = 1
                occlusion = target["attributes"]["occlusion"]
                if occlusion == "Fully_occluded" or occlusion == "Severely_occluded":
                    occlusion = 1
                else:
                    occlusion = 0
                data.append(
                    [
                        int(frame_num),
                        int(id_num),
                        box_attrs[0],
                        box_attrs[1],
                        box_attrs[2],
                        box_attrs[3],
                        int(-1),
                        int(-1),
                        int(-1),
                        int(-1),
                        int(-1),
                        int(-1),
                        int(-1),
                        occlusion,
                    ]
                )
        df = pd.DataFrame(data=np.array(data), columns=data_columns)
        df["frame"] = df["frame"].astype(int)
        df["id"] = df["id"].astype(int)
        df["conf"] = df["conf"].astype(int)
        df["x"] = df["x"].astype(int)
        df["y"] = df["y"].astype(int)
        df["z"] = df["z"].astype(int)
        save_path = os.path.join(opt.outdir, seq_folder, video, "gt_" + cam)
        save_name = os.path.join(save_path, "gt.txt")
        os.makedirs(save_path, exist_ok=True)
        df.to_csv(save_name, index=False, header=False)
        print("Saved 2d gt [%d/%d] in %s" % (i + 1, len(files), indir))


def convert_2d_det(opt, train=True):
    train_seqs, test_seqs = get_train_test_names()
    # load the files in the directory
    if train:
        files = glob(os.path.join(opt.train, "detections", "detections_2d", "*.json"))
        seq_folder = "sequences"
        indir = opt.train
    else:
        files = glob(os.path.join(opt.test, "detections", "detections_2d", "*.json"))
        seq_folder = "test_sequences"
        indir = opt.test
    data_columns = [
        "frame",
        "id",
        "bb_left",
        "bb_top",
        "bb_width",
        "bb_height",
        "x",
        "y",
        "z",
        "l",
        "h",
        "w",
        "theta",
        "conf",
    ]
    # loop over each file
    for i, fname in enumerate(sorted(files)):
        data = []
        video = ("_".join(fname.split("_")[:-1])).split("/")[-1]
        cam = (fname.split("_")[-1]).split(".")[0]
        cam = cam[:-1] + "_" + cam[-1]
        if video not in train_seqs and video not in test_seqs:
            print("%s not in train or test" % video)
            continue
        # actually load a single file
        with open(fname) as json_file:
            orig_labels = json.load(json_file)
        # loop over each frame
        for frame_name in sorted(orig_labels["detections"].keys()):
            frame_info = orig_labels["detections"][frame_name]
            # loop over targets
            for target in frame_info:
                id_num = int(target["label_id"].split(":")[-1])
                box_attrs = target["box"]
                frame_num = int(frame_name.split(".")[0])
                conf = target["score"]
                data.append(
                    [
                        int(frame_num),
                        int(id_num),
                        box_attrs[0],
                        box_attrs[1],
                        box_attrs[2],
                        box_attrs[3],
                        int(-1),
                        int(-1),
                        int(-1),
                        int(-1),
                        int(-1),
                        int(-1),
                        int(-1),
                        int(conf),
                    ]
                )
        try:
            df = pd.DataFrame(data=np.array(data), columns=data_columns)
        except:
            save_path = os.path.join(opt.outdir, seq_folder, video, "det_" + cam)
            save_name = os.path.join(save_path, "det.txt")
            os.makedirs(save_path, exist_ok=True)
            df.to_csv(save_name, index=False, header=False)
            continue
        df["frame"] = df["frame"].astype(int)
        df["id"] = df["id"].astype(int)
        df["conf"] = df["conf"].astype(float)
        df["x"] = df["x"].astype(int)
        df["y"] = df["y"].astype(int)
        df["z"] = df["z"].astype(int)
        save_path = os.path.join(opt.outdir, seq_folder, video, "det_" + cam)
        save_name = os.path.join(save_path, "det.txt")
        os.makedirs(save_path, exist_ok=True)
        df.to_csv(save_name, index=False, header=False)
        print("Saved 2d det [%d/%d] in %s" % (i + 1, len(files), indir))


def convert_stitched_gt(opt, train=True):
    train_seqs, test_seqs = get_train_test_names()

    opt.train = "/pvol2/jrdb_dev/jrdb_website_dev/static/downloads/jrdb_train/train_dataset_with_activity/"
    print(opt)
    # load the files in the directory
    if train:
        files = glob(os.path.join(opt.train, "labels", "labels_2d_stitched", "*.json"))
        seq_folder = "sequences"
        indir = opt.train
    else:
        files = glob(os.path.join(opt.test, "labels", "labels_2d_stitched", "*.json"))
        seq_folder = "test_sequences"
        indir = opt.test
    data_columns = [
        "frame",
        "id",
        "bb_left",
        "bb_top",
        "bb_width",
        "bb_height",
        "x",
        "y",
        "z",
        "l",
        "h",
        "w",
        "theta",
        "conf",
    ]
    # loop over each file
    for i, fname in enumerate(sorted(files)):
        data = []
        video = ("_".join(fname.split("_")[:-1])).split("/")[-1]
        video = fname.split("/")[-1][:-5]
        if video not in train_seqs and video not in test_seqs:
            print("%s not in train or test" % video)
            continue
        # actually load a single file
        with open(fname) as json_file:
            orig_labels = json.load(json_file)
        # loop over each frame
        for frame_name in sorted(orig_labels["labels"].keys()):
            frame_info = orig_labels["labels"][frame_name]
            # loop over targets
            for target in frame_info:
                id_num = int(target["label_id"].split(":")[-1])
                box_attrs = target["box"]
                frame_num = int(frame_name.split(".")[0])
                conf = 1
                occlusion = target["attributes"]["occlusion"]
                if occlusion == "Fully_occluded" or occlusion == "Severely_occluded":
                    occlusion = 1
                else:
                    occlusion = 0
                data.append(
                    [
                        int(frame_num),
                        int(id_num),
                        box_attrs[0],
                        box_attrs[1],
                        box_attrs[2],
                        box_attrs[3],
                        int(-1),
                        int(-1),
                        int(-1),
                        int(-1),
                        int(-1),
                        int(-1),
                        int(-1),
                        occlusion,
                    ]
                )
        df = pd.DataFrame(data=np.array(data), columns=data_columns)
        df["frame"] = df["frame"].astype(int)
        df["id"] = df["id"].astype(int)
        df["conf"] = df["conf"].astype(int)
        df["x"] = df["x"].astype(int)
        df["y"] = df["y"].astype(int)
        df["z"] = df["z"].astype(int)
        save_path = os.path.join(opt.outdir, seq_folder, video, "gt")
        save_name = os.path.join(save_path, "gt.txt")
        os.makedirs(save_path, exist_ok=True)
        df.to_csv(save_name, index=False, header=False)
        print("Saved stitched gt [%d/%d]" % (i + 1, len(files)))


def convert_stitched_det(opt, train=True):
    train_seqs, test_seqs = get_train_test_names()
    # load the files in the directory
    if train:
        files = glob(
            os.path.join(opt.train, "detections", "detections_2d_stitched", "*.json")
        )
        seq_folder = "sequences"
        indir = opt.train
    else:
        files = glob(
            os.path.join(opt.test, "detections", "detections_2d_stitched", "*.json")
        )
        seq_folder = "test_sequences"
        indir = opt.test
    data_columns = [
        "frame",
        "id",
        "bb_left",
        "bb_top",
        "bb_width",
        "bb_height",
        "x",
        "y",
        "z",
        "l",
        "h",
        "w",
        "theta",
        "conf",
    ]
    # loop over each file
    for i, fname in enumerate(sorted(files)):
        data = []
        video = ("_".join(fname.split("_")[:-1])).split("/")[-1]
        video = fname.split("/")[-1][:-5]
        if video not in train_seqs and video not in test_seqs:
            print("%s not in train or test" % video)
            continue
        # actually load a single file
        with open(fname) as json_file:
            orig_labels = json.load(json_file)
        # loop over each frame
        for frame_name in sorted(orig_labels["detections"].keys()):
            frame_info = orig_labels["detections"][frame_name]
            # loop over targets
            for target in frame_info:
                id_num = int(target["label_id"].split(":")[-1])
                box_attrs = target["box"]
                frame_num = int(frame_name.split(".")[0])
                conf = target["score"]
                data.append(
                    [
                        int(frame_num),
                        int(id_num),
                        box_attrs[0],
                        box_attrs[1],
                        box_attrs[2],
                        box_attrs[3],
                        int(-1),
                        int(-1),
                        int(-1),
                        int(-1),
                        int(-1),
                        int(-1),
                        int(-1),
                        int(conf),
                    ]
                )
        try:
            df = pd.DataFrame(data=np.array(data), columns=data_columns)
        except:
            save_path = os.path.join(opt.outdir, seq_folder, video, "det")
            save_name = os.path.join(save_path, "det.txt")
            os.makedirs(save_path, exist_ok=True)
            df.to_csv(save_name, index=False, header=False)
            continue
        df["frame"] = df["frame"].astype(int)
        df["id"] = df["id"].astype(int)
        df["conf"] = df["conf"].astype(float)
        df["x"] = df["x"].astype(int)
        df["y"] = df["y"].astype(int)
        df["z"] = df["z"].astype(int)
        save_path = os.path.join(opt.outdir, seq_folder, video, "det_tmp")
        save_name = os.path.join(save_path, "det.txt")
        os.makedirs(save_path, exist_ok=True)
        df.to_csv(save_name, index=False, header=False)
        print("Saved stitched det [%d/%d]" % (i + 1, len(files)))


def convert_2d(opt):
    train_seqs, test_seqs = get_train_test_names()
    print(opt.train)
    files = glob(os.path.join(opt.train, "labels", "labels_2d_stitched", "*.json"))
    # loop over each file
    print(files)
    for i, fname in enumerate(sorted(files)):
        data = []
        video = fname.split("/")[-1][:-5]
        if video not in train_seqs and video not in test_seqs:
            print("%s not in train or test" % video)
            continue
        # actually load a single file
        with open(fname) as json_file:
            orig_labels = json.load(json_file)
        # loop over each frame
        for frame_name in sorted(orig_labels["labels"].keys()):
            frame_info = orig_labels["labels"][frame_name]
            for target in frame_info:
                id = int(target["label_id"].split(":")[-1])
                box = target["box"]
                # print(box)
                frame = int(frame_name.split(".")[0])

                rotation_y = -1
                x1_2d, y1_2d, x2_2d, y2_2d = box
                truncated = 0
                occlusion = 0
                alpha = -1
                if not opt.gt:
                    conf = target["score"]
                    line = (
                        "%s %s Pedestrian %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n"
                        % (
                            frame,
                            id,
                            truncated,
                            occlusion,
                            alpha,
                            x1_2d,
                            y1_2d,
                            x2_2d,
                            y2_2d,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            rotation_y,
                            conf,
                        )
                    )
                else:
                    occlusion = target["attributes"]["occlusion"]
                    line = (
                        "%s %s Pedestrian %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n"
                        % (
                            frame,
                            id,
                            truncated,
                            occlusion,
                            alpha,
                            x1_2d,
                            y1_2d,
                            x2_2d,
                            y2_2d,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            rotation_y,
                        )
                    )
                data.append(line)
        save_path = os.path.join(opt.outdir, "CIWT", "data",)
        save_name = os.path.join(save_path, "%04d" % i + ".txt")
        os.makedirs(save_path, exist_ok=True)
        # df.to_csv(save_name, index=False, header=False)
        with open(save_name, "w") as f:
            f.writelines(data)
        print(save_path)
        print("Saved 2d [%d/%d]" % (i + 1, len(files)))


def convert_3d(opt):
    train_seqs, test_seqs = get_train_test_names()
    files = glob(os.path.join(opt.train, "labels", "labels_3d", "*.json"))
    # loop over each file
    for i, fname in enumerate(sorted(files)):
        data = []
        video = fname.split("/")[-1][:-5]
        if video not in train_seqs and video not in test_seqs:
            print("%s not in train or test" % video)
            continue
        # actually load a single file
        with open(fname) as json_file:
            orig_labels = json.load(json_file)
        # loop over each frame
        for frame_name in sorted(orig_labels["labels"].keys()):
            frame_info = orig_labels["labels"][frame_name]
            for target in frame_info:
                id = int(target["label_id"].split(":")[-1])
                box = target["box"]
                # print(box)
                frame = int(frame_name.split(".")[0])
                conf = 1
                centerx_3d, centery_3d = (
                    box["cx"],
                    box["cy"],
                )
                truncated = 0
                occlusion = 0
                alpha = -1
                if not opt.gt:
                    conf = target["score"]
                    line = (
                        "%s %s Pedestrian %s %s %s %s %s %s %s %s %s %s\n"
                        % (
                            frame,
                            id,
                            truncated,
                            occlusion,
                            alpha,
                            -1,
                            -1,
                            -1,
                            -1,
                            centery_3d,
                            centerx_3d,
                            conf,
                        )
                    )
                else:
                    # occlusion = target["attributes"]["occlusion"]
                    line = (
                        "%s %s Pedestrian %s %s %s %s %s %s %s %s %s\n"
                        % (
                            frame,
                            id,
                            truncated,
                            occlusion,
                            alpha,
                            -1,
                            -1,
                            -1,
                            -1,
                            centerx_3d, # In KITTI box coordinates
                            centery_3d, # In KITTI box coordinates
                        )
                    )
                data.append(line)
        df = pd.DataFrame(data=np.array(data))
        save_path = os.path.join(opt.outdir, "CIWT", "data",)
        save_name = os.path.join(save_path, "%04d" % i + ".txt")
        os.makedirs(save_path, exist_ok=True)
        # df.to_csv(save_name, index=False, header=False)
        with open(save_name, "w") as f:
            f.writelines(data)
        print(save_path)
        print("Saved 3d [%d/%d]" % (i + 1, len(files)))


def copy_calib(opt, train=True):
    if train:
        sequences = os.listdir(os.path.join(opt.train, "images", "image_0"))
        seq_folder = "sequences"
    else:
        sequences = os.listdir(os.path.join(opt.test, "images", "image_0"))
        seq_folder = "test_sequences"
    for sequence in sequences:
        shutil.copytree(
            os.path.join(opt.train, "calib"),
            os.path.join(opt.outdir, seq_folder, sequence, "calib"),
        )


def main(opt):

    # copy_images_and_pointclouds(opt)
    # convert_2d_det(opt)
    # convert_2d_det(opt, train=False)
    # convert_stitched_det(opt)
    # convert_stitched_det(opt, train=False)
    # convert_2d_gt(opt)
    # convert_2d_gt(opt, train=False)
    # convert_stitched_gt(opt)
    # convert_stitched_gt(opt, train=True)
    # convert_3d_gt(opt)

    # please use this function for converting json file to txt format, others are keeped for further development.
    convert_3d(opt)

    # convert_2d(opt)
    # copy_calib(opt)
    # copy_calib(opt, train=False)


if __name__ == "__main__":
    opt = parse_arguments()
    print(opt)
    main(opt)
