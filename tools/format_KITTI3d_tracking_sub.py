import os
import json
import tqdm



if __name__=="__main__":

    score_thr = 0.2
    path = r'data/JRDB/train/train_images/images/image_0'
    list_dir = os.listdir(path)
    list_dir.sort()
    map_dict = {list_dir[i]:i for i in range(len(list_dir))}
    

    json_file = r'results/submission/results_jrdb.json'
    # mkdir for submission path
    submission_path = os.path.join(os.path.dirname(json_file), 'sparse4D/CIWT/data')
    if not os.path.exists(submission_path):
        os.makedirs(submission_path)
    
    # load json file
    with open(json_file, 'r') as f:
        results = json.load(f)

    # format submission file
    for k, v in tqdm.tqdm(results['results'].items()):
        seq_name, frame_id = k.rsplit('_',1)
        # write submission file
        submission_file = os.path.join(submission_path, f'{map_dict[seq_name]:04d}' + '.txt')
        with open(submission_file, 'a') as f:
            for obj in v:
                # frame, track id, type, truncated,occluded, alpha, bb_left, bb_top, bb_width, bb_height, x, y, z, height, width, length, rotation_y, score
                if obj['detection_score'] < score_thr or obj['cls_score'] < score_thr:
                    continue
                line = f'{int(str(frame_id))} {obj["tracking_id"]} Pedestrian 0 0 -1 -1 -1 -1 -1 {obj["size"][0]:.4f} {obj["size"][1]:.4f} {obj["size"][2]:.4f} {obj["center"][0]:.4f} {obj["center"][1]:.4f} {obj["center"][2]:.4f} {obj["yaw"]:.4f} {obj["detection_score"]:.2f}\n'
                f.write(line)
                

    # zip CIWT folder
    sparse4D_path = os.path.join(os.path.dirname(json_file), 'sparse4D')
    
    os.system(f'cd {sparse4D_path} && zip -r sparse4D.zip ./*')
