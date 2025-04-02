import json
import os





if __name__ == '__main__':

    score_thr = 0.3
    json_file = '/home/lk/workspase/python/dection/Sparse4D/val/work_dirs/JRDB_sparse4dv3_temporal_r50_1x8_bs6_480x768_local/Tue_Jul_30_11_13_51_2024/results_jrdb.json'

    # mkdir for submission path
    submission_path = os.path.join(os.path.dirname(json_file), 'jrdb')
    if not os.path.exists(submission_path):
        os.makedirs(submission_path)

    # load json file
    with open(json_file, 'r') as f:
        results = json.load(f)

    # format submission file
    for k, v in results['results'].items():
        seq_name, frame_id = k.rsplit('_',1)
        # mkdir for sequence
        seq_path = os.path.join(submission_path, seq_name)
        if not os.path.exists(seq_path):
            os.makedirs(seq_path)
        # write submission file
        submission_file = os.path.join(seq_path, frame_id + '.txt')
        with open(submission_file, 'w') as f:
            for obj in v:
                # type(str,1) truncated(-1) occluded(-1) alpha(-1) bbox(float,4) dimensions(float,3) location(float,3) rotation_y(float,1) score(float,1) 
                if obj['detection_score'] < score_thr:
                    continue
                f.write(obj['detection_name'] + ' -1 -1 -1 ' + '-1 -1 -1 -1 ' + ' '.join(map(lambda x: f'{x:.4f}', obj['size'])) + ' ' + ' '.join(map(lambda x: f'{x:.4f}', obj['center'])) + ' ' + f'{obj["yaw"]:.4f}' + ' ' + f'{obj["detection_score"]:.4f}' + '\n')
