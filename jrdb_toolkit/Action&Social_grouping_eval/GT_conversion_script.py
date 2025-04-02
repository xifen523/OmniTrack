import json
import os

#put the path to train-set annotation files here!
annot_path = './train_annot/'

FRAMES_NUM = {1: 1727, 2: 579, 3: 1440, 4: 1008, 5: 1297, 6: 1447, 7: 863, 8: 466, 9: 1010, 10: 725,
              11: 1729, 12: 873, 13: 1155, 14: 724, 15: 1523, 16: 1090, 17: 459, 18: 1013, 19: 717, 20: 852,
              21: 1372, 22: 1736, 23: 867, 24: 872, 25: 431, 26: 504, 27: 1441}

def read_labelmap(labelmap_file):
  labelmap = {}
  class_ids = set()
  name = ""
  # with open labelmap_file as labelmap_file:
  labelmap_file = open(labelmap_file)
  for line in labelmap_file:
    if line.startswith("  name:"):
      name = line.split('"')[1]
    elif line.startswith("  id:") or line.startswith("  label_id:"):
      class_id = int(line.strip().split(" ")[-1])
      labelmap[name] = class_id
      class_ids.add(class_id)
  return labelmap, class_ids

def JRDB_read_annotations(annot_path):
    with open(annot_path, 'r') as f:
        data = json.load(f)
    return data

def construct_GT():
    H = 480
    W = 3760
    dc = 1  #dont care!
    seqs = sorted(os.listdir(annot_path))
    labelmap, _ = read_labelmap('./label_map/task_1.pbtxt')
    for idx, seq in enumerate(seqs):
        data = JRDB_read_annotations(annot_path+'/'+seq)
        for image in sorted(data['labels'].keys()):  # in each image
            img = image.split('.')[0]
            if int(img)%15==0 and int(img)>=15 and int(img)<=FRAMES_NUM[idx + 1]:
                num_box = len(data['labels'][image])
                for i in range(num_box):  # for each box
                    x, y, w, h = data['labels'][image][i]['box']
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    no_eval = data['labels'][image][i]['attributes']['no_eval']
                    if x>=0 and x2<=W and y>=0 and y2<=H and w>0 and h>0 and not no_eval:
                        #social group
                        group_label = data['labels'][image][i]['social_group']['cluster_ID']
                        group_level = data['labels'][image][i]['social_group']['cluster_stat']
                        if group_level>2:
                            lvl = 1
                        else:
                            lvl = 0
                        GT_list = [idx, int(img), x1, y1, x2, y2, group_label, dc, lvl]
                        str_to_be_added = [str(k) for k in GT_list]
                        str_to_be_added = (" ".join(str_to_be_added))
                        f = open('./out/gt_group.txt', "a+")
                        f.write(str_to_be_added + "\r\n")
                        f.close()

                        #individual action
                        for act in data['labels'][image][i]['action_label'].keys():
                            if act in labelmap:
                                action_level = data['labels'][image][i]['action_label'][act]
                                if action_level > 2:
                                    lvl = 1
                                else:
                                    lvl=0
                                GT_list = [idx, int(img), x1, y1, x2, y2, dc, labelmap[act], lvl]
                                str_to_be_added = [str(k) for k in GT_list]
                                str_to_be_added = (" ".join(str_to_be_added))
                                f = open('./out/gt_action.txt', "a+")
                                f.write(str_to_be_added + "\r\n")
                                f.close()

                        #social activity
                        for activ in data['labels'][image][i]['social_activity'].keys():
                            if activ in labelmap:
                                activity_level = max(data['labels'][image][i]['social_activity'][activ],
                                                     data['labels'][image][i]['social_group']['cluster_stat'])
                                if activity_level > 2:
                                    lvl = 1
                                else:
                                    lvl = 0
                                GT_list = [idx, int(img), x1, y1, x2, y2, dc, labelmap[activ], lvl]
                                str_to_be_added = [str(k) for k in GT_list]
                                str_to_be_added = (" ".join(str_to_be_added))
                                f = open('./out/gt_activity.txt', "a+")
                                f.write(str_to_be_added + "\r\n")
                                f.close()

construct_GT()
