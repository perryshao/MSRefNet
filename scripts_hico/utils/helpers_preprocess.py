# coding=utf-8
import time
import json
import numpy as np
import argparse
from random import randint
import cv2
from sklearn.preprocessing import LabelBinarizer

with open('../infos/directory.json') as fp: all_data_dir = json.load(fp)

ANN_FILE_train = all_data_dir + 'Annotations_hico/train_annotations.json'
ANN_FILE_test = all_data_dir + 'Annotations_hico/test_annotations.json'
with open(ANN_FILE_train) as fp:
    ANNOTATIONS_train = json.load(fp)
with open(ANN_FILE_test) as fp:
    ANNOTATIONS_test = json.load(fp)

OBJ_PATH_train_s = all_data_dir + 'Object_Detections_hico/train/'
OBJ_PATH_test_s = all_data_dir + 'Object_Detections_hico/test/'

PER_PATH_train_s = '/home/hzy/VSG_bp/VSGNet-master/All_data/KP_Detections_hico/train/'
PER_PATH_test_s = '/home/hzy/VSG_bp/VSGNet-master/All_data/KP_Detections_hico/test/'

with open(all_data_dir + 'hico_infos/hico_list_obj.json') as fp:
    list_obj = json.load(fp)
obj_list = dict([(value, int(key) - 1) for key, value in list_obj.items()])

VERB2ID = {u'adjust': 0,
           u'assemble': 1,
           u'block': 2,
           u'blow': 3,
           u'board': 4,
           u'break': 5,
           u'brush_with': 6,
           u'buy': 7,
           u'carry': 8,
           u'catch': 9,
           u'chase': 10,
           u'check': 11,
           u'clean': 12,
           u'control': 13,
           u'cook': 14,
           u'cut': 15,
           u'cut_with': 16,
           u'direct': 17,
           u'drag': 18,
           u'dribble': 19,
           u'drink_with': 20,
           u'drive': 21,
           u'dry': 22,
           u'eat': 23,
           u'eat_at': 24,
           u'exit': 25,
           u'feed': 26,
           u'fill': 27,
           u'flip': 28,
           u'flush': 29,
           u'fly': 30,
           u'greet': 31,
           u'grind': 32,
           u'groom': 33,
           u'herd': 34,
           u'hit': 35,
           u'hold': 36,
           u'hop_on': 37,
           u'hose': 38,
           u'hug': 39,
           u'hunt': 40,
           u'inspect': 41,
           u'install': 42,
           u'jump': 43,
           u'kick': 44,
           u'kiss': 45,
           u'lasso': 46,
           u'launch': 47,
           u'lick': 48,
           u'lie_on': 49,
           u'lift': 50,
           u'light': 51,
           u'load': 52,
           u'lose': 53,
           u'make': 54,
           u'milk': 55,
           u'move': 56,
           u'no_interaction': 57,
           u'open': 58,
           u'operate': 59,
           u'pack': 60,
           u'paint': 61,
           u'park': 62,
           u'pay': 63,
           u'peel': 64,
           u'pet': 65,
           u'pick': 66,
           u'pick_up': 67,
           u'point': 68,
           u'pour': 69,
           u'pull': 70,
           u'push': 71,
           u'race': 72,
           u'read': 73,
           u'release': 74,
           u'repair': 75,
           u'ride': 76,
           u'row': 77,
           u'run': 78,
           u'sail': 79,
           u'scratch': 80,
           u'serve': 81,
           u'set': 82,
           u'shear': 83,
           u'sign': 84,
           u'sip': 85,
           u'sit_at': 86,
           u'sit_on': 87,
           u'slide': 88,
           u'smell': 89,
           u'spin': 90,
           u'squeeze': 91,
           u'stab': 92,
           u'stand_on': 93,
           u'stand_under': 94,
           u'stick': 95,
           u'stir': 96,
           u'stop_at': 97,
           u'straddle': 98,
           u'swing': 99,
           u'tag': 100,
           u'talk_on': 101,
           u'teach': 102,
           u'text_on': 103,
           u'throw': 104,
           u'tie': 105,
           u'toast': 106,
           u'train': 107,
           u'turn': 108,
           u'type_on': 109,
           u'walk': 110,
           u'wash': 111,
           u'watch': 112,
           u'wave': 113,
           u'wear': 114,
           u'wield': 115,
           u'zip': 116}

encoder = LabelBinarizer()

object_class = ['0airplane', '1apple', '2backpack', '3banana', '4baseball_bat', '5baseball_glove', '6bear', '7bed', '8bench',
                '9bicycle', '10bird', '11boat', '12book', '13bottle', '14bowl', '15broccoli', '16bus', '17cake',
                '18car', '19carrot', '20cat', '21cell_phone', '22chair', '23clock', '24couch', '25cow',
                '26cup', '27dining_table', '28dog', '29donut', '30elephant', '31fire_hydrant', '32fork',
                '33frisbee', '34giraffe', '35hair_drier', '36handbag', '37horse', '38hot_dog', '39keyboard',
                '40kite', '41knife', '42laptop', '43microwave', '44motorcycle', '45mouse', '46orange', '47oven',
                '48parking_meter', '49person', '50pizza', '51potted_plant', '52refrigerator',
                '53remote', '54sandwich', '55scissors', '56sheep', '57sink', '58skateboard', '59skis',
                '60snowboard', '61spoon', '62sports_ball', '63stop_sign', '64suitcase', '65surfboard', '66teddy_bear',
                '67tennis_racket', '68tie', '69toaster', '70toilet', '71toothbrush', '72traffic_light',
                '73train', '74truck', '75tv', '76umbrella', '77vase', '78wine_glass', '79zebra']

one_hot = encoder.fit_transform(object_class)

MATCHING_IOU = .5
NO_VERBS = 117


def get_detections(segment_key, flag):
    if flag == 'train':
        key_ann = '%.8i' % (segment_key)
        annotation = ANNOTATIONS_train[key_ann]
        cur_obj_path_s = OBJ_PATH_train_s + "HICO_train2015_%.8i.json" % (segment_key)
        cur_per_path_s = PER_PATH_train_s + "HICO_train2015_%.8i.json" % (segment_key)
        SCORE_TH = 0.6
        SCORE_OBJ = 0.3

        select_threshold = 15
    elif flag == 'test':
        key_ann = '%.8i' % (segment_key)
        annotation = ANNOTATIONS_test[key_ann]
        cur_obj_path_s = OBJ_PATH_test_s + "HICO_test2015_%.8i.json" % (segment_key)
        cur_per_path_s = PER_PATH_test_s + "HICO_test2015_%.8i.json" % (segment_key)
        SCORE_TH = 0.6
        SCORE_OBJ = 0.3
        select_threshold = 15

    annotation = clean_up_annotation(annotation)
    with open(cur_obj_path_s) as fp:
        detections = json.load(fp)
    with open(cur_per_path_s) as ff:
        detections_per = json.load(ff)

    img_H = detections['H']
    img_W = detections['W']
    shape = [img_W, img_H]
    persons_d, objects_d = analyze_detections(detections, detections_per, SCORE_TH, SCORE_OBJ)
    d_p_boxes, scores_persons, class_id_humans, person_kp = get_person_det(persons_d)
    d_o_boxes, scores_objects, class_id_objects = get_boxes_det(objects_d, img_H, img_W)

    try:
        d_o_boxes = np.concatenate([d_o_boxes, d_p_boxes]).tolist()
    except:
        import pdb;
        pdb.set_trace()

    class_id_objects = np.concatenate([class_id_objects, class_id_humans]).tolist()
    scores_objects = np.concatenate([scores_objects, scores_persons]).tolist()

    if len(d_p_boxes) > select_threshold:
        d_p_boxes, scores_persons, class_id_humans = d_p_boxes[0:select_threshold], scores_persons[
                                                                                    0:select_threshold], class_id_humans[
                                                                                                         0:select_threshold]
    if len(d_o_boxes) > select_threshold:
        d_o_boxes, scores_objects, class_id_objects = d_o_boxes[0:select_threshold], scores_objects[
                                                                                     0:select_threshold], class_id_objects[
                                                                                                          0:select_threshold]

    # scores_objects.insert(0,1)
    object_one_hot = []
    for i, j in enumerate(class_id_objects):
        object_one_hot.append(one_hot[j])

    return d_p_boxes, d_o_boxes, scores_persons, scores_objects, class_id_humans, class_id_objects, annotation, shape, person_kp, object_one_hot


def get_compact_detections(segment_key, flag):
    d_p_boxes, d_o_boxes, scores_persons, scores_objects, class_id_humans, class_id_objects, annotation, shape,  persons_kp, object_one_hot = get_detections(
        segment_key, flag)
    img_W = shape[0]
    img_H = shape[1]
    no_person_dets = len(d_p_boxes)
    no_object_dets = len(d_o_boxes)
    persons_np = np.zeros([no_person_dets, 4], np.float32)
    objects_np = np.zeros([no_object_dets, 4], np.float32)

    all_o_center = [[0, 0]]
    for i in d_o_boxes:
        o_center = [(i[0]+i[2])/2, (i[1]+i[3])/2]
        all_o_center.append(o_center)

    for i in range(len(persons_kp)):
        for k in range(len(persons_kp[i])):
            del persons_kp[i][k][2]

    if no_person_dets != 0:
        persons_np = np.array(d_p_boxes, np.float32)

    objects_np = np.array(d_o_boxes, np.float32)
    persons_np = persons_np / np.array([img_W, img_H, img_W, img_H])
    objects_np = objects_np / np.array([img_W, img_H, img_W, img_H])

    object_one_hot = np.array(object_one_hot)
    return {'person_bbx': persons_np, 'objects_bbx': objects_np,
            'person_bbx_score': scores_persons, 'objects_bbx_score': scores_objects,
            'class_id_objects': class_id_objects, 'shape': shape, 'kp': persons_kp, 'object_one_hot': object_one_hot, 'all_o_center': all_o_center}


def get_attention_maps(segment_key, flag):
    compact_detections = get_compact_detections(segment_key, flag)
    persons_np, objects_np = compact_detections['person_bbx'], compact_detections['objects_bbx']
    Width = compact_detections['shape'][0]
    Height = compact_detections['shape'][1]
    kp = compact_detections['kp']
    union_box = []
    no_person_dets = len(persons_np)
    no_object_dets = len(objects_np)
    for dd_i in range(no_person_dets):
        for do_i in range(len(objects_np)):
            union_box.append(union_BOX(persons_np[dd_i], objects_np[do_i], segment_key, Height, Width, kp[dd_i]))
    return np.concatenate(union_box)


def get_compact_label(segment_key, flag):
    d_p_boxes, d_o_boxes, scores_persons, scores_objects, class_id_humans, class_id_objects, annotation, shape, person_kp, object_one_hot = get_detections(
        segment_key, flag)

    no_person_dets = len(d_p_boxes)
    no_object_dets = len(d_o_boxes)
    labels_np = np.zeros([no_person_dets, no_object_dets, NO_VERBS], np.int32)

    a_p_boxes = [ann['person_box'] for ann in annotation]
    iou_mtx = get_iou_mtx(a_p_boxes, d_p_boxes)

    if no_person_dets != 0 and len(a_p_boxes) != 0:
        max_iou_for_each_det = np.max(iou_mtx, axis=0)
        index_for_each_det = np.argmax(iou_mtx, axis=0)
        for dd in range(no_person_dets):
            cur_max_iou = max_iou_for_each_det[dd]
            if cur_max_iou < MATCHING_IOU:
                continue
            matched_ann = annotation[index_for_each_det[dd]]
            hoi_anns = matched_ann['hois']
            # verbs with actions######
            object_hois = [oi for oi in hoi_anns if len(oi['obj_box']) != 0]

            a_o_boxes = [oi['obj_box'] for oi in object_hois]
            iou_mtx_o = get_iou_mtx(a_o_boxes, d_o_boxes)

            if a_o_boxes and d_o_boxes:
                for do in range(len(d_o_boxes)):
                    for ao in range(len(a_o_boxes)):
                        cur_iou = iou_mtx_o[ao, do]
                        # enough iou
                        if cur_iou < MATCHING_IOU:
                            continue
                        current_hoi = object_hois[ao]
                        verb_idx = VERB2ID[current_hoi['verb']]
                        labels_np[dd, do, verb_idx] = 1

        comp_labels = labels_np.reshape(no_person_dets * (no_object_dets), NO_VERBS)
        labels_single = np.array([1 if i.any() == True else 0 for i in comp_labels])
        labels_single = labels_single.reshape(np.shape(labels_single)[0], 1)
        return {'labels_all': labels_np, 'labels_single': labels_single}
    else:
        comp_labels = labels_np.reshape(no_person_dets * (no_object_dets + 1), NO_VERBS)
        labels_single = np.array([1 if i.any() == True else 0 for i in comp_labels])
        labels_single = labels_single.reshape(np.shape(labels_single)[0], 1)
        return {'labels_all': labels_np, 'labels_single': labels_single}


def get_bad_detections(segment_key, flag):  # Detections Without any person#

    labels_all = get_compact_label(segment_key, flag)['labels_all']
    if labels_all.size == 0:
        return True
    else:
        return False


def union_BOX(roi_pers, roi_objs, segment_key, Height, Width, kp, H=64, W=64):
    assert H == W
    roi_pers = np.array(roi_pers * H, dtype=int)
    roi_objs = np.array(roi_objs * H, dtype=int)
    sample_box = np.zeros([1, 3, H, W])
    sample_box[0, 0, roi_pers[1]:roi_pers[3] + 1, roi_pers[0]:roi_pers[2] + 1] = 100
    sample_box[0, 1, roi_objs[1]:roi_objs[3] + 1, roi_objs[0]:roi_objs[2] + 1] = 100

    joints = kp
    joints.append([(joints[5][0] + joints[6][0]) / 2, (joints[5][1] + joints[6][1]) / 2])
    joint_relation = [[2, 4], [1, 3], [0, 2], [0, 1], [0, 17], [6, 17], [5, 17], [6, 8], [5, 7], [8, 10], [7, 9],
                      [12, 17], [11, 17], [12, 14], [11, 13], [14, 16], [13, 15]]
    color = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    skeleton = np.zeros((64, 64), dtype="float32")

    for i in range(len(joint_relation)):
        k1 = joints[joint_relation[i][0]][0:2]
        int_k1 = [int(k1[0] * W / Width), int(k1[1] * H / Height)]
        k2 = joints[joint_relation[i][1]][0:2]
        int_k2 = [int(k2[0] * W / Width), int(k2[1] * H / Height)]
        cv2.line(skeleton, tuple(int_k1), tuple(int_k2), (color[i]))
    sample_box[0, 2] = skeleton
    return sample_box


def clean_up_annotation(annotation):
    persons_dict = {}
    person_list = []
    object_list = []
    for hoi in annotation:

        box = hoi['person_bbx']
        box = [int(coord) for coord in box]
        box = clean_person(person_list, np.asarray(box))
        # import pdb;pdb.set_trace()
        dkey = tuple(box)
        objects = hoi['object']
        if len(objects['obj_bbx']) == 0:  # no obj case
            cur_oi = {'verb': hoi['Verbs'],
                      'obj_box': [],
                      'obj_str': '',
                      'obj_id': ''
                      }
        else:
            cur_obj = clean_object(object_list, np.asarray(hoi['object']['obj_bbx']))
            cur_oi = {'verb': hoi['Verbs'],
                      # 'obj_box':[int(coord) for coord in hoi['object']['obj_bbx']],
                      'obj_box': cur_obj,
                      'obj_str': hoi['object']['obj_name'],
                      'obj_id': int(obj_list[hoi['object']['obj_name']])

                      }

        if dkey in persons_dict:
            persons_dict[dkey]['hois'].append(cur_oi)
        else:
            persons_dict[dkey] = {'person_box': box, 'hois': [cur_oi]}

    pers_list = []
    for dkey in persons_dict:
        pers_list.append(persons_dict[dkey])
    # import pdb;pdb.set_trace()
    return pers_list



def get_distance(segment_key, flag):
    compact_detections = get_compact_detections(segment_key, flag)
    persons_np, objects_np = compact_detections['person_bbx'], compact_detections['objects_bbx']
    kp, o_center = compact_detections['kp'], compact_detections['all_o_center']
    Width = compact_detections['shape'][0]
    Height = compact_detections['shape'][1]
    distance = []
    no_person_dets = len(persons_np)
    no_object_dets = len(objects_np)
    for dd_i in range(no_person_dets):
        for do_i in range(no_object_dets):
            distance.append(Dis(kp[dd_i], o_center[do_i], Width, Height))
    return distance

def Dis(kp, o_center, Width, Height):
    distance = []
    for i in kp:
        i = [i[0]/Width, i[1]/Height]
        o_center = [o_center[0]/Width, o_center[1]/Height]
        i, o_center = np.array(i), np.array(o_center)
        d1 = np.linalg.norm(i-o_center)
        distance.append(d1)
    return distance

def clean_person(person_list, box):
    if len(person_list) == 0:
        person_list.append(box)
        return box.tolist()


    else:
        for person in person_list:
            if len(box) == 0:
                if len(person) == 0:
                    return person.tolist()
            elif len(person) == 0:
                continue


            else:
                IOU = IoU_box(person, box)
                if IOU >= 0.5:
                    return person.tolist()
        person_list.append(box)
        return box.tolist()


def clean_object(object_list, box):
    if len(object_list) == 0:
        object_list.append(box)
        return box.tolist()
    else:
        for object in object_list:

            if len(box) == 0:
                if len(object) == 0:
                    return object.tolist()
            elif len(object) == 0:
                continue
            else:
                IOU = IoU_box(object, box)
                if IOU >= 0.5:
                    return object.tolist()

        object_list.append(box)
        return box.tolist()


def get_boxes_det(dets, img_H, img_W):
    boxes = []
    scores = []
    class_no = []
    for det in dets:
        top, left, bottom, right = det['box_coords']
        scores.append(det['score'])
        if len(det['class_str'].split()) == 2:
            # import pdb;pdb.set_trace()
            str = det['class_str'].split()[0] + '_' + det['class_str'].split()[1]
        else:
            str = det['class_str']

        class_no.append(int(obj_list[str]))
        # left, top, right, bottom = left*img_W, top*img_H, right*img_W, bottom*img_H
        left, top, right, bottom = left, top, right, bottom
        boxes.append([left, top, right, bottom])
    # import pdb;pdb.set_trace()
    return boxes, scores, class_no


def get_person_det(dets):
    boxes = []
    scores = []
    class_no = []
    kp = []
    for det in dets:
        left, top, right, bottom = det['box_coords']
        scores.append(det['score'])
        # print(det['class_str'].split())
        # print(len(det['class_str'].split()))
        # time.sleep(66)
        if len(det['class_str'].split()) == 2:
            # import pdb;pdb.set_trace()
            str = det['class_str'].split()[0] + '_' + det['class_str'].split()[1]
            print(str)
        else:
            str = det['class_str']
        class_no.append(int(obj_list[str]))
        kp.append(det['keypoints'])
        # left, top, right, bottom = left, top, right, bottom
        boxes.append([left, top, right, bottom])
    return boxes, scores, class_no, kp


def get_iou_mtx(anns, dets):
    no_gt = len(anns)
    no_dt = len(dets)
    iou_mtx = np.zeros([no_gt, no_dt])

    for gg in range(no_gt):
        gt_box = anns[gg]
        for dd in range(no_dt):
            dt_box = dets[dd]
            iou_mtx[gg, dd] = IoU_box(gt_box, dt_box)

    return iou_mtx


def analyze_detections(detections, detections_per, SCORE_TH, SCORE_OBJ):
    persons = []
    objects = []

    for det in detections_per['detections']:
        if det['class_str'] == 'person':
            if det['score'] < SCORE_TH:
                continue
            persons.append(det)

    for det1 in detections['detections']:
        if det1['class_str'] != 'person':
            if det1['score'] < SCORE_OBJ:
                continue
            objects.append(det1)

    return persons, objects


def IoU_box(box1, box2):
    '''
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2
 
    returns intersection over union
    '''
    try:
        left1, top1, right1, bottom1 = box1
        left2, top2, right2, bottom2 = box2
    except:
        IoU = 0
        return IoU

    left_int = max(left1, left2)
    top_int = max(top1, top2)

    right_int = min(right1, right2)
    bottom_int = min(bottom1, bottom2)

    areaIntersection = max(0, right_int - left_int) * max(0, bottom_int - top_int)

    area1 = (right1 - left1) * (bottom1 - top1)
    area2 = (right2 - left2) * (bottom2 - top2)

    IoU = areaIntersection / float(area1 + area2 - areaIntersection)
    return IoU


def dry_run():

    print("Doing a test run to detect bad detections\n")

    with open('/home/hzy/VSGNet-master/All_data/bad_detections_hico/bad_detections_train.json') as fp:
        bad_detections_train = json.load(fp)

    print("In training set object detector failed to detect any person in {} images".format(len(bad_detections_train)))

    with open('/home/hzy/VSGNet-master/All_data/bad_detections_hico/bad_detections_test.json') as fp:
        bad_detections_test = json.load(fp)
    print("In testing set object detector failed to detect any person in {} images".format(len(bad_detections_test)))
    # import pdb;pdb.set_trace()
    return bad_detections_train, bad_detections_test


if __name__ == "__main__":
    new_anns = {}
    compact_dets = {}
    att_maps = {}
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type_of_data', type=str, required=False, default='train', help="type_of_data")
    args = parser.parse_args()
    flag = args.type_of_data
    bad_detections_train, bad_detections_test = dry_run()
    b_d_tr, b_d_test = [int(l) for l in bad_detections_train], [int(l) for l in bad_detections_test]
    phases = ['train', 'test']
    from tqdm import tqdm

    # import pdb;pdb.set_trace()
    for flag in phases:
        if flag == 'train':
            ALL_SEGS = ANNOTATIONS_train.keys()


        elif flag == 'test':
            ALL_SEGS = ANNOTATIONS_test.keys()
        ALL_SEGS = [int(v) for v in ALL_SEGS]
        ALL_SEGS.sort()
        for segkey in tqdm(ALL_SEGS):
            # import pdb;pdb.set_trace()
            if segkey in (b_d_tr + b_d_test):
                new_anns[segkey] = get_compact_label(segkey, flag)
                compact_dets[segkey] = get_compact_detections(segkey, flag)
                att_maps[segkey] = get_attention_maps(segkey, flag)
                # import pdb;pdb.set_trace()

    pass
