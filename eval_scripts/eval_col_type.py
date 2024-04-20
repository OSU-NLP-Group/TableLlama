import json
import argparse
from collections import Counter
from metric import *


def r_p_f1(true_positive, pred, targets):
    if pred != 0:
        precision = true_positive/pred
    else:
        precision = 0
    recall = true_positive/targets
    if (precision + recall) != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return  recall, precision, f1

def get_r_p_f1_for_each_type(ground_truth_list, pred_list):
    
    # import pdb
    # pdb.set_trace()

    total_ground_truth_col_types = 0
    total_pred_col_types = 0
    joint_items_list = []
    for i in range(len(ground_truth_list)):
        total_ground_truth_col_types += len(ground_truth_list[i])
        total_pred_col_types += len(pred_list[i])
        joint_items = [item for item in pred_list[i] if item in ground_truth_list[i]]
        # total_ground_truth_col_types += len(list(set(ground_truth_list[i])))
        # total_pred_col_types += len(list(set(pred_list[i])))
        # joint_items = [item for item in list(set(pred_list[i])) if item in list(set(ground_truth_list[i]))]
        joint_items_list += joint_items
    
    # import pdb
    # pdb.set_trace()

    gt_entire_col_type = {}
    for i in range(len(ground_truth_list)):
        gt = list(set(ground_truth_list[i]))
        for k in range(len(gt)):
            if gt[k] not in gt_entire_col_type.keys():
                gt_entire_col_type[gt[k]] = 1
            else:
                gt_entire_col_type[gt[k]] += 1
    # print(len(gt_entire_col_type.keys()))

    pd_entire_col_type = {}
    for i in range(len(pred_list)):
        pd = list(set(pred_list[i]))
        for k in range(len(pd)):
            if pd[k] not in pd_entire_col_type.keys():
                pd_entire_col_type[pd[k]] = 1
            else:
                pd_entire_col_type[pd[k]] += 1
    # print(len(pd_entire_col_type.keys()))

    joint_entire_col_type = {}
    for i in range(len(joint_items_list)):
        if joint_items_list[i] not in joint_entire_col_type.keys():
            joint_entire_col_type[joint_items_list[i]] = 1
        else:
            joint_entire_col_type[joint_items_list[i]] += 1
    # print(len(joint_entire_col_type.keys()))

    precision = len(joint_items_list)/total_pred_col_types
    recall = len(joint_items_list)/total_ground_truth_col_types
    f1 = 2 * precision * recall / (precision + recall)

    sorted_gt = sorted(gt_entire_col_type.items(), key=lambda x: x[1], reverse = True)
    # print(sorted_gt)
    # print("len(joint_items_list):", len(joint_items_list))
    # print("total_ground_truth_col_types:", total_ground_truth_col_types)
    # print("total_pred_col_types:", total_pred_col_types)
    print("precision::", precision)
    print("recall:", recall)
    print("f1:", f1)

    # print('r_p_f1(joint_entire_col_type["people.person"])', r_p_f1(joint_entire_col_type["people.person"], pd_entire_col_type["people.person"], gt_entire_col_type["people.person"]))
    # print('r_p_f1(joint_entire_col_type["sports.pro_athlete"])', r_p_f1(joint_entire_col_type["sports.pro_athlete"], pd_entire_col_type["sports.pro_athlete"], gt_entire_col_type["sports.pro_athlete"]))
    # print('r_p_f1(joint_entire_col_type["film.actor"])', r_p_f1(joint_entire_col_type["film.actor"], pd_entire_col_type["film.actor"], gt_entire_col_type["film.actor"]))
    # print('r_p_f1(joint_entire_col_type["location.location"])', r_p_f1(joint_entire_col_type["location.location"], pd_entire_col_type["location.location"], gt_entire_col_type["location.location"]))
    # print('r_p_f1(joint_entire_col_type["location.citytown"])', r_p_f1(joint_entire_col_type["location.citytown"], pd_entire_col_type["location.citytown"], gt_entire_col_type["location.citytown"]))


def remove_ele(list, ele):
    while True:
        if ele in list:
            list.remove(ele)
            continue
        else:
            break
    return list

def get_index(lst=None, item=''):
    return [i for i in range(len(lst)) if lst[i] == item]

def main(args):
    with open(args.pred_file, "r") as f:
            col_type = json.load(f)

    ground_truth_list = []
    pred_list = []
    for i in range(len(col_type)):
        item = col_type[i]
        ground_truth = item["ground_truth"]
        # pred = item["predict"].strip("</s>").split(",")
        pred = item["predict"].split("</s>")[0].split(", ")
        ground_truth_list.append(ground_truth)
        pred_list.append(pred)

    get_r_p_f1_for_each_type(ground_truth_list, pred_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_file', type=str, default='/TableLlama/ckpfinal_pred/col_type_pred.json', help='')
    args = parser.parse_args()
    main(args)
    