import json
import argparse


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
    wrong_extra_pred_idx_list = []

    total_ground_truth_col_types = 0
    total_pred_col_types = 0
    joint_items_list = []
    for i in range(len(ground_truth_list)):
        total_ground_truth_col_types += len(ground_truth_list[i])
        total_pred_col_types += len(pred_list[i])
        joint_items = [item for item in pred_list[i] if item in ground_truth_list[i]]
        joint_items_list += joint_items

    gt_entire_col_type = {}
    for i in range(len(ground_truth_list)):
        gt = list(set(ground_truth_list[i]))
        for k in range(len(gt)):
            if gt[k] not in gt_entire_col_type.keys():
                gt_entire_col_type[gt[k]] = 1
            else:
                gt_entire_col_type[gt[k]] += 1
    print(len(gt_entire_col_type.keys()))

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

    # print('r_p_f1(joint_entire_col_type["location.location.containedby"])', r_p_f1(joint_entire_col_type["location.location.containedby"], pd_entire_col_type["location.location.containedby"], gt_entire_col_type["location.location.containedby"]))
    # print('r_p_f1(joint_entire_col_type["film.film.directed_by"])', r_p_f1(joint_entire_col_type["film.film.directed_by"], pd_entire_col_type["film.film.directed_by"], gt_entire_col_type["film.film.directed_by"]))
    # print('r_p_f1(joint_entire_col_type["award.award_nominated_work.award_nominations-award.award_nomination.award_nominee"])', r_p_f1(joint_entire_col_type["award.award_nominated_work.award_nominations-award.award_nomination.award_nominee"], pd_entire_col_type["award.award_nominated_work.award_nominations-award.award_nomination.award_nominee"], gt_entire_col_type["award.award_nominated_work.award_nominations-award.award_nomination.award_nominee"]))
    # print('r_p_f1(joint_entire_col_type["award.award_winning_work.awards_won-award.award_honor.award_winner"])', r_p_f1(joint_entire_col_type["award.award_winning_work.awards_won-award.award_honor.award_winner"], pd_entire_col_type["award.award_winning_work.awards_won-award.award_honor.award_winner"], gt_entire_col_type["award.award_winning_work.awards_won-award.award_honor.award_winner"]))
    # print('r_p_f1(joint_entire_col_type["sports.sports_team.arena_stadium"])', r_p_f1(joint_entire_col_type["sports.sports_team.arena_stadium"], pd_entire_col_type["sports.sports_team.arena_stadium"], gt_entire_col_type["sports.sports_team.arena_stadium"]))


def main(args):
    with open(args.pred_file, "r") as f:
        data = json.load(f)

    ground_truth_list = []
    pred_list = []
    for i in range(len(data)):
        item = data[i]
        ground_truth = item["ground_truth"]
        # pred = item["predict"].strip("</s>").split(",")
        pred = item["predict"].split("</s>")[0].split(", ")
        ground_truth_list.append(ground_truth)
        pred_list.append(pred)

    get_r_p_f1_for_each_type(ground_truth_list, pred_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_file', type=str, default='/TableLlama/ckpfinal_pred/rel_extraction_pred.json', help='')
    args = parser.parse_args()
    main(args)