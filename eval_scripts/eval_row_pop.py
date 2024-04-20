import json
import argparse
from metric import *


def get_map_recall(data, data_name):
    rs = []
    recall = []
    ap_list = []
    for i in range(len(data)):
        pred = data[i]["predict"].strip(".")
        if "</s>" in pred:
            end_tok_ix = pred.rfind("</s>")
            pred = pred[:end_tok_ix]
        ground_truth_list = data[i]["target"]
        pred_list = pred.split(", ")
        for k in range(len(pred_list)):
            pred_list[k] = pred_list[k].strip("<>")

        # add to remove repeated generated item
        new_pred_list = list(set(pred_list))
        new_pred_list.sort(key = pred_list.index)
        # r = [1 if z in ground_truth_list else 0 for z in pred_list]
        r = [1 if z in ground_truth_list else 0 for z in new_pred_list]
        # ap = average_precision(r)
        ap = row_pop_average_precision(r, ground_truth_list)
        # print("ap:", ap)
        ap_list.append(ap)

    map = sum(ap_list)/len(data)
    m_recall = sum(recall)/len(data)
    f1 = 2 * map * m_recall / (map + m_recall)
    print(data_name, len(data))
    print("mean_average_precision:", map)


# def merge_pred_from_multi_files(data_path, store_path):
#     merged_data = []
#     for i in range(6):
#         with open(data_path + "row_pop_pred_" + str(i) + ".json", "r") as f:
#             temp_data = json.load(f)
#         merged_data += temp_data
#     with open(store_path, "w") as f:
#         json.dump(merged_data, f, indent = 2)


def main(args):
    with open(args.pred_file, "r") as f:
        row_pop = json.load(f)
    get_map_recall(row_pop, 'row_pop')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_file', type=str, default='/TableLlama/ckpfinal_pred/row_pop_pred.json', help='')
    args = parser.parse_args()
    main(args)

