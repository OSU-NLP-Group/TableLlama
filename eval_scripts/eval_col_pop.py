import json
import argparse
from metric import *


def get_map_recall(data, data_name):
    rs = []
    recall = []
    for i in range(len(data)):
        ground_truth = data[i]["target"].strip(".")
        # ground_truth = data[i]["target"].strip(".")
        pred = data[i]["predict"].strip(".")
        if "</s>" in pred:
            end_tok_ix = pred.rfind("</s>")
            pred = pred[:end_tok_ix]
        ground_truth_list = ground_truth.split(", ")
        # ground_truth_list = test_col_pop_rank[i]["target"].strip(".").split(", ")
        pred_list = pred.split(", ")
        for k in range(len(pred_list)):
            pred_list[k] = pred_list[k].strip("<>")

        # print(len(ground_truth_list), len(pred_list))

        # import pdb
        # pdb.set_trace()
        # add to remove repeated generated item
        new_pred_list = list(set(pred_list))
        new_pred_list.sort(key = pred_list.index)
        r = [1 if z in ground_truth_list else 0 for z in new_pred_list]
        ap = average_precision(r)
        # print("ap:", ap)
        rs.append(r)

        # if sum(r) != 0:
        #     recall.append(sum(r)/len(ground_truth_list))
        # else:
        #     recall.append(0)
        recall.append(sum(r)/len(ground_truth_list))
    map = mean_average_precision(rs)
    m_recall = sum(recall)/len(data)
    f1 = 2 * map * m_recall / (map + m_recall)
    print(data_name, len(data))
    print("mean_average_precision:", map)


def main(args):
    file = args.pred_file

    with open(file, "r") as f:
        col_pop = json.load(f)

    get_map_recall(col_pop, 'col_pop')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_file', type=str, default='/TableLlama/ckpfinal_pred/col_pop_pred.json', help='')
    args = parser.parse_args()
    main(args)
