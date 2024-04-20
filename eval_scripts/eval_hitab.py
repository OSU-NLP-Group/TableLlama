import json
import sys
import argparse
from table_utils import evaluate


def main(args):
    with open(args.pred_file, "r") as f:
        data = json.load(f)

    pred_list = []
    gold_list = []
    for i in range(len(data)):
        if len(data[i]["predict"].strip("</s>").split(">, <")) > 1:
            instance_pred_list = data[i]["predict"].strip("</s>").split(">, <")
            pred_list.append(instance_pred_list)
            gold_list.append(data[i]["output"].strip("</s>").split(">, <"))
        else:
            pred_list.append(data[i]["predict"].strip("</s>"))
            gold_list.append(data[i]["output"].strip("</s>"))

    print(evaluate(gold_list, pred_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_file', type=str, default='/TableLlama/ckpfinal_pred/hitab_pred.json', help='')
    args = parser.parse_args()
    main(args)