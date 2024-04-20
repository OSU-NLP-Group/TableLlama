import json
import argparse


def main(args):
    with open(args.pred_file, "r") as f:
        data = json.load(f)

    correct = 0
    remove_count = 0
    for i in range(len(data)):
        ground_truth = data[i]["output"]
        prediction = data[i]["predict"].strip("</s>")
        # if prediction.find(ground_truth) == 0:
        if prediction == ground_truth:
            correct += 1
        if prediction.find("<s>") == 0:
            remove_count += 1

    print("correct:", correct)        
    # print("remove_count:", remove_count)
    print("accuracy:", correct/(len(data)-remove_count))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_file', type=str, default='/TableLlama/ckpfinal_pred/tabfact_pred.json', help='')
    args = parser.parse_args()
    main(args)
