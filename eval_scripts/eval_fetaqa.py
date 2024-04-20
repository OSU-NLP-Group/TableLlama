import argparse
import evaluate
import json

def main(args):
    with open(args.pred_file, "r") as f:
        data = json.load(f)

    test_examples_answer = [x["output"] for x in data]
    test_predictions_pred = [x["predict"].strip("</s>") for x in data]
    predictions = test_predictions_pred
    references = test_examples_answer

    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=predictions, references=references)
    print(results)

    # bleu = evaluate.load('bleu')
    # results = bleu.compute(predictions=predictions, references=references)
    # print(results)

    sacrebleu = evaluate.load('sacrebleu')
    results = sacrebleu.compute(predictions=predictions, references=references)
    print(results)

    meteor = evaluate.load('meteor')
    results = meteor.compute(predictions=predictions, references=references)
    print(results)

    # bleurt = evaluate.load('bleurt')
    # results = bleurt.compute(predictions=predictions, references=references)
    # print(results)

    # bertscore = evaluate.load('bertscore')
    # results = bertscore.compute(predictions=predictions, references=references)
    # print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_file', type=str, default='/TableLlama/ckpfinal_pred/fetaqa_pred.json', help='')
    args = parser.parse_args()
    main(args)