import re
import math
import numpy as np
import babel
from babel import numbers

from qa_datadump_utils import naive_str_to_float


def hmt_score(prediction, answer):
    prediction = hmt_process_answer(prediction)
    answer = hmt_process_answer(answer)
    if hmt_equal(prediction, answer):
        return 1.0
    else:
        return 0.0


def hmt_process_answer(answer):
    """ 4 types of answer: 1)region; 2)num_list(aggr); 3)header_list(argmax); 4)num(count/div)"""
    if isinstance(answer, int) or isinstance(answer, float):
        return float(answer)
    if isinstance(answer, str):
        return naive_str_to_float(answer.strip().lower())
    if isinstance(answer, list):
        if isinstance(answer[0], list):  # pred region
            if len(answer) == 1 and len(answer[0]) == 1:  # pred region with one cell, flatten
                return hmt_process_answer(answer[0][0])
            elif len(answer) == 1:  # pred region with one line
                return hmt_process_answer(answer[0])
            elif len(answer[0]) == 1:  # pred region with one line
                return hmt_process_answer([row[0] for row in answer])
            else:  # pred region is a matrix
                return [hmt_process_answer(a) for a in answer]
        else:  # list or processed single-line region
            if len(answer) == 1:  # answer with one cell or pred list
                return hmt_process_answer(answer[0])
            else:
                return [hmt_process_answer(a) for a in answer]


def hmt_equal(prediction, answer):
    # import pdb
    # pdb.set_trace()
    try:
        if prediction.split(",")[0] not in ['inf', 'infinity', 'INF', 'INFINITY', 'True', 'NAN', 'nan', 'False', '-inf', '-INF', '-INFINITY', '-infinity', 'NaN', 'Nan'] and float(prediction.split(",")[0]):
            prediction = float(prediction.split(",")[0])
    except:
        prediction = prediction

    if type(prediction) != type(answer):
        return False
    if isinstance(prediction, str):
        # print("str:", answer, prediction)
        return prediction.find(answer) == 0
    if isinstance(prediction, int) or isinstance(prediction, float):
        # return math.fabs(prediction - answer) < 1e-5
        # print("float/int:", answer, prediction)
        return math.fabs(prediction - answer) < 1e-5
    if isinstance(prediction, list):
        if len(prediction) != len(answer):
            return False
        return all([hmt_equal(prediction[i], answer[i]) for i in range(len(prediction))])


def evaluate(golds, preds):
    correct_num = 0.0
    # print(golds[0]["answer"])
    # print(preds[0])
    # import pdb
    # pdb.set_trace()

    correct_list = []
    wrong_list = []
    for i in range(len(preds)):
        gold = golds[i]
        pred = preds[i]
        if hmt_score(pred, gold) == 1:
            correct_item = {}
            correct_item["idx"] = i
            correct_item["output"] = gold
            correct_item["pred"] = pred
            correct_list.append(correct_item)
        else:
            wrong_item = {}
            wrong_item["idx"] = i
            wrong_item["output"] = gold
            wrong_item["pred"] = pred
            wrong_list.append(wrong_item)

        correct_num += hmt_score(pred, gold)
        
    correct_percent = 100.0 * correct_num/len(golds)

    # import json
    # with open("/users/PAA0201/shubaobao/LongLoRA/tommy_server/ckp-15k-pred/correct_hitab.json", "w") as f:
    #     json.dump(correct_list, f, indent = 2)
    # with open("/users/PAA0201/shubaobao/LongLoRA/tommy_server/ckp-15k-pred/wrong_hitab.json", "w") as f:
    #     json.dump(wrong_list, f, indent = 2)
    # print("correct_percent:", correct_percent)
    # import pdb
    # pdb.set_trace()
    return {'exact_match': correct_percent}



if __name__ == '__main__':
    # main()
    # test()
    pred = "-0.8"
    gold = 0.8
    print(hmt_score(pred, gold))
