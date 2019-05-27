import pickle

from utils.threshold import load_pkl, f1


def intra_sen_classify(list,predict,true_y):
    intra_true_y=[]
    intra_predict=[]

    for i in range(len(predict)):
        if i not in list:
            temp = predict[i]
            if(predict[i]>0.5):
                intra_predict.append(1)
            else:
                intra_predict.append(0)

    true_y = true_y.tolist()
    for i in range(len(true_y)):
        if i not in list:
            temp= true_y[i][0]
            intra_true_y.append(true_y[i][0])

    return intra_predict,intra_true_y


def inter_sen_classify(list,predict,true_y):
    intre_true_y = []
    intre_predict = []

    for i in range(len(predict)):
        if i in list:
            temp = predict[i]
            if (predict[i] > 0.5):
                intre_predict.append(1)
            else:
                intre_predict.append(0)

    true_y = true_y.tolist()
    for i in range(len(true_y)):
        if i in list:
            temp = true_y[i][0]
            intre_true_y.append(true_y[i][0])

    return intre_predict, intre_true_y


if __name__ == "__main__":
    test_inter_sen_list = load_pkl('test_delete_list')
    list = test_inter_sen_list[0]
    file = open('../result/' + '0.5632286995515694_predict', 'rb')
    predict = pickle.load(file)
    file.close()
    true_y = load_pkl('tests_y')

    intra_predict, intra_true_y = intra_sen_classify(list, predict, true_y)
    intre_predict, intre_true_y = inter_sen_classify(list, predict, true_y)

    p1, r1, f_1 = f1(intra_true_y, intra_predict)
    print(f_1)

    p2, r2, f_2 = f1(intre_true_y, intre_predict)
    print(f_2)

    print()