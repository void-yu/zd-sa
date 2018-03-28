from aip import AipNlp

APP_ID = '10671217'
API_KEY = 'uHzrhAWY15cMgOlj0WmYGb0Q'
SECRET_KEY = 'lWA0VON2gCXrTwjnourPNcTOrE3oZXk1'
client = AipNlp(APP_ID, API_KEY, SECRET_KEY)


def f_value(num, labels, expections):
    # 真正例
    TP = 0
    # 假正例
    FP = 0
    # 假反例
    FN = 0
    # 真反例
    TN = 0

    # We pay more attention on negative samples.
    for i in range(num):
        if labels[i] == 0 and expections[i] == 0:
            TP += 1
        elif labels[i] == 0 and expections[i] == 1:
            FN += 1
        elif labels[i] == 1 and expections[i] == 0:
            FP += 1
        elif labels[i] == 1 and expections[i] == 1:
            TN += 1

    P = TP / (TP + FP + 0.0001)
    R = TP / (TP + FN + 0.0001)
    F = 2 * P * R / (P + R + 0.0001)
    P_ = TN / (TN + FN + 0.0001)
    R_ = TN / (TN + FP + 0.0001)
    F_ = 2 * P_ * R_ / (P_ + R_ + 0.0001)
    ACC = (TP + TN) / (TP + TN + FP + FN + 0.0001)

    print("Accuracy rate: {:.4f}".format(ACC))

    print("About negative samples:")
    print("     precision rate: {:.4f}".format(P))
    print("     recall rate: {:.4f}".format(R))
    print("     f-value: {:.4f}".format(F))

    print("About positive samples:")
    print("     precision rate: {:.4f}".format(P_))
    print("     recall rate: {:.4f}".format(R_))
    print("     f-value: {:.4f}".format(F_))


# import pickle
# import pandas as pd
# import numpy as np
# import time
#
# start = time.time()
#
# df = pd.read_excel('data/corpus/check/klb.xlsx')
# text = df.iloc[:, [0, 1]]
# text = [[i[0], i[1]] for i in np.array(text).tolist()]
# text = np.array(text).tolist()
#
# labels = []
# expections = []
# write2file = []
#
# for i in text:
#     try:
#         a = client.sentimentClassify(i[1])
#         labels.append(0 if i[0] == 'F' else 1)
#         posi = a['items'][0]['positive_prob']
#         nega = a['items'][0]['negative_prob']
#         if posi > nega:
#             expections.append(1)
#         else:
#             expections.append(0)
#         print(labels[-1], expections[-1], i[1])
#         write2file.append([i[1], labels[-1], expections[-1]])
#     except Exception as e:
#         print(e)
#
# with open('data/klb_baidu_result', 'wb') as fp:
#     pickle.dump(write2file, fp)

import pickle
with open('data/klb_baidu_result', 'rb') as fp:
    write2file = pickle.load(fp)

# for i in write2file:
#     if i[1] == 1 and i[2] == 0:
#         print(i)
# labels = [i[1] for i in write2file]
# expections = [i[2] for i in write2file]
# f_value(len(labels), labels, expections)

import pandas as pd
df_o = pd.DataFrame(write2file)
writer = pd.ExcelWriter('data/klb.xlsx')
df_o.to_excel(writer, 'Sheet1')
writer.save()

