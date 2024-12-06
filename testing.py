import csv
import torch
import torch.nn as nn
import numpy as np

# Best_config = {'l1': 128, 'l2': 64, 'lr': 0.00075, 'batch_size': 64}
# Best_config['hp_time'] = str(12) + ' minutes' + str(5) + ' seconds'
# with open('Best_config.csv', 'w') as f:
#     for key in Best_config.keys():
#         f.write("%s,%s\n"%(key,Best_config[key]))
'''
loss = nn.CrossEntropyLoss()

Y = torch.tensor([2,0,1])

Y_pred_good = torch.tensor([[-.1, 1.0, 2.1],[2.0, 1.0, 0.1],[0.1, 3.0, 0.1]])  # After softmax: 0.0922, 0.2267, 0.6811 0.6590, 0.2424, 0.0986 0.0496, 0.9009, 0.0496
Y_pred_bad = torch.tensor([[2.1, 1.0, 0.1],[0.1, 1.0, 2.1],[0.1, 3.0, 0.1]])   # After softmax: 0.6811, 0.2267, 0.0922 0.0922, 0.2267, 0.6811 0.0496, 0.9009, 0.0496

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(l1.item())
print(l2.item())

loss = nn.NLLLoss()

Y = torch.tensor([2,0,1])

Y_pred_good = torch.tensor([[0.1, 1.0, 2.1],[2.0, 1.0, 0.1],[0.1, 3.0, 0.1]])  # After softmax: 0.6590, 0.2424, 0.0986
Y_pred_bad = torch.tensor([[2.1, 1.0, 0.1],[0.1, 1.0, 2.1],[0.1, 3.0, 0.1]])   # After softmax: 0.1587, 0.7113, 0.1299

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(l1.item())
print(l2.item())

loss = nn.Softmax()

Y = torch.tensor([2,0,1])

Y_pred_good = torch.tensor([[0.1, 1.0, 2.1],[2.0, 1.0, 0.1],[0.1, 3.0, 0.1]])  # After softmax: 0.6590, 0.2424, 0.0986
Y_pred_bad = torch.tensor([[2.1, 1.0, 0.1],[0.1, 1.0, 2.1],[0.1, 3.0, 0.1]])   # After softmax: 0.1587, 0.7113, 0.1299

l1 = loss(Y_pred_good)
l2 = loss(Y_pred_bad)

print(l1)
print(l2)
'''

a_list = [5,7,8,1,3]


def select_sort(items, sequence= lambda x,y: x<y):
    sequence_items = items[:].copy()
    # items_sum = np.sum(items, axis = 0)
    for i in range(len(sequence_items)-1):
        min_index = i
        for j in range(i+1, len(sequence_items)):
            if sequence(sequence_items[j],sequence_items[min_index]):
                min_index = j
        sequence_items[i], sequence_items[min_index] = sequence_items[min_index], sequence_items[i]
    return sequence_items

b_list = (select_sort(np.array(a_list)))
print(type(a_list))
print(type(b_list))
c_list = (a_list + b_list)/2
print(np.array(a_list).shape)
d_list = np.multiply((np.array(a_list).reshape(-1)),2)

print(a_list)
print(b_list)
print(c_list)
print(d_list)