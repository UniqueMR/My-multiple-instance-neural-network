
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import torch.nn.functional as func

device = torch.device('cpu')

class MI_Net(torch.nn.Module):
    def __init__(self):
        super(MI_Net,self).__init__()
        self.fc1 = torch.nn.Linear(38,256)
        self.fc2 = torch.nn.Linear(256,128)
        self.fc3 = torch.nn.Linear(128,64)
        self.fc4a = torch.nn.Linear(64,2)
        self.fc4b = torch.nn.Linear(128,2)
        self.fc4c = torch.nn.Linear(256,2)
        self.pl = torch.nn.AdaptiveAvgPool1d(1)
        #self.fc = torch.nn.Sequential(
        #    torch.nn.Linear(166,256),
        #    torch.nn.ReLU(),
        #    torch.nn.Linear(256,128),
        #    torch.nn.ReLU(),
        #    torch.nn.Linear(128,64),
        #    torch.nn.ReLU(),
        #    torch.nn.Linear(64,2))
        #self.pl = torch.nn.AdaptiveMaxPool1d(1)

    def forward(self,x):
        out = []
        x1 = self.fc1(x)
        x1 = F.relu(x1)
        x2 = self.fc2(x1)
        x2 = F.relu(x2)
        x3 = self.fc3(x2)
        x3 = F.relu(x3)
        x1 = x1.transpose(0,1)
        x1 = torch.unsqueeze(x1,0)
        x1 = self.pl(x1)
        x1 = torch.squeeze(x1,0)
        x1 = x1.transpose(0,1)
        out1 = self.fc4c(x1)
        x2 = x2.transpose(0,1)
        x2 = torch.unsqueeze(x2,0)
        x2 = self.pl(x2)
        x2 = torch.squeeze(x2,0)
        x2 = x2.transpose(0,1)
        out2 = self.fc4b(x2)
        x3 = x3.transpose(0,1)
        x3 = torch.unsqueeze(x3,0)
        x3 = self.pl(x3)
        x3 = torch.squeeze(x3,0)
        x3 = x3.transpose(0,1)
        out3 = self.fc4a(x3)
     
        out.append(out1)
        out.append(out2)
        out.append(out3)
        #out = self.fc(x)
        #out = torch.transpose(out,0,1)
        #out = torch.unsqueeze(out,0)
        #out = self.pl(out)
        #out = torch.squeeze(out,0)
        #out = torch.transpose(out,0,1)
        return out

    def predict(self,x):
        pred0 = func.softmax(self.forward(x)[0])
        pred1 = func.softmax(self.forward(x)[1])
        pred2 = func.softmax(self.forward(x)[2])
        pred = pred0 + pred1 + pred2
        ans = []
        for t in pred:
            if t[0] > t[1]:
                ans.append(0)
            else:
                ans.append(1)

        return torch.tensor(ans)

def load_dataset(dataset_nm, n_folds):
    """Load data from file, do pre-processing, split it into train/test set.
    Parameters
    -----------------
    dataset_nm : string
        Name of dataset.
    n_folds : int
        Number of cross-validation folds.
    Returns
    -----------------
    datasets : list
        List contains split datasets for k-Fold cross-validation.
    """
    # load data from file
    data = sio.loadmat('./dataset/'+dataset_nm+'.mat')
    ins_fea = data['x']['data'][0,0]
    if dataset_nm.startswith('musk'):
        bags_nm = data['x']['ident'][0,0]['milbag'][0,0]
    else:
        bags_nm = data['x']['ident'][0,0]['milbag'][0,0][:,0]
    bags_label = data['x']['nlab'][0,0][:,0] - 1

    # L2 norm for musk1 and musk2
    if dataset_nm.startswith('newsgroups') is False:
        mean_fea = np.mean(ins_fea, axis=0, keepdims=True)+1e-6
        std_fea = np.std(ins_fea, axis=0, keepdims=True)+1e-6
        ins_fea = np.divide(ins_fea-mean_fea, std_fea)

    # store data in bag level
    ins_idx_of_input = {}            # store instance index of input
    for id, bag_nm in enumerate(bags_nm):
        if bag_nm in ins_idx_of_input:      
            ins_idx_of_input[bag_nm].append(id)
        else:                                
            ins_idx_of_input[bag_nm] = [id]
    bags_fea = []
    for bag_nm, ins_idxs in ins_idx_of_input.items():
        bag_fea = ([], [])
        for ins_idx in ins_idxs:
            bag_fea[0].append(ins_fea[ins_idx])
            bag_fea[1].append(bags_label[ins_idx])
        bags_fea.append(bag_fea)

    # random select 90% bags as train, others as test
    num_bag = len(bags_fea)
    kf = KFold(n_folds, shuffle=True, random_state=None)
    datasets = []
    for train_idx, test_idx in kf.split(bags_fea):
        dataset = {}
        dataset['train'] = [bags_fea[ibag] for ibag in train_idx]
        dataset['test'] = [bags_fea[ibag] for ibag in test_idx]
        datasets.append(dataset)
    return datasets


# save batches, per batch contains instance features of a bag and bag label
def convertToBatch(bags):
    """Convert to batch format.
    Parameters
    -----------------
    bags : list
        A list contains instance features of bags and bag labels.
    Return
    -----------------
    data_set : list
        Convert dataset to batch format(instance features, bag label).
    """
    batch_num = len(bags)
    data_set = []
    for ibag, bag in enumerate(bags):
        batch_data = np.asarray(bag[0], dtype='float32')
        batch_label = np.asarray(bag[1])
        data_set.append((batch_data, batch_label))
    return data_set

if __name__ == "__main__":
    model = MI_Net()
    lossFunc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.001)
    optimizer.zero_grad()

    datasets = load_dataset("birds",10)
    losses = []
    acc = []

    for i in range(10):
        dataset = datasets[i]
        train_bag = dataset['train']
        test_bag = dataset['test']

        train_set = convertToBatch(train_bag)
        test_set = convertToBatch(test_bag)

        for epoch in range(20):

            for index in range(len(train_set)):
                singleBag = train_set[index]
                x = singleBag[0]
                y = singleBag[1]

                x = torch.tensor(x)
                y = torch.LongTensor(y)
                #y = torch.transpose(y,0,1)

                y_pred = model(x)
                #y_ans = model.predict(x)

                #print(y_pred)
                #print(torch.tensor([y[0]]))

                loss1 = lossFunc(y_pred[0],torch.tensor([y[0]]))
                loss2 = lossFunc(y_pred[1],torch.tensor([y[0]]))
                loss3 = lossFunc(y_pred[2],torch.tensor([y[0]]))

                if index % 1000 == 0:
                    losses.append((loss1 + loss2 + loss3)/3)

                optimizer.zero_grad()
                loss1.backward(retain_graph=True)
                loss2.backward(retain_graph=True)
                loss3.backward(retain_graph=True)
                optimizer.step()

                #for num in range(3):

                #    loss = lossFunc(y_pred[num],torch.tensor([y[0]]))
                #    print(loss)

                #    if index % 100 == 0 and num == 0:
                #        print(loss)
                #        losses.append(loss)

                #    optimizer.zero_grad()
                #    loss.backward(retain_graph=True)
                #    optimizer.step()

        yes = 0

        for index in range(len(test_set)):
            singleBag = test_set[index]
            x = singleBag[0]
            y = singleBag[1]

            x = torch.tensor(x)
            y = torch.LongTensor(y)
            y_pred = model(x)
            y_ans = model.predict(x)

            if y[0] == 0:
                if y_ans[0] == 0:
                    yes += 1
            else:
                if y_ans[0] == 1:
                    yes += 1

        accuracy = yes/len(test_set)
        acc.append(accuracy)

    print(acc)

    plt.figure(1)
    plt.plot(losses)
    plt.ylabel("loss")
    plt.title("loss")

    plt.figure(2)
    plt.scatter(np.array([range(10)]),acc)
    plt.ylabel("acc")
    plt.title("accuracy")
    plt.show()

