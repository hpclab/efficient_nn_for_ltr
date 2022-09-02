


import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dims = [400, 200, 200, 100], drop_prob = 0.1, batch_norm = False):
        super(MLP, self).__init__()

        self.n_layers = len(hidden_dims)
        self.input_dim = input_dim

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(1, self.n_layers):
            self.layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))

        self.layers.append(nn.Linear(hidden_dims[-1], 1))

        self.relu6 = nn.ReLU6()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.relu6(layer(x))
            if i == 0:
                x = self.dropout(x)
        out = self.layers[-1](x)
        return torch.squeeze(out, 1)



"""

class MLP(nn.Module):
    def __init__(self, input_size=136, n_layers = 4, layers_sizes = [500, 500, 500, 100], p=0.0):

        super(MLP, self).__init__()
        self.n_layers = n_layers

        assert len(layers_sizes) == n_layers, "Wrong parameters to construct the MLP"
        assert len(layers_sizes) > 1, "Provide at least two layers"

        self.fcs = []
        first_layer = nn.Linear(input_size, layers_sizes[0])
        self.fcs.append(first_layer)

        self.add_module("fc"+ str(0), self.fcs[0])
        for i in range(1,n_layers):
            self.fcs.append(nn.Linear(layers_sizes[i-1], layers_sizes[i]))
            self.add_module("fc" + str(i), self.fcs[i])
        self.last_layer = nn.Linear(layers_sizes[i], 1)
        self.fcs.append(self.last_layer)
        self.add_module("fc" + str(self.last_layer), self.last_layer)

        self.relu = nn.ReLU()
        self.relu6 = nn.ReLU6()
        self.dropout = nn.Dropout(p=p)


    def forward(self, x):
        for i in range(self.n_layers):
            x = self.fcs[i](x)
            if (i==0):
                x = self.dropout(x)
            x = self.relu6(x)
        x = self.last_layer(x)#no relu
        return torch.squeeze(x, 1)

"""

#LARGE MLP and SMALL MLP are maintained for compatibility with early experiments

class largeMLP(nn.Module):
    def __init__(self, input_size=136, h1=2000, h2=500, h3=500, h4=100, p=0.5, batch_norm = False):
        print("Large Netork")

        super(largeMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.fc5 = nn.Linear(h4, 1)
        self.relu = nn.ReLU()
        self.relu6 = nn.ReLU6()
        self.dropout = nn.Dropout(p=p)
        self.sigmoid = nn.Sigmoid()
        print("Dropout:", p)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu6(out)

        out = self.fc2(out)
        #out = self.dropout(out)
        out = self.relu6(out)

        out = self.fc3(out)
        #out = self.dropout(out)
        out = self.relu6(out)

        out = self.fc4(out)

        out = self.dropout(out)
        out = self.relu6(out)

        out = self.fc5(out)  # no relu on the output
        #out = self.sigmoid(out)
        return torch.squeeze(out, 1)


class smallMLP(nn.Module):

    def __init__(self, input_size=136, h1=500, h2=100, p=0.):
        print("Small Network")

        super(smallMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)
        self.relu6 = nn.ReLU6()
        print ("Dropout:", p)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)

        out = self.relu6(out)

        out = self.fc2(out)
        out = self.dropout(out)

        out = self.relu6(out)

        out = self.fc3(out)  # no relu on the output
        return torch.squeeze(out, 1)


class smallMLP_PreAct(nn.Module):

    def __init__(self, input_size=136, h1=500, h2=100, p=0.):
        print("Small Network")

        super(smallMLP_PreAct, self).__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)
        self.relu6 = nn.ReLU6()
        print ("Dropout:", p)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        out = self.dropout(x)

        out = self.relu6(out)
        out = self.fc1(out)

        out = self.dropout(out)

        out = self.relu6(out)
        out = self.fc2(out)


        out = self.fc3(out)  # no relu on the output
        return torch.squeeze(out, 1)