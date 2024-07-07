########################################################################
########################################################################
##                                                                    ##
##                      ORIGINAL _ DO NOT PUBLISH                     ##
##                                                                    ##
########################################################################
########################################################################

import torch
from torch.nn.functional import pad
import torch.nn as nn
import numpy as np
import loader as ld
import matplotlib.pyplot as plt

batch_size = 32
output_size = 2
hidden_size = 32  # to experiment with

run_recurrent = False  # else run Token-wise MLP
use_RNN = False  # otherwise GRU
atten_size = 5  # atten > 0 means using restricted self atten

reload_model = True
num_epochs = 15
learning_rate = 0.001
test_interval = 50

# Loading sataset, use toy = True for obtaining a smaller dataset

train_dataset, test_dataset, num_words, input_size = ld.get_data_set(batch_size, toy=True)


# Special matrix multipication layer (like torch.Linear but can operate on arbitrary sized
# tensors and considers its last two indices as the matrix.)

class MatMul(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True):
        super(MatMul, self).__init__()
        self.matrix = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(in_channels, out_channels)),
                                         requires_grad=True)
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(1, 1, out_channels), requires_grad=True)

        self.use_bias = use_bias

    def forward(self, x):
        x = torch.matmul(x, self.matrix)
        if self.use_bias:
            x = x + self.bias
        return x


class ExRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExRNN, self).__init__()

        self.hidden_size = hidden_size
        self.sigmoid = torch.sigmoid

        # RNN Cell weights
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden2out = nn.Linear(hidden_size, output_size)

    def name(self):
        return "RNN"

    def forward(self, x, hidden_state):
        combined = torch.cat((x, hidden_state), 1)
        hidden = self.sigmoid(self.in2hidden(combined))
        output = self.sigmoid(self.hidden2out(hidden))
        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size)


class ExGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExGRU, self).__init__()
        self.hidden_size = hidden_size
        # GRU Cell weights
        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def name(self):
        return "GRU"

    def forward(self, x, hidden_state):
        z = self.sigmoid(self.W_z(torch.cat((x, hidden_state), 1)))
        r = self.sigmoid(self.W_r(torch.cat((x, hidden_state), 1)))
        h_tilde = self.tanh(self.W_h(torch.cat((x, r * hidden_state), 1)))
        hidden = (1 - z) * hidden_state + z * h_tilde
        output = self.out(hidden)

        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size)


class ExMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExMLP, self).__init__()

        self.ReLU = torch.nn.ReLU()

        # Token-wise MLP network weights
        self.layer1 = MatMul(input_size, hidden_size)
        self.layer2 = MatMul(hidden_size, output_size)
        # additional layer(s)

    def name(self):
        return "MLP"

    def forward(self, x):
        # Token-wise MLP network implementation

        x = self.layer1(x)
        x = self.ReLU(x)
        x = self.layer2(x)
        # x = self.ReLU(x)
        return x


class ExRestSelfAtten(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExRestSelfAtten, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.sqrt_hidden_size = np.sqrt(float(hidden_size))
        self.ReLU = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(2)

        # Token-wise MLP + Restricted Attention network implementation

        self.layer1 = MatMul(input_size, hidden_size)
        self.W_q = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_k = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_v = MatMul(hidden_size, hidden_size, use_bias=False)
        self.out = MatMul(hidden_size, output_size)

        self.positional_encoding = nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty(1, 2 * atten_size + 1, hidden_size)), requires_grad=True)

    def name(self):
        return "MLP_atten"

    def forward(self, x):
        # Token-wise MLP + Restricted Attention network implementation

        x = self.layer1(x)
        x = self.ReLU(x)

        # Add positional encoding
        seq_len = x.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0).unsqueeze(-1)
        position_encoding = self.positional_encoding[:, :seq_len, :]
        x = x + position_encoding

        # generating x in offsets between -atten_size and atten_size 
        # with zero padding at the ends

        padded = pad(x, (0, 0, atten_size, atten_size, 0, 0))

        x_nei = []
        for k in range(-atten_size, atten_size + 1):
            x_nei.append(torch.roll(padded, k, 1))

        x_nei = torch.stack(x_nei, 2)
        x_nei = x_nei[:, atten_size:-atten_size, :]

        # x_nei has an additional axis that corresponds to the offset

        # Applying attention layer

        query = self.W_q(x).unsqueeze(2)
        keys = self.W_k(x_nei)
        vals = self.W_v(x_nei)
        attention_weights = self.softmax(torch.matmul(query, keys.transpose(-1, -2)) / self.sqrt_hidden_size)
        attention_output = torch.matmul(attention_weights, vals).squeeze(2)
        output = self.out(attention_output)

        return output, atten_weights


class ExMLPWithAttention(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, atten_size):
        super(ExMLPWithAttention, self).__init__()
        self.attention_layer = ExRestSelfAtten(input_size, hidden_size, atten_size)
        self.mlp_layer = ExMLP(hidden_size, output_size, hidden_size)

    def name(self):
        return "MLP_with_Attention"

    def forward(self, x):
        attention_output, attention_weights = self.attention_layer(x)
        sub_scores = self.mlp_layer(attention_output)
        return sub_scores, attention_weights

# prints portion of the review (20-30 first words), with the sub-scores each work obtained
# prints also the final scores, the softmaxed prediction values and the true label values

def print_review(rev_text, sbs1, sbs2, lbl1, lbl2):
    print(rev_text[:30])
    print("Sub-scores:")
    for i in range(min(len(rev_text), 10)):
        print(f"word={rev_text[i]}, score of pos={sbs1[i].item()}, score of neg={sbs2[i].item()}")
    print("\nFinal scores:")
    print(f"Final scores: pos={sbs1.mean().item()}, neg={sbs2.mean().item()}")
    print(f"prediction={torch.nn.functional.softmax(torch.tensor([sbs1.mean(), sbs2.mean()]), 0).tolist()}")
    print(f"True label: pos={lbl1.item()}, neg={lbl2.item()}")
    print("\n")


# select model to use

if run_recurrent:
    if use_RNN:
        model = ExRNN(input_size, output_size, hidden_size)
    else:
        model = ExGRU(input_size, output_size, hidden_size)
else:
    if atten_size > 0:
        model = ExMLPWithAttention(input_size, output_size, hidden_size, atten_size)
    else:
        model = ExMLP(input_size, output_size, hidden_size)

print("Using model: " + model.name())

if reload_model:
    print("Reloading model")
    model.load_state_dict(torch.load(model.name() + ".pth"))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss = 1.0
test_loss = 1.0

train_accuracies = []
test_accuracies = []
# training steps in which a test step is executed every test_interval

for epoch in range(num_epochs):

    itr = 0  # iteration counter within each epoch
    epoch_test_accuracies = []
    epoch_train_accuracies = []
    for labels, reviews, reviews_text in train_dataset:  # getting training batches

        itr = itr + 1

        if (itr + 1) % test_interval == 0:
            test_iter = True
            labels, reviews, reviews_text = next(iter(test_dataset))  # get a test batch
        else:
            test_iter = False

        # Recurrent nets (RNN/GRU)
        sub_score = []
        if run_recurrent:
            hidden_state = model.init_hidden(int(labels.shape[0]))

            for i in range(num_words):
                output, hidden_state = model(reviews[:, i, :], hidden_state)  # HIDE

        else:

            # Token-wise networks (MLP / MLP + Atten.)


            if atten_size > 0:
                # MLP + atten
                sub_score, atten_weights = model(reviews)
            else:
                # MLP
                sub_score = model(reviews)

            output = torch.mean(sub_score, 1)

        # cross-entropy loss

        loss = criterion(output, labels)
        accuracy = (output.argmax(1) == labels.argmax(1)).float().mean()

        # optimize in training iterations

        # if not test_iter:
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        # averaged losses
        if test_iter:
            test_loss = 0.8 * float(loss.detach()) + 0.2 * test_loss
            epoch_test_accuracies.append(accuracy)
        else:
            train_loss = 0.9 * float(loss.detach()) + 0.1 * train_loss
            if (itr - 1) % 50 == 0:
                epoch_train_accuracies.append(accuracy)

        if test_iter:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Step [{itr + 1}/{len(train_dataset)}], "
                f"Train Loss: {train_loss:.4f}, "
                f"Test Loss: {test_loss:.4f}"
            )

            if not run_recurrent:
                nump_subs = sub_score.detach()
                labels = labels.detach()
                print_review(reviews_text[0], nump_subs[0, :, 0], nump_subs[0, :, 1], labels[0, 0], labels[0, 1])

    # accuracy = epoch_accuracy / len(train_dataset)
    test_accuracies.append(sum(epoch_test_accuracies) / len(epoch_test_accuracies))
    train_accuracies.append(sum(epoch_train_accuracies) / len(epoch_train_accuracies))

# saving the model
torch.save(model.state_dict(), model.name() + ".pth")

fig, ax = plt.subplots()
ax.plot(train_accuracies, label='Train accuracy')
ax.plot(test_accuracies, label='Test accuracy')
ax.legend()
plt.title(f"{model.name()}")
plt.show()
