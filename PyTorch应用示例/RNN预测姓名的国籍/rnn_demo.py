"""
训练一个RNN将人名进行国籍分类
"""
from io import open
import glob
import unicodedata
import string
import math
import os
import time
import torch.nn as nn
import torch
import random
import matplotlib.pyplot as plt
import torch.utils.data
from tools.common_tools import set_seed

set_seed(1)  # 设置随机种子
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters)


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def randomChoice(l): # 在列表l中随机选择一个
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)                 # 所有类别中随机选一个
    line = randomChoice(category_lines[category])           # 这个类别所有样本中随机选一个
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)    # str to one-hot
    return category, line, category_tensor, line_tensor # 返回类别，名字样本，类别张量，名字张量


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))


def get_lr(iter, learning_rate):
    lr_iter = learning_rate if iter < n_iters else learning_rate*0.1
    return lr_iter

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.u = nn.Linear(input_size, hidden_size)
        self.w = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, output_size)

        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden):

        u_x = self.u(inputs)

        hidden = self.w(hidden)
        hidden = self.tanh(hidden + u_x)

        output = self.softmax(self.v(hidden))

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def train(category_tensor, line_tensor):
    hidden = rnn.initHidden() # 初始的全0张量

    rnn.zero_grad() # 清空梯度，因为这个循环网络不止使用一次，会有之前使用遗留的梯度

    line_tensor = line_tensor.to(device)         # Xt
    hidden = hidden.to(device)                   # W
    category_tensor = category_tensor.to(device) # 预期输出（正确类别张量）

    for i in range(line_tensor.size()[0]): # RNN循环遍历字符串张量，即前向传播
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor) # 计算损失函数
    loss.backward() # 反向传播

    # 使用负梯度进行更新参数
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() # 返回预测值与Loss


if __name__ == "__main__":
    # config
    path_txt = os.path.join("data", "names", "*.txt") # 使用通配符匹配文件名
    all_letters = string.ascii_letters + " .,;'" # 所有的大小写以及三种标点
    n_letters = len(all_letters)    # 52 + 5 = 57 字符总类别数
    print_every = 5000
    plot_every = 5000
    learning_rate = 0.005
    n_iters = 200000

    # step 1 读入数据
    # Build the category_lines dictionary, a list of names per language
    category_lines = {} # 字典用于存放所有类别的数据
    all_categories = [] # 存放所有类的类名
    for filename in glob.glob(path_txt):
        category = os.path.splitext(os.path.basename(filename))[0] # Chinese.txt => Chinese
        all_categories.append(category) # 添加类名
        lines = readLines(filename)     # 将文件读入并转换为Ascii编码，用列表保存每行（每个姓名）
        category_lines[category] = lines

    n_categories = len(all_categories)  # 类别数

    # step 2 实例化模型
    n_hidden = 128
    rnn = RNN(n_letters, n_hidden, n_categories)

    rnn.to(device)

    # step 3 实例化损失函数，负对数似然分类损失
    criterion = nn.NLLLoss()

    # step 4 之后手动优化，不定义优化器，所以这个部分空着

    # step 5 迭代更新
    current_loss = 0
    all_losses = []
    start = time.time()
    for iter in range(1, n_iters + 1):
        # 随机选取一个样本
        category, line, category_tensor, line_tensor = randomTrainingExample()

        # training，包括前向、反向传播、更新参数
        output, loss = train(category_tensor, line_tensor)

        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output) # 当前这个样本的预测类别
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('Iter: {:<7} time: {:>8s} loss: {:.4f} name: {:>10s}  pred: {:>8s} label: {:>8s}'.format(
                iter, timeSince(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
            
path_model = "rnn_state_dict.pkl"
torch.save(rnn.state_dict(), path_model)
plt.plot(all_losses)
plt.show()

predict('Chen Yixiong')
predict('Chen yixiong')
predict('chenyixiong')

predict('test your name')



