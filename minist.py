import numpy as np
import scipy.special
import time
from matplotlib import pyplot as plt


class NN:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, lr):
        # 输入节点个数 隐藏层节点个数 输出层节点个数 学习率
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.lr = lr
        '''初始化权重矩阵，我们有两个矩阵：一个是weight_itoh表示输入层与中间层之间的链路权重形成的矩阵
        一个是weight_htoo表示中间层与输出层之间的链路权重形成的矩阵'''
        np.random.seed(2)
        self.weight_itoh = np.random.rand(self.hidden_nodes, self.input_nodes)-0.5
        self.weight_htoo = np.random.rand(self.output_nodes, self.hidden_nodes)-0.5
        '''sigmoid激活函数：激活每个节点，得到结果作为信号输入到下一层'''
        self.activation = lambda x: scipy.special.expit(x)

    '''接着我们先看forward函数的实现，它接收输入数据，通过神经网络的层层计算后，在输出层输出最终结果。
    输入数据要依次经过输入层，中间层，和输出层，并且在每层的节点中还得执行激活函数以便形成对下一层节点的输出信号。
    我们知道可以通过矩阵运算把这一系列复杂的运算流程给统一起来。'''
    def test(self, x):
        # 根据输入数据计算并输出答案
        # 计算中间层从输入层接收到的信号量
        hidden_input = np.dot(self.weight_itoh, x)
        # 计算中间层经过激活函数后形成的输出信号量
        hidden_output = self.activation(hidden_input)  # 激活
        # 计算结束层从中间层接受的信号量
        final_input = np.dot(self.weight_htoo, hidden_output)
        # 计算最外层神经元（结束层）经过激活函数后形成的输出信号量
        finnal_output = self.activation(final_input)
        return finnal_output

    def train(self, train_data, train_label):
        # forward 正向传播
        hidden_input = np.dot(self.weight_itoh, train_data)
        hidden_output = self.activation(hidden_input)
        finnal_input = np.dot(self.weight_htoo, hidden_output)
        final_output = self.activation(finnal_input)
        # calculate error 计算每层的误差
        output_errors = train_label - final_output
        loss = sum(output_errors ** 2 / len(train_data))
        hidden_errors = np.dot(self.weight_htoo.T, output_errors * final_output * (1 - final_output))
        # backward 反向传播
        self.weight_htoo += self.lr * np.dot((output_errors * final_output * (1 - final_output)), np.transpose(hidden_output))
        self.weight_itoh += self.lr * np.dot((hidden_errors * hidden_output * (1 - hidden_output)), np.transpose(train_data))
        return loss


if __name__ == '__main__':

    input_nodes = 784
    hidden_nodes = 250
    output_nodes = 10
    epochs = 30
    lr = 0.1

    datafile = open("./dataset/mnist_train.csv")
    data = datafile.readlines()
    datafile.close()

    net = NN(input_nodes, hidden_nodes, output_nodes, lr)

    print('########### Start Training ###########')
    total_loss = []
    acc = []
    for epoch in range(epochs):
        losses = []
        time.sleep(0.5)
        for record in data:
            # 预处理数据
            # 归一化：/255防止为0链路更新出现问题，+0.01使结果处于 0.01~1 中
            # 转换为numpy支持的二维矩阵
            label_and_input = record.split(',')
            train_data = np.asfarray(label_and_input[1:])/255.0 * 0.99 + 0.01
            train_data = np.array(train_data, ndmin=2).T  # shape(784,1)

            # label 设置图片与数值的对应关系
            train_label = np.zeros(output_nodes)
            train_label[int(label_and_input[0])] = 1
            train_label = np.array(train_label, ndmin=2).T  # shape(10,1)

            loss = net.train(train_data=train_data, train_label=train_label)
            losses.append(loss)
        total_loss.append(sum(losses))
        print('epoch ' + str(epoch) + ' loss:', sum(losses))

        # 测试
        datafile = open("./dataset/mnist_test.csv")
        data = datafile.readlines()
        datafile.close()
        score = []

        print('########### Evaluation epoch', epoch, '###########')
        for record in data:
            # 预处理数字图片
            all_value = record.split(',')
            input = np.asfarray(all_value[1:])/255*0.99+0.01  # shape(784,)
            input = np.array(input, ndmin=2).T  # shape(784,1)

            correct_number = int(all_value[0])
            # print('Truth:', correct_number)
            # 网络推理数字图片 对应的数字
            predict = net.test(input)
            # 找到数值最大的神经元对应的编号
            predict = np.argmax(predict)
            # 这里的argmax返回的是索引，只不过此时索引和图片数字相同，都是0~9，所以输出网络预测可以用预测的索引
            # print('predict:', predict)
            if predict == correct_number:
                score.append(1)
            else:
                score.append(0)
        correct_rate = np.array(score).sum()/len(score)
        print(score)
        print('correct rate:', correct_rate)
        acc.append(correct_rate)

    # 保存模型
    file = open('./model.pth', mode='w')
    file.write(str(net.weight_itoh))
    file.write(str(net.weight_htoo))
    file.close()

    # 绘制损失 验证曲线
    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    plt.plot(total_loss)
    plt.xlabel('loss')
    plt.subplot(122)
    plt.plot(acc)
    plt.xlabel('correct_rate')
    plt.show()
