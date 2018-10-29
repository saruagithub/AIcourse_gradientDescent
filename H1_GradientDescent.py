# -*- coding:utf-8 -*-
# author:XueWang
import numpy as np
import random
import argparse

#parameters explaination
parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, default=0.01, help='step length')
parser.add_argument('--max_interation', type=int, default=100000, help='max gradient descent interation times')
parser.add_argument('--gradient_stop', type=float, default=0.001, help='when the gradient less than gradientStop, stop interate')

Args = parser.parse_args()

ALPHA = Args.alpha
MaxInteration = Args.max_interation
GradientStop = Args.gradient_stop

"""
随机梯度下降算法，从样本中随机抽取一个数更新梯度
"""
def StotisticGradientDescent(x,y,theta1,theta2):
	for i in range(0,MaxInteration):
		#随机取出Sample中的一个
		RandomIndex = random.randint(0,999)
		#回归模型
		hypothesis = theta1 + theta2 * x[RandomIndex]
		#损失差值函数
		loss = hypothesis - y[RandomIndex]
		#计算梯度
		gradient1 = 2 * loss
		gradient2 = 2 * x[RandomIndex] * loss
		Stopflag = (gradient1**2 + gradient2**2)**0.5
		print(i,' SGD loss is: ', loss,'gradient is:', Stopflag,'\n')
		if (Stopflag < GradientStop):
			break
		#更新
		theta1 = theta1 - ALPHA * gradient1
		theta2 = theta2 - ALPHA * gradient2
	return  theta1,theta2


"""
小批量梯度细节算法，从样本中随机选取部分数据更新梯度
"""
def MiniBatchGradientDescent(x,y,theta1,theta2):
	for i in range(0,MaxInteration):
		gradient1 = 0.0
		gradient2=0.0
		loss = 0.0
		# M can be change
		M = random.randint(50,100)
		for j in range(0,M):
			RandomIndex = random.randint(0,999)
			hypothesis = theta1 + theta2 * x[RandomIndex]
			loss = hypothesis - y[RandomIndex]
			gradient1 = 2 * loss + gradient1
			gradient2 = 2 * x[RandomIndex] * loss + gradient2
		Stopflag = ((gradient1/M) * (gradient1/M) + (gradient2/M) * (gradient2/M)) ** 0.5
		print(i,'M is:',M, ' MSGD loss is: ', loss, 'gradient is:', Stopflag, '\n')
		if (Stopflag < GradientStop):
			break
		# 更新
		theta1 = theta1 - ALPHA * (gradient1/M)
		theta2 = theta2 - ALPHA * (gradient2/M)
	return theta1,theta2



def ReadData():
	data = np.loadtxt("xy.csv",dtype=np.str,delimiter=",")
	traindata = data[0:,0].astype(np.float)
	trainlabel = data[0:,1].astype(np.float)
	return traindata,trainlabel


def main():
	traindata,trainlabel = ReadData()
	theta1 = 1.0
	theta2 = 1.0
	# theta1,theta2 = StotisticGradientDescent(traindata,trainlabel,theta1,theta2)
	theta1,theta2 = MiniBatchGradientDescent(traindata,trainlabel,theta1,theta2)
	print('y=',theta1,'+',theta2,'* x')

if __name__ == "__main__":
	main()