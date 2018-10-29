# -*- coding:utf-8 -*-
# author:XueWang
import numpy as np
import random
import argparse
import matplotlib
import matplotlib.pyplot as plt

#parameters explaination
parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, default=0.01, help='step length')
parser.add_argument('--max_interation', type=int, default=100000, help='max gradient descent interation times')
parser.add_argument('--gradient_stop', type=float, default=0.001, help='when the gradient less than gradientStop, stop interate')
parser.add_argument('--gama', type=float, default=0.009, help='resistance factor in momentum')
parser.add_argument('--epsilon', type=float, default=0.00001, help='avoid 0 gradient')
Args = parser.parse_args()

ALPHA = Args.alpha
MaxInteration = Args.max_interation
GradientStop = Args.gradient_stop
GAMA = Args.gama
Epsilon = Args.epsilon

"""
随机梯度下降算法，从样本中随机抽取一个数更新梯度
优化算法使用推荐技巧更新参数
"""
def SGD_TIPS(x,y,theta):
	v = np.array([.0, .0])
	u = np.array([.0, .0])
	w = np.array([.0, .0])
	delta_theta = np.array([.0,.0])
	for i in range(0,MaxInteration):
		#随机取出Sample中的一个
		RandomIndex = random.randint(0,999)
		#计算梯度 推荐技巧,first 预先先走一步
		theta = theta - GAMA * v
		# 回归模型
		hypothesis = theta[0] + theta[1] * x[RandomIndex]
		# 损失差值函数
		loss = hypothesis - y[RandomIndex]
		#走一步后的梯度
		gradient = np.array([2 * loss , 2 * loss * x[RandomIndex]])
		#判断梯度是否很小
		if((gradient[0]*gradient[0] + gradient[1]*gradient[1])**0.5 < GradientStop):
			print('gradient stop:',(gradient[0]*gradient[0] + gradient[1]*gradient[1])**0.5 )
			break
		#计算u v w delta_theta
		v = GAMA*v + (1-GAMA) * gradient
		u = GAMA*u + (1-GAMA) * gradient * gradient
		delta_theta = - np.sqrt( (w + np.array([Epsilon,Epsilon])) ) * v / np.sqrt((u + np.array([Epsilon,Epsilon])))
		print('delta_theta',delta_theta)
		w = GAMA*w + (1-GAMA) * delta_theta * delta_theta
		#更新参数
		theta = theta + delta_theta
	return  theta


"""
小批量梯度细节算法，从样本中随机选取部分数据更新梯度
类似于SGD，不过梯度变为M个样本的均值
"""

def ReadData():
	data = np.loadtxt("xy.csv",dtype=np.str,delimiter=",")
	traindata = data[0:,0].astype(np.float)
	trainlabel = data[0:,1].astype(np.float)
	return traindata,trainlabel

def Draw(x,y):
	f1 = plt.figure(1)
	plt.scatter(x,y)
	f1.show()

def main():
	traindata,trainlabel = ReadData()
	# Draw(traindata,trainlabel)
	theta = np.array([0.1, 0.1])
	theta = SGD_TIPS(traindata,trainlabel,theta)
	# theta = MiniBatchGradientDescent(traindata,trainlabel)
	print('y=',theta[0],'+ (',theta[1],') * x')

if __name__ == "__main__":
	main()