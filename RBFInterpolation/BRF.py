# -*- coding:utf-8 -*-
# author:XueWang
import numpy as np
import argparse
import matplotlib.pyplot as plt

#参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--max_inte_times', type=int, default=1, help='max gradient descent interation times')
parser.add_argument('--gradient_stop', type=float, default=0.000001, help='when the gradient less than gradientStop, stop')
parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
#parser.add_argument('--sigma', type=float, default=1,nargs='+', help='Φ sigma value')
parser.add_argument('--batch_num', type=int, default=50, help='batch num in the gradient descent')

Args = parser.parse_args()
MaxInteration = Args.max_inte_times
GradientStop = Args.gradient_stop
Learning_rate = Args.learning_rate
#Sigma = Args.sigma
SigmaList =  [0.5,0.7,1.0,2]
print(SigmaList)
Batch_Num = Args.batch_num


def readData(filename):
	datafile = np.loadtxt(filename,dtype=np.str, delimiter=",")
	data = datafile.astype(np.float)
	return data

#二范式距离平方
def norm_distance(x0,x1):
	#print(np.sum(np.multiply((x0 - x1),(x0-x1))))
	return np.linalg.norm(x0-x1, ord=2, axis=0)**2
	#return np.sum(np.multiply((x0 - x1),(x0 - x1)))

#200 小样本直接求逆矩阵
def inverse_solve(traindata,trainlabel,sigma):
	# 初始化200维 alpha向量和 200*200的fai矩阵
	shape = traindata.shape[0]
	fai = np.zeros((shape,shape))
	#Φ矩阵
	for i in range(shape):
		for j in range(shape):
			fai[i][j] = np.e**( - sigma * sigma * norm_distance(traindata[i,0:],traindata[j,0:])) #使用高斯函数
	alpha = np.dot( np.linalg.inv(fai) , trainlabel)
	return alpha

#梯度下降法求解alpha,初始化alpha
def descent(traindata,trainlabel,alpha,sigma):
	#多少批样本数据
	Batchs = traindata.shape[0] // Batch_Num
	#对全部数据迭代 MaxInteration 次
	for i in range(MaxInteration):
		#全部样本数据计算一次,每一批
		#loss = 0.0
		for j in range(Batchs):
			# 每一批的Φ矩阵
			fai = np.zeros((Batch_Num, traindata.shape[0]))
			for m in range(Batch_Num):
				for n in range(traindata.shape[0]):
					fai[m][n] = np.e**( - sigma * sigma * norm_distance(traindata[Batch_Num * j + m,0:],traindata[n,0:]))
			Y_estimate = np.dot(fai,alpha)
			#print(Y_estimate,'batch:',j,'y----\n')
			#所有样本数据的误差，每一批loss求和
			#loss = 1/2 * np.sum( norm_distance(Y_estimate,trainlabel[Batch_Num * j: Batch_Num * (j+1) , 0:]) ) + loss

			#update the gradient
			delta_alpha = np.dot( fai.transpose(), Y_estimate - trainlabel[Batch_Num * j: Batch_Num * (j+1) , 0:] )
			alpha = alpha - Learning_rate * delta_alpha
			# for row in range(delta_alpha.shape[0]):
			# 	if(delta_alpha[row][0] > 0.001):
			# 		alpha[row][0] = alpha[row][0] - Learning_rate * delta_alpha
			print(alpha[0:20,0:])

		#全部样本的误差为0，停止计算，目前还未用到
		# print('loss:\n', loss)
		# if loss == 0.0:
		# 	break
	return alpha

#插值计算结果
def IMQ_Interpolation(testdata,traindata,alpha,sigma):
	fai = np.zeros((testdata.shape[0],traindata.shape[0]))
	for i in range(testdata.shape[0]):
		for j in range(traindata.shape[0]):
			# 计算test值的 Φ矩阵
			fai[i][j] = np.e**( - sigma * sigma * norm_distance(testdata[i,0:],traindata[j,0:]))
			#print(fai[i][j])
	Y_estimate = np.dot(fai,alpha)
	#print(Y_estimate.shape)
	return Y_estimate

#1个y值的LOOCV误差计算
def Loocv_loss(traindata,traindata_temp,trainlabel,alpha,sigma):
	fai = np.zeros((1,traindata_temp.shape[0]))
	for i in range(traindata_temp.shape[0]):
		fai[0][i] = np.e**( - sigma * sigma * norm_distance(traindata,traindata_temp[i,0:]))
	Y_estimate = np.dot( fai, alpha )
	error = (trainlabel - Y_estimate)**2
	return error[0][0]


def main():
	'''''
	#任务1 默认sigma = 5.0
	traindata = readData('a3data1')
	trainlabel = readData('Y1.txt')
	traindata = np.array(traindata)
	trainlabel = np.array(trainlabel)
	trainlabel = np.expand_dims(np.array(trainlabel), axis=1)
	#print(trainlabel[0,0:],'\n')
	# 插值计算
	alpha = inverse_solve(traindata,trainlabel,sigma=5.0)
	testdata = readData('a3data1t')
	res = IMQ_Interpolation(testdata, traindata, alpha, sigma=5.0)
	# 插值结果画图
	plt.plot(res)
	plt.show()

	#任务2
	errorlist = [] #不同的sigma的全部LOOCV误差（留一法）
	for sigma in SigmaList:
		errors = 0.0
		for i in range(traindata.shape[0]):
			#去除i个x y值的训练集
			traindata_temp = np.delete(traindata,i,axis=0)
			trainlabel_temp = np.delete(trainlabel,i,axis=0)
			# 求逆得到alpha
			alpha = inverse_solve(traindata_temp,trainlabel_temp,sigma) #重新训练alpha
			#每一个Xi,Yi的LOOCV误差
			errors = Loocv_loss(traindata[i,0:],traindata_temp,trainlabel[i,0:],alpha,sigma) + errors
		print(errors/traindata.shape[0])
		errorlist.append(errors/traindata.shape[0])

	plt.plot(SigmaList,errorlist)
	plt.show()
	
	'''''
	#任务3，梯度下降
	#读取2的数据
	traindata2 = readData('a3data2')
	trainlabel2 = readData('Y2.txt')
	trainlabel2 = np.expand_dims(np.array(trainlabel2),axis=1)
	#init alpha 1
	alpha2 = np.ones((traindata2.shape[0],1))
	alpha2 = descent(traindata2,trainlabel2,alpha2,sigma=10)

	testdata2 = readData('a3data2t')
	res2 = IMQ_Interpolation(testdata2,traindata2,alpha2)

	# 画图
	plt.plot(res2)
	plt.show()



if __name__ == '__main__':
	main()