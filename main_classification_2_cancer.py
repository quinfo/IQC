# -*- coding: utf-8 -*-
from sklearn.metrics import (accuracy_score, recall_score, f1_score, precision_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import pandas as pd
from qutip import *
from util import *
import math
import warnings
import logging
import time
import os,sys

###############全局信息设定###############
warnings.filterwarnings("ignore")

dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))


logger = logging.getLogger('dataset')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('./log/'+filename.replace('.py', '.log'), 'a')
logger.addHandler(handler)
time_str = time.strftime("%Y %b %d %H:%M:%S", time.localtime())
logger.info("操作时间:> %s" % (time_str))


################公共变量##################
batch = 1
rate = 0.015
bias = 0
starVal = 0.01



logger.info("batch:%d rate:%f starVal:%f bias:%f" % (batch, rate, starVal, bias))
################读取数据##################

cancer = datasets.load_breast_cancer()
X = get_map_data(cancer.data)
Y = cancer.target

logger.info("数据集信息: %s %s" % (str(X.shape), str(Y.shape)))
print('\n乳腺癌数据集:>', cancer.data.shape,
		'\n各类数据量:>',
		list(cancer.target).count(0),
	  	list(cancer.target).count(1))




##############准备正确的结果##############
results = np.zeros(Y.shape[0])
skf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)# 会因参数调整


################程序执行##################
for k, (train_index, test_index) in enumerate(skf.split(X, Y)):


	# 准备参数(有调控的作用)
	param = np.ones((1, 30)) * starVal # 会因参数调整
	param_Best = None
	param_Flag = 0
	y = Y[train_index]

	# 准备量子态
	psi_sys = (basis(2, 0) + basis(2, 1)).unit()
	psi_env = (basis(30, 0) + basis(30, 1) + basis(30, 2) + basis(30, 3) + basis(30, 4)
		 + basis(30, 5) + basis(30, 6) + basis(30, 7) + basis(30, 8) + basis(30, 9)
		 + basis(30, 10) + basis(30, 11) + basis(30, 12) + basis(30, 13) + basis(30, 14)
		 + basis(30, 15) + basis(30, 16) + basis(30, 17) + basis(30, 18) + basis(30, 19)
		 + basis(30, 20) + basis(30, 21) + basis(30, 22) + basis(30, 23) + basis(30, 24)
		 + basis(30, 25) + basis(30, 26) + basis(30, 27) + basis(30, 28) + basis(30, 29)).unit()
	psi = tensor(psi_sys, psi_env) # 会因参数调整


	##############训练模型#######################
	for j in range(batch):


		##########训练############################
		for i, x in enumerate(X[train_index]):
			# 处理分割产生的参数不足
			if x.shape[0] < param.shape[0]: continue


			# 准备算子
			H_sys = sigmax() + sigmay() + sigmaz()
			#H_sys = sigmam() + sigmap() + sigmax() + sigmay() + sigmaz()
			#H_sys = v1 * sigmax() + v2 * sigmay() + v3 * sigmaz()
			Hami = get_Hami(H_sys, x.reshape(1, -1), param)


			# 执行运算
			result = ptrace(Hami * psi, 0).diag().tolist()


			# 参数学习
			d = np.ones(param.shape[0]) * y[i]
			z = result.index(max(result))
			cz = 1 - max(result)

			#print(d, z, np.array((d - z)).reshape(param.shape[0], 1))

			#if max(result) - min(result) >= bias:
			#	z = result.index(max(result))
			#else:
			#	z = result.index(min(result))
				
			
			dire = np.array((d - z)).reshape(param.shape[0], 1)
			#param = param - rate * dire * max(result)**2 * x

			#param = param - cz * rate * dire * x
			param = param - rate * cz * dire * x



		##########测试############################
		yes_num = 0
		for i in train_index:
			# 准备算子
			H_sys = sigmax() + sigmay() + sigmaz()
			Hami = get_Hami(H_sys, X[i], param)

			result_list = ptrace(Hami * psi, 0).diag().tolist()
			result = result_list.index(max(result_list))

			if result == Y[i]: yes_num += 1

		#打印结果
		mean_num = yes_num / len(train_index)
		print('Batch:>', j, ' Prop:', mean_num)
		if mean_num > param_Flag:
			param_Flag = mean_num
			param_Best = param

		



	##############测试集测试#######################
	for i in test_index:
		# 准备算子
		H_sys = sigmax() + sigmay() + sigmaz()
		Hami = get_Hami(H_sys, X[i], param_Best)

		result_list = ptrace(Hami * psi, 0).diag().tolist()
		result = result_list.index(max(result_list))

		results[i] = result



	#############结果打印##################
	counter = [i for i in test_index if results[i] == Y[i]]
	print('Fold:', k, 'Prop:>',len(counter) / len(test_index), len(counter), len(test_index)-len(counter))
	logger.info("%d:> %f" % (k, len(counter) / len(test_index)))




print(results)
counter = [i for i in range(len(Y)) if results[i] == Y[i]]
print("precision_score:> %f" % (precision_score(Y, results)))
print("recall_score:   > %f" % (recall_score(Y, results)))
print("f1_score:       > %f" % (f1_score(Y, results)))
print("accuracy_score: > %f" % (accuracy_score(Y, results)))
logger.info("precision_score:> %f" % (precision_score(Y, results)))
logger.info("recall_score:   > %f" % (recall_score(Y, results)))
logger.info("f1_score:       > %f" % (f1_score(Y, results)))
logger.info("accuracy_score: > %f" % (accuracy_score(Y, results)))








#########################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.datasets.samples_generator import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, recall_score, f1_score, brier_score_loss, precision_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.svm import SVC
from sklearn.naive_bayes import *
from qutip import *
from util import *
import math
import warnings
import logging
import time
import os,sys
import csv



X = X
Y = Y

logger.info("数据集信息: %s %s" % (str(X.shape), str(Y.shape)))



##########################经典算法################################
print("\n\n#############数据集信息############")
logger.info("\n#############数据集信息############")


###############准备数据###################
KFold_val = 10


kf = KFold(n_splits=KFold_val, random_state=0, shuffle=True)
skf = StratifiedKFold(n_splits=KFold_val, random_state=0, shuffle=True)




precision_score_val = np.zeros(KFold_val)
recall_score_val  	= np.zeros(KFold_val)
f1_score_val  		= np.zeros(KFold_val)
accuracy_score_val  = np.zeros(KFold_val)



############ 不同模型下的结果 ############
print("\n朴素贝叶斯算法1 Naive Bayesian Model (NBM)")
logger.info("朴素贝叶斯算法1")
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
	model= GaussianNB()
	model.fit(X[train_index], Y[train_index])
	predict = model.score(X[test_index], Y[test_index])
	y_pred = model.predict(X[test_index])

	precision_score_val[i] 	= precision_score(Y[test_index], y_pred)
	recall_score_val[i] 	= recall_score(Y[test_index], y_pred)
	f1_score_val[i] 		= f1_score(Y[test_index], y_pred)
	accuracy_score_val[i] 	= accuracy_score(Y[test_index], y_pred)

print("均值:> \nprecision_score:%f \nrecall_score:%f \nf1_score:%f \naccuracy_score:%f"
	    % (np.mean(precision_score_val),
	    	np.mean(recall_score_val),
	    	np.mean(f1_score_val),
	    	np.mean(accuracy_score_val)))
logger.info("均值:> \nprecision_score:%f \nrecall_score:%f \nf1_score:%f \naccuracy_score:%f"
	    % (np.mean(precision_score_val),
	    	np.mean(recall_score_val),
	    	np.mean(f1_score_val),
	    	np.mean(accuracy_score_val)))
with open('./log/'+filename.replace('.py', '.csv'), 'a', newline='') as csv_file:
	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(["NBM", np.mean(precision_score_val),
						np.mean(recall_score_val),
						np.mean(f1_score_val),
						np.mean(accuracy_score_val)])







print("\n逻辑回归 logistic regressive (LR)")
logger.info("逻辑回归")
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
	model= LogisticRegression(penalty='l2')
	model.fit(X[train_index], Y[train_index])
	predict = model.score(X[test_index], Y[test_index])
	y_pred = model.predict(X[test_index])

	precision_score_val[i] 	= precision_score(Y[test_index], y_pred)
	recall_score_val[i] 	= recall_score(Y[test_index], y_pred)
	f1_score_val[i] 		= f1_score(Y[test_index], y_pred)
	accuracy_score_val[i] 	= accuracy_score(Y[test_index], y_pred)

print("均值:> \nprecision_score:%f \nrecall_score:%f \nf1_score:%f \naccuracy_score:%f"
	    % (np.mean(precision_score_val),
	    	np.mean(recall_score_val),
	    	np.mean(f1_score_val),
	    	np.mean(accuracy_score_val)))
logger.info("均值:> \nprecision_score:%f \nrecall_score:%f \nf1_score:%f \naccuracy_score:%f"
	    % (np.mean(precision_score_val),
	    	np.mean(recall_score_val),
	    	np.mean(f1_score_val),
	    	np.mean(accuracy_score_val)))
with open('./log/'+filename.replace('.py', '.csv'), 'a', newline='') as csv_file:
	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(['LR', np.mean(precision_score_val),
						np.mean(recall_score_val),
						np.mean(f1_score_val),
						np.mean(accuracy_score_val)])



print("\n随机森林 Random Forest (RF)")
logger.info("随机森林")
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
	model= RandomForestClassifier(n_estimators=8)
	model.fit(X[train_index], Y[train_index])
	predict = model.score(X[test_index], Y[test_index])
	y_pred = model.predict(X[test_index])

	precision_score_val[i] 	= precision_score(Y[test_index], y_pred)
	recall_score_val[i] 	= recall_score(Y[test_index], y_pred)
	f1_score_val[i] 		= f1_score(Y[test_index], y_pred)
	accuracy_score_val[i] 	= accuracy_score(Y[test_index], y_pred)

print("均值:> \nprecision_score:%f \nrecall_score:%f \nf1_score:%f \naccuracy_score:%f"
	    % (np.mean(precision_score_val),
	    	np.mean(recall_score_val),
	    	np.mean(f1_score_val),
	    	np.mean(accuracy_score_val)))
logger.info("均值:> \nprecision_score:%f \nrecall_score:%f \nf1_score:%f \naccuracy_score:%f"
	    % (np.mean(precision_score_val),
	    	np.mean(recall_score_val),
	    	np.mean(f1_score_val),
	    	np.mean(accuracy_score_val)))
with open('./log/'+filename.replace('.py', '.csv'), 'a', newline='') as csv_file:
	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(['RF', np.mean(precision_score_val),
						np.mean(recall_score_val),
						np.mean(f1_score_val),
						np.mean(accuracy_score_val)])




print("\n决策树分类器 Decision Tree (DT)")
logger.info("决策树分类器")
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
	model= tree.DecisionTreeClassifier()
	model.fit(X[train_index], Y[train_index])
	predict = model.score(X[test_index], Y[test_index])
	y_pred = model.predict(X[test_index])

	precision_score_val[i] 	= precision_score(Y[test_index], y_pred)
	recall_score_val[i] 	= recall_score(Y[test_index], y_pred)
	f1_score_val[i] 		= f1_score(Y[test_index], y_pred)
	accuracy_score_val[i] 	= accuracy_score(Y[test_index], y_pred)

print("均值:> \nprecision_score:%f \nrecall_score:%f \nf1_score:%f \naccuracy_score:%f"
	    % (np.mean(precision_score_val),
	    	np.mean(recall_score_val),
	    	np.mean(f1_score_val),
	    	np.mean(accuracy_score_val)))
logger.info("均值:> \nprecision_score:%f \nrecall_score:%f \nf1_score:%f \naccuracy_score:%f"
	    % (np.mean(precision_score_val),
	    	np.mean(recall_score_val),
	    	np.mean(f1_score_val),
	    	np.mean(accuracy_score_val)))
with open('./log/'+filename.replace('.py', '.csv'), 'a', newline='') as csv_file:
	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(['DT', np.mean(precision_score_val),
						np.mean(recall_score_val),
						np.mean(f1_score_val),
						np.mean(accuracy_score_val)])



print("\n梯度提升分类器 Gradient Boosting Decision Tree (GBDT)")
logger.info("梯度提升分类器")
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
	model= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0) 
	model.fit(X[train_index], Y[train_index])
	predict = model.score(X[test_index], Y[test_index])
	y_pred = model.predict(X[test_index])

	precision_score_val[i] 	= precision_score(Y[test_index], y_pred)
	recall_score_val[i] 	= recall_score(Y[test_index], y_pred)
	f1_score_val[i] 		= f1_score(Y[test_index], y_pred)
	accuracy_score_val[i] 	= accuracy_score(Y[test_index], y_pred)

print("均值:> \nprecision_score:%f \nrecall_score:%f \nf1_score:%f \naccuracy_score:%f"
	    % (np.mean(precision_score_val),
	    	np.mean(recall_score_val),
	    	np.mean(f1_score_val),
	    	np.mean(accuracy_score_val)))
logger.info("均值:> \nprecision_score:%f \nrecall_score:%f \nf1_score:%f \naccuracy_score:%f"
	    % (np.mean(precision_score_val),
	    	np.mean(recall_score_val),
	    	np.mean(f1_score_val),
	    	np.mean(accuracy_score_val)))
with open('./log/'+filename.replace('.py', '.csv'), 'a', newline='') as csv_file:
	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(['GBDT', np.mean(precision_score_val),
						np.mean(recall_score_val),
						np.mean(f1_score_val),
						np.mean(accuracy_score_val)])


print("\nAda提升分类器 Ada Boosting Decision Tree (ABDT)")
logger.info("Ada提升分类器")
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
	model= AdaBoostClassifier(n_estimators=100) 
	model.fit(X[train_index], Y[train_index])
	predict = model.score(X[test_index], Y[test_index])
	y_pred = model.predict(X[test_index])

	precision_score_val[i] 	= precision_score(Y[test_index], y_pred)
	recall_score_val[i] 	= recall_score(Y[test_index], y_pred)
	f1_score_val[i] 		= f1_score(Y[test_index], y_pred)
	accuracy_score_val[i] 	= accuracy_score(Y[test_index], y_pred)

print("均值:> \nprecision_score:%f \nrecall_score:%f \nf1_score:%f \naccuracy_score:%f"
	    % (np.mean(precision_score_val),
	    	np.mean(recall_score_val),
	    	np.mean(f1_score_val),
	    	np.mean(accuracy_score_val)))
logger.info("均值:> \nprecision_score:%f \nrecall_score:%f \nf1_score:%f \naccuracy_score:%f"
	    % (np.mean(precision_score_val),
	    	np.mean(recall_score_val),
	    	np.mean(f1_score_val),
	    	np.mean(accuracy_score_val)))
with open('./log/'+filename.replace('.py', '.csv'), 'a', newline='') as csv_file:
	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(['ABDT', np.mean(precision_score_val),
						np.mean(recall_score_val),
						np.mean(f1_score_val),
						np.mean(accuracy_score_val)])


print("\nK近邻分类器 k-Nearest Neighbor (KNN)")
logger.info("K近邻分类器(KNN)")
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
	model= KNeighborsClassifier(n_neighbors = 5)
	model.fit(X[train_index], Y[train_index])
	predict = model.score(X[test_index], Y[test_index])
	y_pred = model.predict(X[test_index])

	precision_score_val[i] 	= precision_score(Y[test_index], y_pred)
	recall_score_val[i] 	= recall_score(Y[test_index], y_pred)
	f1_score_val[i] 		= f1_score(Y[test_index], y_pred)
	accuracy_score_val[i] 	= accuracy_score(Y[test_index], y_pred)

print("均值:> \nprecision_score:%f \nrecall_score:%f \nf1_score:%f \naccuracy_score:%f"
	    % (np.mean(precision_score_val),
	    	np.mean(recall_score_val),
	    	np.mean(f1_score_val),
	    	np.mean(accuracy_score_val)))
logger.info("均值:> \nprecision_score:%f \nrecall_score:%f \nf1_score:%f \naccuracy_score:%f"
	    % (np.mean(precision_score_val),
	    	np.mean(recall_score_val),
	    	np.mean(f1_score_val),
	    	np.mean(accuracy_score_val)))
with open('./log/'+filename.replace('.py', '.csv'), 'a', newline='') as csv_file:
	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(['KNN', np.mean(precision_score_val),
						np.mean(recall_score_val),
						np.mean(f1_score_val),
						np.mean(accuracy_score_val)])


print("\n支持向量机 Support Vector Machine (SVM)")
logger.info("支持向量机(SVM)")
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
	model= SVC(C=1.0, kernel='linear', probability=True)
	model.fit(X[train_index], Y[train_index])
	predict = model.score(X[test_index], Y[test_index])
	y_pred = model.predict(X[test_index])

	precision_score_val[i] 	= precision_score(Y[test_index], y_pred)
	recall_score_val[i] 	= recall_score(Y[test_index], y_pred)
	f1_score_val[i] 		= f1_score(Y[test_index], y_pred)
	accuracy_score_val[i] 	= accuracy_score(Y[test_index], y_pred)

print("均值:> \nprecision_score:%f \nrecall_score:%f \nf1_score:%f \naccuracy_score:%f"
	    % (np.mean(precision_score_val),
	    	np.mean(recall_score_val),
	    	np.mean(f1_score_val),
	    	np.mean(accuracy_score_val)))
logger.info("均值:> \nprecision_score:%f \nrecall_score:%f \nf1_score:%f \naccuracy_score:%f"
	    % (np.mean(precision_score_val),
	    	np.mean(recall_score_val),
	    	np.mean(f1_score_val),
	    	np.mean(accuracy_score_val)))
with open('./log/'+filename.replace('.py', '.csv'), 'a', newline='') as csv_file:
	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(['SVM', np.mean(precision_score_val),
						np.mean(recall_score_val),
						np.mean(f1_score_val),
						np.mean(accuracy_score_val)])


print("\n线性判别分析 linear discriminant analysis (LDA)")
logger.info("线性判别分析(LDA)")
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
	model= LinearDiscriminantAnalysis(solver='svd', store_covariance=True) 
	model.fit(X[train_index], Y[train_index])
	predict = model.score(X[test_index], Y[test_index])
	y_pred = model.predict(X[test_index])

	precision_score_val[i] 	= precision_score(Y[test_index], y_pred)
	recall_score_val[i] 	= recall_score(Y[test_index], y_pred)
	f1_score_val[i] 		= f1_score(Y[test_index], y_pred)
	accuracy_score_val[i] 	= accuracy_score(Y[test_index], y_pred)

print("均值:> \nprecision_score:%f \nrecall_score:%f \nf1_score:%f \naccuracy_score:%f"
	    % (np.mean(precision_score_val),
	    	np.mean(recall_score_val),
	    	np.mean(f1_score_val),
	    	np.mean(accuracy_score_val)))
logger.info("均值:> \nprecision_score:%f \nrecall_score:%f \nf1_score:%f \naccuracy_score:%f"
	    % (np.mean(precision_score_val),
	    	np.mean(recall_score_val),
	    	np.mean(f1_score_val),
	    	np.mean(accuracy_score_val)))
with open('./log/'+filename.replace('.py', '.csv'), 'a', newline='') as csv_file:
	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(['LDA', np.mean(precision_score_val),
						np.mean(recall_score_val),
						np.mean(f1_score_val),
						np.mean(accuracy_score_val)])


print("\n二次判别分析 Quadratic Discriminant Analysis (QDA)")
logger.info("二次判别分析(QDA)")
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
	model= QuadraticDiscriminantAnalysis(store_covariance=True) 
	model.fit(X[train_index], Y[train_index])
	predict = model.score(X[test_index], Y[test_index])
	y_pred = model.predict(X[test_index])

	precision_score_val[i] 	= precision_score(Y[test_index], y_pred)
	recall_score_val[i] 	= recall_score(Y[test_index], y_pred)
	f1_score_val[i] 		= f1_score(Y[test_index], y_pred)
	accuracy_score_val[i] 	= accuracy_score(Y[test_index], y_pred)

print("均值:> \nprecision_score:%f \nrecall_score:%f \nf1_score:%f \naccuracy_score:%f"
	    % (np.mean(precision_score_val),
	    	np.mean(recall_score_val),
	    	np.mean(f1_score_val),
	    	np.mean(accuracy_score_val)))
logger.info("均值:> \nprecision_score:%f \nrecall_score:%f \nf1_score:%f \naccuracy_score:%f"
	    % (np.mean(precision_score_val),
	    	np.mean(recall_score_val),
	    	np.mean(f1_score_val),
	    	np.mean(accuracy_score_val)))
with open('./log/'+filename.replace('.py', '.csv'), 'a', newline='') as csv_file:
	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(['QDA', np.mean(precision_score_val),
						np.mean(recall_score_val),
						np.mean(f1_score_val),
						np.mean(accuracy_score_val)])


print("\n多层感知机(神经网络) Multi-Layer Perceptron MLP")
logger.info("多层感知机(神经网络)")
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
	model = MLPClassifier(activation='relu', solver='adam', alpha=0.0001)
	model.fit(X[train_index], Y[train_index])
	predict = model.score(X[test_index], Y[test_index])
	y_pred = model.predict(X[test_index])

	precision_score_val[i] 	= precision_score(Y[test_index], y_pred)
	recall_score_val[i] 	= recall_score(Y[test_index], y_pred)
	f1_score_val[i] 		= f1_score(Y[test_index], y_pred)
	accuracy_score_val[i] 	= accuracy_score(Y[test_index], y_pred)

print("均值:> \nprecision_score:%f \nrecall_score:%f \nf1_score:%f \naccuracy_score:%f"
	    % (np.mean(precision_score_val),
	    	np.mean(recall_score_val),
	    	np.mean(f1_score_val),
	    	np.mean(accuracy_score_val)))
logger.info("均值:> \nprecision_score:%f \nrecall_score:%f \nf1_score:%f \naccuracy_score:%f"
	    % (np.mean(precision_score_val),
	    	np.mean(recall_score_val),
	    	np.mean(f1_score_val),
	    	np.mean(accuracy_score_val)))
with open('./log/'+filename.replace('.py', '.csv'), 'a', newline='') as csv_file:
	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(['MLP', np.mean(precision_score_val),
						np.mean(recall_score_val),
						np.mean(f1_score_val),
						np.mean(accuracy_score_val)])
