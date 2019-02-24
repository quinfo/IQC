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
from sklearn import preprocessing

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
batch = 60
rate = 0.009
bias = 0
starVal = 0.1

logger.info("batch:%d rate:%f starVal:%f" % (batch, rate, starVal))


################读取数据##################
iris = datasets.load_iris()
#X = get_map_data(iris.data)
X = iris.data
Y = iris.target


print('\nIRIS数据集:>', X.shape,
		'\nIRIS数据集:>',
		list(Y).count(0),
	  	list(Y).count(1),
	  	list(Y).count(2),)

logger.info("IRIS数据集: %s %s" % (str(X.shape), str(Y.shape)))


X = np.transpose(X)


#标准化
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)


#正规化
#scaler = preprocessing.Normalizer().fit(X)
#X = scaler.transform(X)


#尺度变换
#scaler = preprocessing.MinMaxScaler(feature_range=(-1,0))
#X = scaler.fit_transform(X)

X = np.transpose(X)



##############准备正确的结果##############
#最终结果
Results = np.zeros(Y.shape[0])



#要学习的参数
Param = np.ones((3, 4)) * starVal # 会因参数调整
Param_Best = Param
Param_Flag = np.zeros(3)



# 准备量子态
psi_sys = (basis(2, 0) + basis(2, 1)).unit()
psi_env = (basis(4, 0) + basis(4, 1) + basis(4, 2) + basis(4, 3)).unit()
psi = tensor(psi_sys, psi_env) # 会因参数调整



#对数据分折
skf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)# 会因参数调整
################程序执行##################
for fold, (train_index, test_index) in enumerate(skf.split(X, Y)):

	

	#训练模型
	for p, param  in enumerate(Param):

		#重置参数Flag 
		param_Flag = 0


		X_train = X[train_index]
		Y_train = Y[train_index]



		#将要训练的设定为1，其它设定为0。
		if p == 0:
			Y_train = (Y_train == 0).astype(int)

		elif p == 1:
			Y_train = (Y_train == 1).astype(int)

		elif p == 2:
			Y_train = (Y_train == 2).astype(int)



		#补齐正负不平衡的问题
		X_train, Y_train = sorting_data3(X_train, Y_train)


		#数据打乱处理
		#from scipy.sparse import coo_matrix
		#X_sparse = coo_matrix(X_train)
		#from sklearn.utils import shuffle
		#X_train, _, Y_train = shuffle(X_train, X_sparse, Y_train, random_state=0)
		


		##############训练模型#######################
		for j in range(batch):


			##########训练############################
			for i, x in enumerate(X_train):

				# 处理分割产生的参数不足
				#if x.shape[0] < param.shape[0]: continue


				# 准备算子
				H_sys = sigmax() + sigmay() + sigmaz()
				Hami = get_Hami(H_sys, x, param)


				# 执行运算
				result = ptrace(Hami * psi, 0).diag().tolist()


				# 参数学习
				cz = 1 - max(result)
				z = result.index(max(result))
				dire = Y_train[i] - z

				Param[p,:] = param - rate * cz * dire * x






			##########测试############################
			yes_num = 0
			for i, x in enumerate(X_train):
				# 准备算子
				H_sys = sigmax() + sigmay() + sigmaz()
				Hami = get_Hami(H_sys, x, param)

				result_list = ptrace(Hami * psi, 0).diag().tolist()
				result = result_list.index(max(result_list))

				if result == Y_train[i]: yes_num += 1

			#打印结果
			mean_num = yes_num / len(X_train)
			print('Model:>', p, 'batch:', j, ' Prop:', mean_num)
			if mean_num > param_Flag:
				param_Flag = mean_num
				Param_Best[p] = param
				Param_Flag[p] = mean_num
				#if mean_num == 1: break






	##############测试集测试#######################
	for i in test_index: 

		test_results = []

		for p, param in enumerate(Param_Best):
			H_sys = sigmax() + sigmay() + sigmaz()
			Hami = get_Hami(H_sys, X[i], param)			
			test_results.append(ptrace(Hami * psi, 0).diag().tolist())

		test_results_list = list(np.array(test_results)[:, 1])

		Results[i] = list(test_results_list).index(max(test_results_list))


		#if list(Param_Flag).index(max(Param_Flag)) == list(test_results_list).index(max(test_results_list)):
		#	Results[i] = list(test_results_list).index(max(test_results_list))
		#else:
		#	test_results_list = test_results_list * Param_Flag
		#	Results[i] = list(test_results_list).index(max(test_results_list))


		#print(test_results_list, max(test_results_list), Y[i])


	#############结果打印##################
	counter = [i for i in test_index if Results[i] == Y[i]]
	print('Fold:', fold, 'Prop:>',len(counter) / len(test_index), len(counter), len(test_index)-len(counter))
	logger.info("%d:> %f" % (fold, len(counter) / len(test_index)))









counter = [i for i in range(len(Y)) if Results[i] == Y[i]]
print("precision_score:> %f" % (precision_score(Y, Results, average='macro')))
print("recall_score:   > %f" % (recall_score(Y, Results, average='micro')))
print("f1_score:       > %f" % (f1_score(Y, Results, average='weighted')))
print("accuracy_score: > %f" % (accuracy_score(Y, Results)))
logger.info("precision_score:> %f" % (precision_score(Y, Results, average='macro')))
logger.info("recall_score:   > %f" % (recall_score(Y, Results, average='micro')))
logger.info("f1_score:       > %f" % (f1_score(Y, Results, average='weighted')))
logger.info("accuracy_score: > %f" % (accuracy_score(Y, Results)))
























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

	precision_score_val[i] 	= precision_score(Y[test_index], y_pred, average='macro')
	recall_score_val[i] 	= recall_score(Y[test_index], y_pred, average='micro')
	f1_score_val[i] 		= f1_score(Y[test_index], y_pred, average='weighted')
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

	precision_score_val[i] 	= precision_score(Y[test_index], y_pred, average='macro')
	recall_score_val[i] 	= recall_score(Y[test_index], y_pred, average='micro')
	f1_score_val[i] 		= f1_score(Y[test_index], y_pred, average='weighted')
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

	precision_score_val[i] 	= precision_score(Y[test_index], y_pred, average='macro')
	recall_score_val[i] 	= recall_score(Y[test_index], y_pred, average='micro')
	f1_score_val[i] 		= f1_score(Y[test_index], y_pred, average='weighted')
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

	precision_score_val[i] 	= precision_score(Y[test_index], y_pred, average='macro')
	recall_score_val[i] 	= recall_score(Y[test_index], y_pred, average='micro')
	f1_score_val[i] 		= f1_score(Y[test_index], y_pred, average='weighted')
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

	precision_score_val[i] 	= precision_score(Y[test_index], y_pred, average='macro')
	recall_score_val[i] 	= recall_score(Y[test_index], y_pred, average='micro')
	f1_score_val[i] 		= f1_score(Y[test_index], y_pred, average='weighted')
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

	precision_score_val[i] 	= precision_score(Y[test_index], y_pred, average='macro')
	recall_score_val[i] 	= recall_score(Y[test_index], y_pred, average='micro')
	f1_score_val[i] 		= f1_score(Y[test_index], y_pred, average='weighted')
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

	precision_score_val[i] 	= precision_score(Y[test_index], y_pred, average='macro')
	recall_score_val[i] 	= recall_score(Y[test_index], y_pred, average='micro')
	f1_score_val[i] 		= f1_score(Y[test_index], y_pred, average='weighted')
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

	precision_score_val[i] 	= precision_score(Y[test_index], y_pred, average='macro')
	recall_score_val[i] 	= recall_score(Y[test_index], y_pred, average='micro')
	f1_score_val[i] 		= f1_score(Y[test_index], y_pred, average='weighted')
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

	precision_score_val[i] 	= precision_score(Y[test_index], y_pred, average='macro')
	recall_score_val[i] 	= recall_score(Y[test_index], y_pred, average='micro')
	f1_score_val[i] 		= f1_score(Y[test_index], y_pred, average='weighted')
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

	precision_score_val[i] 	= precision_score(Y[test_index], y_pred, average='macro')
	recall_score_val[i] 	= recall_score(Y[test_index], y_pred, average='micro')
	f1_score_val[i] 		= f1_score(Y[test_index], y_pred, average='weighted')
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

	precision_score_val[i] 	= precision_score(Y[test_index], y_pred, average='macro')
	recall_score_val[i] 	= recall_score(Y[test_index], y_pred, average='micro')
	f1_score_val[i] 		= f1_score(Y[test_index], y_pred, average='weighted')
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
