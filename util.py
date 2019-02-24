from qutip import *
import numpy as np
from random import shuffle
import random
import math


psi_2 = (basis(2, 0) + basis(2, 1)).unit()
psi_3 = (basis(3, 0) + basis(3, 1) + basis(3, 2)).unit()
psi_4 = (basis(4, 0) + basis(4, 1) + basis(4, 2) + basis(4, 3)).unit()
psi_5 = (basis(5, 0) + basis(5, 1) + basis(5, 2) + basis(5, 3) + basis(5, 4)).unit()


def get_Psi2(sys_psi, env_psi, env_count = 1):
	pass


def get_Psi(sys_size = 2, env_size = 2, env_count = 4):

	# 给变量赋值并校验输入值正误
	if sys_size == 2:
		psi = psi_2
	elif sys_size == 3:
		psi = psi_3
	elif sys_size == 4:
		psi = psi_4
	elif sys_size == 5:
		psi = psi_5
	else:
		raise ValueError('输入值不能匹配')

	# 给变量赋值并校验输入值正误
	if not isinstance(env_count, int):
		raise ValueError('输入值有误')

	# 给变量赋值并校验输入值正误
	for i in range(env_count):
		if env_size == 2:
			psi = tensor(psi, psi_2)
		elif env_size == 3:
			psi = tensor(psi, psi_3)
		elif env_size == 4:
			psi = tensor(psi, psi_4)
		elif env_size == 5:
			psi = tensor(psi, psi_5)
		else:
			raise ValueError('输入值不能匹配')

	return psi



def get_Param(env_size = 2, env_count = 3, default = 0.1):
	result = []
	for i in range(env_count):

		if not isinstance(env_size, int):
			raise ValueError('输入值有误')
		else:
			temp = np.ones(env_size) * default

		result.append(temp)

	return np.array(result)



def get_Hami(H_sys, data_x, param):


	if data_x.ndim != 2:
		data_x = np.reshape(data_x, (1,len(data_x)))
		
	if param.ndim != 2:
		param = np.reshape(param, (1,len(param)))


	try:
		# 准备算子
		H_env = None
		for row in np.split(data_x + param, param.shape[0], axis = 0):
			H = Qobj(np.diag(row.reshape(-1)))
			if H_env is None: H_env = H
			else: H_env = tensor(H_env, H)

		Hami = (-complex(0, 1) * tensor(H_sys, H_env)).expm()
	except Exception as e:
		print(data_x, param)
		raise e
	


	return Hami


def sorting_data(X, Y):

	# 统计数量与种类
	clazz = np.unique(Y)
	size = []
	for i in clazz:
		size.append(list(Y).count(i))


	# 生成要添加的ID号
	new_ids = []
	for x in range(len(size)):
		max_size = max(size)
		if size[x] >= max_size: continue

		num = max_size - size[x]
		ids = list(np.where(Y == x)[0])

		for i in range(num):
			new_ids.append(random.sample(ids, 1)[0])


	# 添加元素,生成就X，Y
	for i in new_ids:
		Y = np.r_[Y, [Y[i, :]]]
		X = np.r_[X, [X[i, :]]]


	return X, Y




def sorting_data2(X, Y):

	# 统计数量与种类
	clazz = np.unique(Y)
	size = []
	for i in clazz:
		size.append(list(Y).count(i))


	# 生成要添加的ID号
	new_ids = []
	for x in range(len(size)):
		min_size = min(size)
		if size[x] >= min_size: continue

		num = max_size - size[x]
		ids = list(np.where(Y == x)[0])

		for i in range(num):
			new_ids.append(random.sample(ids, 1)[0])


	# 添加元素,生成就X，Y
	for i in new_ids:
		Y = np.r_[Y, [Y[i, :]]]
		X = np.r_[X, [X[i, :]]]


	return X, Y




def sorting_data3(X, Y):

	# 统计数量与种类
	clazz = np.unique(Y)
	size = []
	for i in clazz:
		size.append(list(Y).count(i))



	# 生成要添加的ID号
	new_ids = []
	for x in range(len(size)):
		max_size = max(size)
		if size[x] >= max_size: continue

		num = max_size - size[x]
		ids = list(np.where(Y == x)[0])

		for i in range(num):
			new_ids.append(random.sample(ids, 1)[0])


	# 添加元素,生成就X，Y
	for i in new_ids:
		Y = np.r_[Y, [Y[i]]]
		X = np.r_[X, [X[i, :]]]


	return X, Y




def get_map_data(X):
	temp_X = X
	for c, col in enumerate(np.split(X, X.shape[1], axis = 1)):
		a = float(min(col))
		b = float(max(col))
		try:
			for r, row in enumerate(col):
				temp_X[r, c] = (1/(b-a)) * X[r, c] - (b / (b - a))
		except Exception as e:
			print(a, b)
			raise e
		


	return temp_X



def shuffle_data(X, Y):

	arr = np.arange(len(Y))
	shuffle(arr)

	new_X = []; new_Y = []
	for i in arr:
		new_X.append(X[i])
		new_Y.append(Y[i])

	return np.array(new_X), np.array(new_Y)



def get_shuffle_data(X, Y):

	# 将数据映射到指定区间
	def change(item, x):
		return (1/max(X[:, item])) * x - 1


	# 参数检验
	if X.shape[0] != len(Y):
		print(X.shape, len(Y))
		raise ValueError('输入值有误')


	arr = np.arange(len(Y))
	shuffle(arr)

	new_X = []; new_Y = []
	for i in arr:
		arr = [change(i, x) for i, x in enumerate(X[i,:])]

		new_X.append(np.array(arr))
		new_Y.append(Y[i])

	return np.array(new_X), new_Y



def get_proportion_data(X, Y, prop = (8, 2)):

	# 参数检验
	if not isinstance(prop, tuple):
		raise ValueError('要求参数为元组')
	elif prop[0] + prop[1] != 10:
		raise ValueError('参数之和不为10')


	# 参数检验
	if X.shape[0] != len(Y):
		print(X.shape, len(Y))
		raise ValueError('输入值有误')

	P = math.floor((len(Y) / 10) * prop[0])

	return X[:P,:], Y[:P], X[P:,:], Y[P:]



def get_sigmaXYZ():
	
	sigmax = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

	sigmay_0 = np.array([[0, 0, 1j], [0, -1j, 0], [1j, 0, 0]])
	sigmay_1 = np.array([[0, 0, -1j], [0, -1j, 0], [-1j, 0, 0]])
	sigmaz_0 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
	sigmaz_1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
	sigmaz_2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

	return Qobj(sigmax + sigmay_1 + sigmaz_1)






