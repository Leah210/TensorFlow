# coding: UTF-8
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
hello = tf.constant('hello, tensorflow')
sess = tf.Session()
print('hello =',sess.run(hello))
x = tf.constant(np.random.randn(32).astype(np.float32))
print('x =\n',sess.run(x))
x1 = np.array([[1.,2.,3.],[3.,4.,5.]])
x2 = tf.convert_to_tensor(x1,dtype=tf.float32) #convert python obj to tensor obj
print('x2 =\n',sess.run(x2))

# matrix operation------------------------------------------------------------------------------------------------------
x = tf.constant([[2, 5, 3, -5],
                 [0, 3,-2,  5],
                 [4, 3, 5,  3],
                 [6, 1, 4,  0]])
y = tf.constant([[4, -7, 4, -3, 4],
                 [6, 4,-7,  4, 7],
                 [2, 3, 2,  1, 4],
                 [1, 5, 5,  5, 2]])
floatx = tf.constant([[2., 5., 3., -5.],
                      [0., 3.,-2.,  5.],
                      [4., 3., 5.,  3.],
                      [6., 1., 4.,  0.]])
print("\n")
print("matrix operation-----------------------------------------------------------------------------------------------------")
print('original x =\n', sess.run(x))
print('transpose x = \n',sess.run(tf.transpose(x))) # 转置
print('multiply x*y = \n', sess.run(tf.matmul(x, y))) # 矩阵乘法
print('determinant floatx = \n', sess.run(tf.matrix_determinant(floatx))) # 求行列式
print('inverse floatx = \n', sess.run(tf.matrix_inverse(floatx))) # 求逆矩阵
print('solve floatx = \n', sess.run(tf.matrix_solve(floatx, [[1], [1], [1], [1]]))) # 求线性方程组

# reduction, 指定维度做指定操作-------------------------------------------------------------------------------------------
x = tf.constant([[1, 2, 3],
                 [3, 2, 1],
                 [-1, -2, -3]])
x2 = tf.constant([[[1, 2, 3, 5],
                 [3, 2, 1, 4]],
                 [[-1, -2, -3, 2],
                  [-1, -2, -3, 3]],
                 [[-1, -2, -3, 4],
                  [-1, -2, -3, 8]]])

boolean_tensor = tf.constant([[True, False, True],
                              [False, False, True],
                              [True, False, False]])
print("\n")
print("reduction, 指定维度做指定操作-------------------------------------------------------------------------------------------")
print('prod x = \n', sess.run(tf.reduce_prod(x, reduction_indices=1)))
print('min in each dimension of x = \n', sess.run(tf.reduce_min(x, reduction_indices=1)))
print('max in each dimension of x = \n', sess.run(tf.reduce_max(x, reduction_indices=1)))
print('mean in each dimension of x = \n', sess.run(tf.reduce_mean(x, reduction_indices=1)))
print('logical and in each dimension of x = \n', sess.run(tf.reduce_all(boolean_tensor, reduction_indices=1))) # reduction_indices的值指的是维度
print('logical or in each dimension of x = \n', sess.run(tf.reduce_any(boolean_tensor, reduction_indices=0)))

print("\n")
print('shape of x2 = ', sess.run(tf.shape(x2)))
print('x2 = \n',sess.run(x2))
print("\n")
reduce_sum_x2 = tf.reduce_sum(x2,axis=0)
print('the shape of reduce_sum_x2 = ', sess.run(tf.shape(reduce_sum_x2)))
print('reduce_sum_x2 = \n', sess.run(reduce_sum_x2))

# segmentation, 类似分组操作---------------------------------------------------------------------------------------------
seg_ids = tf.constant([0,1,1,2,2]); # Group indexes : 0|1,2|3,4
tens1 = tf.constant([[2, 5, 3, -5],
                    [0, 3,-2,  5],
                    [4, 3, 5,  3],
                    [6, 1, 4,  0],
                    [6, 1, 4,  0]])  # A sample constant matrix
print("\n")
print("segmentation, 类似分组操作---------------------------------------------------------------------------------------------")
print('sum segmentation =\n', sess.run(tf.segment_sum(tens1, seg_ids))) # sum segmentation
print('prod segmentation =\n', sess.run(tf.segment_prod(tens1, seg_ids))) # proc segmentation
print('min segmentation =\n', sess.run(tf.segment_min(tens1, seg_ids))) # min segmentation
print('max segmentation =\n', sess.run(tf.segment_max(tens1, seg_ids))) # max segmentation
print('mean segmentation =\n', sess.run(tf.segment_mean(tens1, seg_ids))) # mean segmentation

# indexed operations----------------------------------------------------------------------------------------------------
x = tf.constant([[2, 5, 3, -5],
                 [0, 3,-2,  5],
                 [4, 3, 5,  3],
                 [6, 1, 4,  0]])
listx = tf.constant([1,20,3,4,5,6,7,8,9])
listy = tf.constant([40,5,8,9])
boolx = tf.constant([[True,False], [False, True]])
# print('position of max value in col = ', tf.arg_max(x, dimension=1)) # tf.arg_max弃用了
# print('position of min value in col = ', tf.arg_min(x, dimension=1)) # tf.arg_mmin弃用了
print("\n")
print("indexed operations----------------------------------------------------------------------------------------------------")
print('position of max value in col = ', sess.run(tf.argmax(x, 1)))
print('position of min value in col = ', sess.run(tf.argmin(x, 1)))
print('list diff = \n', sess.run(tf.setdiff1d(listx, listy)[0])) #这里显示的结果是取差集的结果
print('list diff index = \n', sess.run(tf.setdiff1d(listx, listy)[1])) #这里显示的结果是取差集的索引
print('where boolx=\n', sess.run(tf.where(boolx)))
print('unique listx=\n', sess.run(tf.unique(listx)[0]))

# rank and reshape------------------------------------------------------------------------------------------------------
x = tf.constant([[2, 5, 3, -5],
                 [0, 3,-2,  5],
                 [4, 3, 5,  3],
                 [6, 1, 4,  0]])
print("\n")
print("rank and reshape------------------------------------------------------------------------------------------------------")
print('show the dimension of x = \n', sess.run(tf.shape(x))) # 维度
print('show the size/length of x = \n', sess.run(tf.size(x))) # 大小
print('show the rank of x = \n', sess.run(tf.rank(x))) # 秩
print('reshpe the x =\n', sess.run(tf.reshape(x,[8,2]))) #reshape the x
print('expand dimension of x =\n', sess.run(tf.expand_dims(x,1))) #expand dimension,将原来的数据增加一个维度,可尝试增加2个维度看效果.

# slicing, joining, padding, pack, unpack-------------------------------------------------------------------------------
t_matrix = tf.constant([[1,2,3],
                        [4,5,6],
                        [7,8,9],
                        [10,11,12]])
t_array = tf.constant([[1, 2, 3, 4, 9, 8, 6, 5]])
t_array2 = tf.constant([[2, 3, 4, 5,6, 7, 8, 9]])
print("\n")
print("slicing, joining, padding, pack, unpack------------------------------------------------------------------------------")
print('the original t_matrix = \n', sess.run(t_matrix))
print('slice the t_matrix = \n', sess.run(tf.slice(t_matrix, [1, 1], [3,2]))) # 下标从0开始
print('split the t_array = \n', sess.run(tf.split(t_matrix,num_or_size_splits=4,axis=0))) # 将矩阵划分为指定的份数num_or_size_splits, 按照维度axis来划分.0是按照行来划分.
print('tiling this little tensor N times =\n', sess.run(tf.tile([1,2,3],[3]))) #构建tensor, 即重复3次[1,2,3]
pad = tf.constant([[3,4],[2,3]])
print('pad the matrix =\n', sess.run(tf.pad(t_matrix, pad))) #填充tensor
print('concat the tensor = \n',sess.run(tf.concat([t_array,t_array2],0))) #concat the tensor by row, 教材由于版本原因参数位置不同
print('concat the tensor_1 = \n',sess.run(tf.concat([t_array,t_array2],1))) #concat the tensor by col
# 注意: 在1.10版本中
# pack和unpack已经没找到
# 更新成stack和unstack, 详情见接口
t_matrix2 = tf.constant([[[1,2,3],
                          [4,5,6]],
                         [[7,8,9],
                          [2,5,8]]])
print('original t_matrix2= \n', sess.run(t_matrix2))
# t_matrix2总共有3维度, 即0,1,2,那么在反转过程中应该理解tensor对应的维度是怎么反转的, 可尝试不同参数
print('reverse the matrix = \n', sess.run(tf.reverse(t_matrix2,[0]))) #根据指定维数反转张量, 教材上是[FALSE,TRUE],实际上是[0,1]
print('reverse2 the matrix = \n', sess.run(tf.reverse(t_matrix2,[0,2]))) #根据指定维数反转张量, 教材上是[FALSE,TRUE],实际上是[0,1]. 该例子相当于指定维度0和2进行反转