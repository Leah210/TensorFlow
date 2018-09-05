# coding=utf-8
import tensorflow as tf
import numpy as np
import sklearn as sk
import time
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_circles
from sklearn.datasets.samples_generator import make_blobs

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

DATA_TYPE = 'blobs'
N = 200
if DATA_TYPE=='circle':
    K=2
else:
    K=4

MAX_ITERS = 1000
start = time.time()
centers = [(-2, -2), (-2, 1.5), (1.5, -2), (2, 1.5)]

if DATA_TYPE=='circle':
    data, features = make_circles(200, shuffle=True, noise=0.1, factor=0.4)
else:
    data, features = make_blobs(200, centers=centers, n_features=2, cluster_std=0.8, shuffle=False, random_state=42)

# fig, ax = plt.subplots()
# ax.scatter(np.asarray(centers).transpose()[0], np.asarray(centers).transpose()[1], marker = 'o', s=250)
# plt.show() # 在pycharm中加入这句话才能够显示图像

# fig, ax = plt.subplots()
# if DATA_TYPE=='blobs':
#     ax.scatter(np.asarray(centers).transpose()[0], np.asarray(centers).transpose()[1], marker='o', s=250)
#     ax.scatter(data.transpose()[0], data.transpose()[1], marker='o', s=100, c=features, cmap=plt.cm.coolwarm)
#     plt.show()

sess = tf.Session()
points = tf.Variable(data) # data数据放入points变量中
cluster_assignments = tf.Variable(tf.zeros([N], dtype=tf.int64))
# 如果需要用一个变量的值给另一个变量初始化的时候，则用到intial_value()
# slice [0,0][K,2]从第0行第0列开始, 取出K行2列
# K=4,class=2,即4个簇,2个类别
centroids = tf.Variable(tf.slice(points.initialized_value(),[0,0],[K,2])) # 这里醉人, 我用成了initial_value(),因此出错!!当心!!


sess.run(tf.global_variables_initializer())
sess.run(centroids)

print("----------------------------------------------------")
print('the shape of points is = ', sess.run(tf.shape(points)))
print('the sub of points is = \n', sess.run(tf.slice(points,[0,0],[5,2])))

# 簇中心和样本保持维度一样, 扩充的作用就是后面不需要循环计算距离
# 例如某个样本扩充后为
# [[-1.60262868 -2.11061144]
#   [-1.60262868 -2.11061144]
#   [-1.60262868 -2.11061144]
#   [-1.60262868 -2.11061144]]
# 而簇中心(重复200次)为
# [[-1.60262868 -2.11061144]
#   [-1.48184917 -0.78157611]
#   [-2.1873227  -2.18730957]
#   [-0.73662975 -1.38605222]]
# 这样计算距离则利用tf.square(rep_points-rep_centroids)来计算: 一个样本与4个簇中心的距离值,得到200*4的tensor
# 即该样本与这4个簇中心的距离为
# [[ 0.          0.        ]
#   [ 0.01458769  1.7663349 ]
#   [ 0.3418671   0.0058826 ]
#   [ 0.74995415  0.52498607]]
#  其中这里的[0. 0.]表示[(x1-x2)^2, (y1-y2)^2]
rep_centroids = tf.reshape(tf.tile(centroids,[N,1]), [N,K,2]) # 每个簇中心重复200次
rep_points = tf.reshape(tf.tile(points,[1,K]), [N,K,2]) #每一个样本重复4次

print("----------------------------------------------------")
print('the shape of centroids is = ', sess.run(tf.shape(centroids)))
print('the shape of rep_centrios is = ',sess.run(tf.shape(rep_centroids)))

print("----------------------------------------------------")
print("the sub of centroids is =\n", sess.run(tf.slice(centroids,[0,0], [4,2])),'\n')
print("the sub of rep_centroids is =\n", sess.run(tf.slice(rep_centroids,[0,0,0], [2,4,2])))

print("----------------------------------------------------")
print('the shape of rep_points is = ', sess.run(tf.shape(rep_points)))
print('the sub of rep_points is = \n', sess.run(tf.slice(rep_points,[0,0,0],[2,4,2])))

# 每个样本到簇中心的距离(该距离是通过所有维度进行度量的)
square_distance = tf.square(rep_points-rep_centroids)
sum_squares = tf.reduce_sum(square_distance, reduction_indices = 2)# reduction_indices: The old (deprecated) name for axis.
# 需要先了解square_distance的维度情况, 后面发现是[200,4,2]
# 那么求解sum_squares时, 指定降低的维度reduction_indices=2, 那么sum_squares的维度最终为[200,4]
print("----------------------------------------------------")
print("the shape of distance between sample and centroids = ",sess.run(tf.shape(square_distance)))
print("the square of distance between sample and centroids =\n",sess.run(tf.slice(square_distance,[0,0,0],[1,4,2])))

print("----------------------------------------------------")
print("the shape of sum_squares = ", sess.run(tf.shape(sum_squares)))
print("show the sub sum_squares = \n", sess.run(tf.slice(sum_squares,[0,0],[5,4])))

# the shape of sum_squares is [200,4]
best_centroids = tf.argmin(sum_squares, 1) # 求sum_squares中索引为1的维度的最小值, 即样本与簇中心最小的距离
# 当所有的簇中心不再变化, 则停止(cluster_assignments是初始化为0的)
did_assignment_change = tf.reduce_any(tf.not_equal(best_centroids, cluster_assignments))

def bucket_mean(data, bucket_ids, num_buckets):
    total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
    count = tf.unsorted_segment_sum(tf.ones_like(data), bucket_ids,num_buckets)
    return total/count

means = bucket_mean(points,best_centroids,K) # 求所有个样本到距离最近的簇中心的距离均值

# When eager execution is enabled, any callable object in the control_inputs list will be called
with tf.control_dependencies([did_assignment_change]):
    do_update = tf.group(
        centroids.assign(means), #将centroids 的值更新为距离均值
        cluster_assignments.assign(best_centroids) #将计算好的best_centroids赋给记录簇中心的cluster_assignments
    )

changed = True
iters = 0

fig, ax = plt.subplots()

if DATA_TYPE=='blobs':
    colorindex = [2,1,4,3]
else:
    colorindex = [2,1] #这个只是画图所用
while changed and iters < MAX_ITERS:
    # fig, ax = plt.subplots()
    iters += 1;
    [changed,_] = sess.run([did_assignment_change,do_update])  #记得要do_update,因为忘记指挥重复上面操作而没有更新簇中心
    [centers, assignments] = sess.run([centroids, cluster_assignments])
    if sess.run(tf.reduce_any(did_assignment_change))==False & changed==False: #画出最后结果即可
        ax.scatter(sess.run(points).transpose()[0], sess.run(points).transpose()[1], marker='o', s=200, c=assignments,
                   cmap=plt.cm.coolwarm)
        ax.scatter(centers[:, 0], centers[:, 1], marker='^', s=550, c=colorindex, cmap=plt.cm.plasma)
        ax.set_title('Iteration ' + str(iters))
        plt.savefig("kmeans" + str(iters) + ".png")
        break
# ax.scatter(sess.run(points).transpose()[0], sess.run(points).transpose()[1], marker = 'o', s = 200, c = assignments, cmap=plt.cm.coolwarm )
plt.show()
end = time.time()
print('result:---------------------------------------------')
print('Found in %.2f seconds', (end-start), iters, 'Iteration')
print("centers:", centers)
print('Cluster assignments:\n', assignments)
# 教材上的写法会多一次运行(对结果没什么影响),并且会每次运行都打开fig, 保存下所有的图片(没有必要)