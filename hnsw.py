#将feature文件存为pkl.gz文件来存储，但是10000条都存不了
# -*- coding: utf-8 -*-
import pprint
import sys
import time
from heapq import heapify, heappop, heappush, heapreplace, nlargest, nsmallest
from math import log2
from operator import itemgetter
from random import random
from re import search

import numpy as np
import pickle
import math
import random
import os
from Bio import SeqIO
import gzip
from collections import defaultdict



#设置参数
# 指定地址并且读取序列
fasta_file = "/Users/xiangdongsheng/Desktop/项目代码包/minhash-lsh-hnsw20250107/fasta_file/6000-10000.fasta"
#从文件读取序列数目
sequence_number_all = 10000
#数据集划分的起始数目
sequence_number_train_start = 6000
#数据集划分的起始数目-1
sequence_number_train_start_subtract1 = 6000
#提取出的特征文件名
signature_file = 'signature_6000e10000test.pkl'
#提取出的相似度【二维数组】文件名字
feature_file ='/Users/xiangdongsheng/Desktop/项目代码包/minhash-lsh-hnsw20250107/feature_matrix_normalization6000e10000test.pkl.gz'
feature_filename = "feature_matrix_normalization6000e10000test.pkl.gz"
# 最后要除以的整数以变成概率
divisor = 4000
#定义时间参数
get_feature_time = 0
calculate_similar_time = 0
build_hnsw_time = 0
search_time = 0
#定义要搜索多少个最相似的序列
kn = 1
#先设置时间为0，不然用已有文件时候会报错，此时时间都为0
start_get_feature_time = 0
stop_get_feature_time = 0
start_calculate_similar_time = 0
stop_calculate_similar_time = 0




#读取1000条序列
def read_fasta_sequences(fasta_file, num_sequences=sequence_number_all):
    sequences = []
    count = 0
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq).upper())
        count += 1
        if count >= num_sequences:
            break
    return sequences


sequences = read_fasta_sequences(fasta_file, sequence_number_all)

#创建shingles
def build_shingles(sentence: str, k: int):
    shingles = []
    for i in range(len(sentence) - k):
        shingles.append(sentence[i:i+k])
    return shingles

#创建vocab(其由shingles构成)
def build_vocab(shingle_sets: list):
    # 将shingle集合列表转换为单个集合
    full_set = {item for set_ in shingle_sets for item in set_}
    vocab = {}
    for i, shingle in enumerate(list(full_set)):
        vocab[shingle] = i
    return vocab

#创建one-hot编码
def one_hot(shingles: set, vocab: dict):
    vec = np.zeros(len(vocab))
    for shingle in shingles:
        idx = vocab[shingle]
        vec[idx] = 1
    return vec

#此处是新版的函数，将哈希签名的矩阵换的行列换了一下，便于接下来使用
def getMinHashSignature(shingleList, signatureNum):
    # tatalSet用于存放所有集合的并集,shingleList: 一个列表，其中每个元素是一个集合，代表一个文档的 shingles
    totalSet = shingleList[0]
    for i in range(1, len(shingleList)):
        totalSet = set(totalSet) | set(shingleList[i])

    temp = int(math.sqrt(signatureNum))
    # randomArray用于模拟随机哈希函数
    randomArray = []
    # signatureList用于存放总的哈希签名
    signatureList = []
    g=1
    maxNum = sys.maxsize
    for i in range(signatureNum):
        randomArray.append(random.randint(1, temp * 2))
        randomArray.append(random.randint(1, temp * 2))
    # buketNum用于记录所有元素的个数，作为随机哈希函数的桶号
    buketNum = len(totalSet)
    """
    此处将不同文档的自己的哈希签名存成一个list，然后再进行汇总到一个总的list
    """
    for shingleSet in shingleList:
        """
        signature用于存放哈希函数产生的签名
        """
        signature = []
        for i in range(signatureNum):
            minHash = maxNum
            for index, item in enumerate(totalSet):
                if item in shingleSet:
                    num = (randomArray[i * 2] * index + randomArray[i * 2 + 1]) % buketNum
                    minHash = min(minHash, num)
            signature.append(minHash)
        signatureList.append(signature)
        print("提取序列特征数量:"+str(g))
        g=g+1
    return signatureList

#此函数通过比较两个文档的最小哈希签名进行计算相似度，传入的参入是两个文档的最小哈希签名的集合，存放在list中，最后结果返回相似度
def calSimilarity(signatureSet1, signatureSet2):
    count = 0
    for index in range(len(signatureSet1)):
        if (signatureSet1[index] == signatureSet2[index]):
            count += 1
    return count / (len(signatureSet1) * 1.0)

#此函数用于将计算所有文档的相似度，并将结果存放在一个list中，结果用元组存放
def calAllSimilarity(signatureList, filesName):
    signatureNum = len(signatureList)
    fileNum = len(filesName)
    result = []
    for index1, signatureSet1 in enumerate(signatureList):
        for index2, signatureSet2 in enumerate(signatureList):
            if (index1 < index2):
                result.append((calSimilarity(signatureSet1, signatureSet2), filesName[index1], filesName[index2]))
    return result

# 保存压缩的.pkl文件
def save_compressed_pickle(obj, filename, protocol=pickle.HIGHEST_PROTOCOL):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)

# 加载压缩的.pkl文件
def load_compressed_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)


#判断模块，看看是不是已经生成了特征矩阵feature_matrix_normalization_ys，生成了就不用再生成了
script_dir = os.path.dirname(os.path.abspath("D:\研究项目相关代码\hnsw-python-master\minhash-lsh-hnsw20241015\test\test_保存.py"))
# 定义文件存储路径，确保它位于项目文件夹内
file_path = os.path.join(script_dir, feature_file)
print(file_path)
# 检查文件是否存在
if os.path.exists(file_path):
    try:
        feature_matrix_normalization = load_compressed_pickle(feature_filename)
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
    #     # 文件存在，直接加载
    #     with open(file_path, 'rb') as f:
    #         feature_matrix_normalization = pickle.load(f)
    #     print(f"Feature matrix normalization loaded from {file_path}")
    # except Exception as e:
    #     print(f"An error occurred while loading the file: {e}")

else:
    # 文件不存在，生成 featurematrixnormalization 并保存
    # shingle 尺寸
    k = 6
    # 建立shingles
    shingles = []
    for sequence in sequences:
        shingles.append(build_shingles(sequence, k))
    print(shingles)

    #提取特征并且计时
    start_get_feature_time = time.time()
    signatureList = getMinHashSignature(shingles, 100)
    stop_get_feature_time = time.time()

    # 将生成的特征signature也存入文件
    script_dir1 = os.path.dirname(
        os.path.abspath("D:\研究项目相关代码\hnsw-python-master\minhash-lsh-hnsw20241015\test\test_保存.py"))
    # 定义文件存储路径，确保它位于项目文件夹内
    file_path1 = os.path.join(script_dir1, signature_file)
    try:
        with open(file_path1, 'wb') as f:
            pickle.dump(signatureList, f)
        print(f"Signature saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")
    print(signatureList)


    print(len(signatureList[0]))
    # def calSimilarity(signatureSet1, signatureSet2):
    # # 初始化相似性矩阵，大小为 signatureList 长度的平方
    # similarityMatrix = [[1 for _ in range(len(signatureList))] for _ in range(len(signatureList))]
    # print(similarityMatrix)
    # # 遍历 signatureList，计算每个元素与其他元素之间的相似度
    # for i in range(len(signatureList)):
    #     for j in range(len(signatureList)):
    #         if i != j:
    #             # 计算相似度并存储在相似性矩阵中
    #             similarity = calSimilarity(signatureList[i], signatureList[j])
    #             similarityMatrix[i][j] = similarity
    #             print(f"第{i}和第{j}条序列的相似度为{similarity}")
    # # 现在相似性矩阵中包含了 signatureList 中所有元素之间的相似度
    # print(similarityMatrix)

    # 初始化相似性矩阵，大小为 signatureList 长度的平方
    start_calculate_similar_time = time.time()
    similarityMatrix = [[1 for _ in range(len(signatureList))] for _ in range(len(signatureList))]
    print(similarityMatrix)
    # 遍历 signatureList，计算每个元素与其他元素之间的相似度
    for i in range(len(signatureList)):
        for j in range(i+1,len(signatureList)):
            if i != j:
                # 计算相似度并存储在相似性矩阵中
                similarity = calSimilarity(signatureList[i], signatureList[j])
                similarityMatrix[i][j] = similarity
                similarityMatrix[j][i] = similarity
                print(f"第{i}和第{j}条序列的相似度为{similarity}")
    stop_calculate_similar_time = time.time()
    # 现在相似性矩阵中包含了 signatureList 中所有元素之间的相似度
    print(similarityMatrix)


    # 将特征矩阵存储到文件中
    # 使用
    save_compressed_pickle(similarityMatrix, feature_file)
    print(f"similarityMatrix saved to {file_path}")
    # try:
    #     with open(file_path, 'wb') as f:
    #         pickle.dump(similarityMatrix, f)
    #     print(f"similarityMatrix saved to {file_path}")
    # except Exception as e:
    #     print(f"An error occurred while saving the file: {e}")



#定义HNSW类
class HNSW(object):
    # self._graphs[level][i] contains a {j: dist} dictionary,where j is a neighbor of i and dist is distance
    def l2_distance(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return np.linalg.norm(a - b)
    def cosine_distance(self, a, b):
        # try:
        return np.dot(a, b)/(np.linalg.norm(a)*(np.linalg.norm(b)))
        # except ValueError:
        #     print(a)
        #     print(b)
    def juccard_distance(self, a, b):
        a_set = set(a)
        b_set = set(b)
        # 计算两个集合的交集
        intersection = len(a_set.intersection(b_set))
        # 计算两个集合的并集
        union = len(a_set.union(b_set))
        # 计算Jaccard相似度
        jaccard_index = intersection / union
        return jaccard_index
    def lsh_distance(self, a, b):
        # for i in signature
        a1=signature.index(a)
        a2=signature.index(b)
        lsh_distance1=1-feature_matrix_normalization[a1][a2]
        return lsh_distance1

# np.ot(a, b) 计算向量 a 和 b 的点积。
# np.linalg.norm(a) 和 np.linalg.norm(b) 分别计算向量 a 和 b 的 L2 范数（即它们的长度）。
# np.dot(a, b)/(np.linalg.norm(a)*(np.linalg.norm(b))) 计算向量 a 和 b 之间的余弦相似度。
# 如果在计算过程中出现 ValueError（例如，如果两个向量的长度不一致），则打印出这两个向量。

    def _distance(self, x, y):
        return self.distance_func(x, [y])[0]
    def vectorized_distance_(self, x, ys):
        pprint.pprint([self.distance_func(x, y) for y in ys])
        return [self.distance_func(x, y) for y in ys]
    def __init__(self, distance_type, m=5, ef=200, m0=None, heuristic=True, vectorized=False):#heuristic是否使用启发式方法来优化图的构建和搜索过程，vectorized是否使用向量化操作来加速距离计算。如果设置为 False，则算法不会使用向量化操作，可能是因为数据结构或操作不支持向量化，或者向量化在这种情况下不会带来性能提升。
        self.data = []  # 存储数据点的列表
        # 根据距离类型设置距离函数
        if distance_type == "l2":
            distance_func = self.l2_distance
        elif distance_type == "cosine":
            distance_func = self.cosine_distance
        elif distance_type == "juccard":
            distance_func = self.juccard_distance
        elif distance_type == "lsh":
            distance_func = self.lsh_distance
        else:
            raise TypeError('Please check your distance type!')
        self.distance_func = distance_func
        # 根据是否向量化的设置距离计算方法
        if vectorized:
            self.distance = self._distance
            self.vectorized_distance = distance_func
        else:
            self.distance = distance_func
            self.vectorized_distance = self.vectorized_distance_
        #设置算法参数
        self._m = m #每个层级中保留的最大邻居数量
        self._ef = ef #代表“扩展因子”（expansion factor），它用于控制搜索过程中的精度和性能。ef=200 意味着在搜索最近邻时，算法将考虑最多 200 个候选邻居。
        self._m0 = 2 * m if m0 is None else m0   #初始化图结构时，表示第一层（最顶层）的邻居数量限制。如果 m0 设置为 None，则可能意味着第一层的邻居数量没有特定的限制，或者它将使用与其它层级相同的 m 值
        self._level_mult = 1 / log2(m)
        self._graphs = []  # 存储图的列表
        self._enter_point = None
        # 设置选择方法
        self._select = (self._select_heuristic if heuristic else self._select_naive)
    def add(self, elem, ef=None):#向图中添加一个新的数据点
        if ef is None:
            ef = self._ef
        # 获取距离函数、数据列表、图列表和入口点
        distance = self.distance
        data = self.data
        graphs = self._graphs
        point = self._enter_point
        m = self._m
        # 计算插入级别
        level = int(-log2(random()) * self._level_mult) + 1
        # print("level: %d" % level)
        # 添加元素并获取其索引
        # elem will be at data[idx]
        idx = len(data)
        data.append(elem)
        # 如果图不为空，则从入口点开始搜索
        if point is not None:  # the HNSW is not empty, we have an entry point
            dist = distance(elem, data[point])
            # 在不需要插入的级别上搜索最近邻
            # for all levels in which we dont have to insert elem,
            # we search for the closest neighbor
            for layer in reversed(graphs[level:]):
                point, dist = self._search_graph_ef1(elem, point, dist, layer)
            # at these levels we have to insert elem; ep is a heap of entry points.
            # 在需要插入的级别上，使用堆来存储入口点
            ep = [(-dist, point)]
            # pprint.pprint(ep)
            # 在第一层和其他层插入元素
            layer0 = graphs[0]
            for layer in reversed(graphs[:level]):
                level_m = m if layer is not layer0 else self._m0
                # navigate the graph and update ep with the closest
                # nodes we find
                ep = self._search_graph(elem, ep, layer, ef)
                # insert in g[idx] the best neighbors
                layer[idx] = layer_idx = {}
                self._select(layer_idx, ep, level_m, layer, heap=True)
                # assert len(layer_idx) <= level_m
                # insert backlinks to the new node
                for j, dist in layer_idx.items():
                    self._select(layer[j], (idx, dist), level_m, layer)
                    # assert len(g[j]) <= level_m
                # assert all(e in g for _, e in ep)
        for i in range(len(graphs), level):
            # for all new levels, we create an empty graph
            graphs.append({idx: {}})
            self._enter_point = idx
    def balanced_add(self, elem, ef=None):#用于将一个新的元素（elem）添加到多层图中，同时保持图的平衡，这个方法确保了在添加新元素时，多层图结构保持平衡，从而优化了搜索性能
        if ef is None:
            ef = self._ef # 如果 ef 未指定，则使用默认值
        # 获取距离函数、数据列表、图列表和入口点
        distance = self.distance
        data = self.data
        graphs = self._graphs
        point = self._enter_point
        m = self._m
        m0 = self._m0
        # 添加元素到数据列表并获取其索引
        idx = len(data)
        data.append(elem)
        if point is not None:# 如果图不为空
            dist = distance(elem, data[point])# 计算新元素与入口点的距离
            pd = [(point, dist)] # 初始化一个列表来存储每个层的最近邻和距离
            # 从倒数第二层开始向上搜索，找到每层的最近邻
            #pprint.pprint(len(graphs))
            for layer in reversed(graphs[1:]):
                point, dist = self._search_graph_ef1(elem, point, dist, layer)
                pd.append((point, dist))# 将最近邻和距离添加到列表中
            # 遍历所有层，为新元素建立邻居关系
            for level, layer in enumerate(graphs):
                # print('\n')
                # pprint.pprint(layer)
                level_m = m0 if level == 0 else m # 第一层使用 m0，其他层使用 m
                candidates = self._search_graph(
                    elem, [(-dist, point)], layer, ef)  # 搜索邻居
                layer[idx] = layer_idx = {}   # 为新元素创建一个邻居字典
                self._select(layer_idx, candidates, level_m, layer, heap=True)
                # add reverse edges
                # 添加反向边，确保图中每个节点都有指向新元素的反向链接
                for j, dist in layer_idx.items():
                    self._select(layer[j], [idx, dist], level_m, layer)
                    assert len(layer[j]) <= level_m
                # 如果邻居数量少于 m 或 m0，则不需要继续添加到更高级别
                if len(layer_idx) < level_m:
                    return
                # 如果当前层的邻居中有任何节点在下一层中没有表示，则不需要继续添加
                if level < len(graphs) - 1:
                    if any(p in graphs[level + 1] for p in layer_idx):
                        return
                # 更新 point 和 dist 为下一层的最近邻和距离
                point, dist = pd.pop()
        # 如果所有层都处理完毕，则在图列表中添加一个新层，并将新元素作为入口点
        graphs.append({idx: {}})
        self._enter_point = idx
    def search(self, q, k=None, ef=None): #用于在多层图中找到与查询点 q 最接近的 k 个点。
        """Find the k points closest to q."""
        # 获取距离函数、图列表和入口点
        distance = self.distance
        graphs = self._graphs
        point = self._enter_point
        # 如果 ef 未指定，则使用默认值
        if ef is None:
            ef = self._ef
        # 如果图是空的，则抛出异常
        if point is None:
            raise ValueError("Empty graph")
        # 计算查询点 q 与入口点的距离
        dist = distance(q, self.data[point])
        # 从顶层开始向下搜索到第二层，找到最接近的邻居
        # look for the closest neighbor from the top to the 2nd level
        for layer in reversed(graphs[1:]):
            point, dist = self._search_graph_ef1(q, point, dist, layer)
        # 在底层搜索 ef 个邻居
        # look for ef neighbors in the bottom level
        ep = self._search_graph(q, [(-dist, point)], graphs[0], ef)
        # 如果 k 已指定，则从 ep 中选出距离最近的 k 个点
        if k is not None:
            ep = nlargest(k, ep)
        else:
            # 如果 k 未指定，则按距离降序排序 ep
            ep.sort(reverse=True)
        # 返回邻居的索引和距离，距离取负值以便按距离升序排序
        return [(idx, -md) for md, idx in ep]
    def _search_graph_ef1(self, q, entry, dist, layer):#用于在给定层的图中搜索与查询点 q 最接近的 1 个点
        """Equivalent to _search_graph when ef=1."""
        # 获取类中定义的向量化距离计算方法和数据列表
        vectorized_distance = self.vectorized_distance
        data = self.data
        # 初始化最佳点、最佳距离和候选点列表
        best = entry
        best_dist = dist
        candidates = [(dist, entry)]# 使用堆（优先队列）来保持候选点的顺序
        visited = set([entry])# 已访问的点的集合
        # 当候选点列表不为空时，继续搜索
        while candidates:
            # 从候选点列表中弹出距离最小的点
            dist, c = heappop(candidates)
            # 如果当前点的距离大于最佳距离，则停止搜索
            if dist > best_dist:
                break
            # 获取当前点的未访问邻居
            edges = [e for e in layer[c] if e not in visited]
            # 将这些邻居添加到已访问集合中
            visited.update(edges)
            # 计算查询点 q 与这些邻居的距离
            dists = vectorized_distance(q, [data[e] for e in edges])
            # 遍历邻居和对应的距离
            for e, dist in zip(edges, dists):
                # 如果找到更近的距离，则更新最佳点和最佳距离
                if dist < best_dist:
                    best = e
                    best_dist = dist
                    # 将新的最佳点添加到候选点列表中
                    heappush(candidates, (dist, e))
                    # 由于 ef=1，找到更近的点后可以立即停止搜索
                #break
        # 返回最佳点和最佳距离
        return best, best_dist
    def _search_graph(self, q, ep, layer, ef):#于在给定层的图中搜索与查询点 q 最接近的 ef 个点
        # 获取类中定义的向量化距离计算方法和数据列表
        vectorized_distance = self.vectorized_distance
        data = self.data
        # 初始化候选点列表，使用负距离以便使用最小堆（优先队列）
        candidates = [(-mdist, p) for mdist, p in ep]
        heapify(candidates) # 将列表转换为堆
        visited = set(p for _, p in ep)# 创建已访问点的集合
        while candidates:
            # 从候选点列表中弹出距离最小的点
            dist, c = heappop(candidates)
            mref = ep[0][0]
            if dist > -mref: # 获取当前 ef 个邻居中的最大距离
                break # 如果当前点的距离大于 ef 个邻居中的最大距离，则停止搜索
            # 获取当前点的未访问邻居
            # pprint.pprint(layer[c])
            edges = [e for e in layer[c] if e not in visited]
            # 将这些邻居添加到已访问集合中
            visited.update(edges)
            # 计算查询点 q 与这些邻居的距离
            dists = vectorized_distance(q, [data[e] for e in edges])
            # 遍历邻居和对应的距离
            for e, dist in zip(edges, dists):
                mdist = -dist# 取负值以便使用最小堆
            # 如果 ep 中的元素少于 ef，则直接添加新的候选点
                if len(ep) < ef:
                    heappush(candidates, (dist, e))
                    heappush(ep, (mdist, e))
                    mref = ep[0][0]# 更新 ef 个邻居中的最大距离
            # 如果 ep 已满且新点的距离小于 ef 个邻居中的最大距离，则替换最大距离的点
                elif mdist > mref:
                    heappush(candidates, (dist, e))
                    heapreplace(ep, (mdist, e))
                    mref = ep[0][0] # 更新 ef 个邻居中的最大距离
        # 返回 ef 个最近邻的列表
        return ep
    def _select_naive(self, d, to_insert, m, layer, heap=False):#用于从给定的候选点中选择最近的 m 个点

        if not heap:# 不使用堆的简单选择方法
            idx, dist = to_insert
            assert idx not in d # 确保待插入的索引不在字典中
            if len(d) < m:# 如果字典中的元素数量小于m，直接添加新元素
                d[idx] = dist
            else:# 找出字典中距离最大的元素
                max_idx, max_dist = max(d.items(), key=itemgetter(1))
                if dist < max_dist:# 如果新元素的distance小于最大distance，则替换掉距离最大的元素
                    del d[max_idx]
                    d[idx] = dist
            return
        # 使用堆进行选择
        assert not any(idx in d for _, idx in to_insert) # 确保待插入的索引都不在字典中
        to_insert = nlargest(m, to_insert)  # smallest m distances# 从待插入元素中选出distance最小的m个
        unchecked = m - len(d)# 计算可以不检查直接添加的新元素数量
        assert 0 <= unchecked <= m
        to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]# 将待插入元素分为无需检查和需要检查两部分
        to_check = len(checked_ins)
        if to_check > 0:
            checked_del = nlargest(to_check, d.items(), key=itemgetter(1))# 找出字典中距离最大的to_check个元素
        else:
            checked_del = []
        for md, idx in to_insert:
            d[idx] = -md# 将无需检查的新元素添加到字典中，距离取负值（为了与堆兼容）
        zipped = zip(checked_ins, checked_del)
        for (md_new, idx_new), (idx_old, d_old) in zipped:
            if d_old <= -md_new:
                break# 如果字典中的旧distance不大于新distance，则停止替换
            del d[idx_old] # 移除距离较大的旧元素
            d[idx_new] = -md_new  # 添加距离较小的元素
            assert len(d) == m  # 确保字典的大小始终为m
    def _select_heuristic(self, d, to_insert, m, g, heap=False):#用于选择一组元素插入到图中，同时遵循某种启发式策略
        nb_dicts = [g[idx] for idx in d]
        def prioritize(idx, dist):
            return any(nd.get(idx, float('inf')) < dist for nd in nb_dicts), dist, idx
        if not heap:
            idx, dist = to_insert
            to_insert = [prioritize(idx, dist)]
        else:
            to_insert = nsmallest(m, (prioritize(idx, -mdist) for mdist, idx in to_insert))
        assert len(to_insert) > 0
        assert not any(idx in d for _, _, idx in to_insert)
        unchecked = m - len(d)
        assert 0 <= unchecked <= m
        to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]
        to_check = len(checked_ins)
        if to_check > 0:
            checked_del = nlargest(to_check, (prioritize(idx, dist) for idx, dist in d.items()))
        else:
            checked_del = []
        for _, dist, idx in to_insert:
            d[idx] = dist
        zipped = zip(checked_ins, checked_del)
        for (p_new, d_new, idx_new), (p_old, d_old, idx_old) in zipped:
            if (p_old, d_old) <= (p_new, d_new):
                break
            del d[idx_old]
            d[idx_new] = d_new
            assert len(d) == m
    def __getitem__(self, idx):
        for g in self._graphs:
            try:
                yield from g[idx].items()
            except KeyError:
                return




if __name__ == "__main__":
    dim = 500
    num_elements = 1000
    script_dir1 = os.path.dirname(
        os.path.abspath("D:\研究项目相关代码\hnsw-python-master\minhash-lsh-hnsw20241015\test\test_保存.py"))
    # # 定义文件存储路径，确保它位于项目文件夹内
    file_path1 = os.path.join(script_dir1, signature_file)
    try:
        # 文件存在，直接加载
        with open(file_path1, 'rb') as f:
            signature = pickle.load(f)
        print(f"Feature matrix normalization loaded from {file_path1}")
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
    # import h5py
    from progressbar import *
    import pickle

    #构建hnsw并且计时
    start_build_hnsw_time = time.time()
    hnsw = HNSW('lsh', m0=10, ef=150)
    for i in range(sequence_number_train_start):
        if i == sequence_number_train_start_subtract1:
           break
        # print(i)
        hnsw.balanced_add(signature[i])
        #hnsw.add(signature[i])
        # hnsw.balanced_add(signature[i])
        # hnsw.add(data[i])
    stop_build_hnsw_time = time.time()

    # for index, i in enumerate(data):
    #      if index % 100 == 0:#每隔100个元素打印训练进度：
    #          pprint.pprint('train No.%d' % index)
    #      #hnsw.balanced_add(i)
    #      hnsw.add(i)
    #将hnsw 对象序列化并保存到指定的二进制文件中
    #with open('glove-25-angular-balanced-128.ind', 'wb') as f:
    #picklestring = pickle.dump(hnsw, f, pickle.HIGHEST_PROTOCOL)
    #记录添加点后的时间并执行搜索操作：
#记录添加点后的时间，然后使用 hnsw.search 搜索一个随机生成的向量（大小为1x500，转换为32位浮点数），只搜索最近邻的1个点，并记录搜索完成后的时间
    # idx = hnsw.search(np.float32(np.random.random((1, 500))), 1)
    # idx = hnsw.search(signature[5], 10)
    # search_time = time.time()
    # pprint.pprint(idx)
    # # pprint.pprint("add point time: %f" % (add_point_time - time_start))
    # pprint.pprint("本次搜索时间为: %f" % (search_time - add_point_time))
    # print('\n')
    print("HNSW每层节点以及每个节点的连接情况：")
    pprint.pprint(hnsw._graphs)
    print("HNSW每层数目（从0层开始）：")
    for n in hnsw._graphs:
         pprint.pprint(len(n))
    print("HNSW总层数：")
    pprint.pprint(len(hnsw._graphs))
    # print(hnsw.data)
    print(signature[5])
    # 请求用户输入一系列序列
    user_input = input("请输入你要搜索的序列: ")
    # s="CCGTGGGGATTCGTCCCCATTGAGATAGCACCCTTTGTTCATGAGTACCCTCGTTTCCTCGGCGGGCTCGCCCGCCAGCAGGACAACTTCAAACCCTTTGCAGTAGCAGTAACTTCAGTTAATAACAAATATTAAAACTTTCAACAACGGATCTCTTGGTTCTGGCATCGATGAAGAACGCAGCGAAATGCGATAAGTAGTGTGAATTGCAGAATTCAGTGAATCATCGAATCTTTGAACGCACATTGCGCCCTTCGGTATTCCGTTGGGCATGCCTGTTCGAGCGTCATTTAAACCTTCAAGCTATGCTTGGTGTTGGGTGTCTGTCCCGCCTCAGCGCGTGGACTCGCCTCAAATCCATTGGCAGCCGGTATGTTGGCTTCGTGCGCAGCACATTGCAAGCGGAACCATCAGACCCCCTCCC"
    # #将序列其转换成数字签名
    # # num_shingles_1hot = len(shingles_1hot)
    # # k = 500  # 假设我们想要500个哈希函数
    # # signature = []
    # # for i in range(num_shingles_1hot):
    # #     signature.append(generate_minhash_signature(shingles_1hot[i], k))
    # # print(signature)
    # # shingle 尺寸
    # # user_input = list(user_input)
    # k_user = 5
    # # 建立shingles
    # shingles_user = []
    # shingles_user.append(build_shingles(user_input, k_user))
    # vocab_user = build_vocab(shingles_user)
    # # 用one-hot对shingles进行编码
    # shingles_1hot_user = []
    # for shingle_set in shingles_user:
    #     shingles_1hot_user.append(one_hot(shingle_set, vocab_user))
    # # 堆叠成一个单一的NumPy数组
    # shingles_1hot_user = np.stack(shingles_1hot_user)
    # shingles_1hot_user.shape
    # # # 对每个哈希函数进行操作
    # # for i in range(k):
    # #     a, c, m = a_values[i], c_values[i], m_values[i]
    # #     min_hash_value = np.inf
    # #     for j, shingle in enumerate(shingles_1hot_user):
    # #         if shingle == 1:
    # #             hash_value = linear_congruential_hash(a, c, m, j)
    # #             if hash_value < min_hash_value:
    # #                 min_hash_value = hash_value
    # #     signature[i] = min_hash_value
    # # return signature
    # # 使用
    # num_shingles_1hot_user = len(shingles_1hot_user)
    # k2 = 500  # 假设我们想要500个哈希函数
    # signature_user = []
    # for i in range(num_shingles_1hot_user):
    #     signature_user.append(generate_minhash_signature(shingles_1hot_user[i], k2))
    # print(signature_user)
    # signature_user=signature_user[0]
    # print(signature_user)
#622-664打了注释
    indexss=0
 # 遍历列表以查找序列
    for i, seq in enumerate(sequences):
        if seq == user_input:
            indexss = i
            break  # 找到后跳出循环
    add_point_time = time.time()
    idx = hnsw.search(signature[indexss], 1)
    # idx = hnsw.search(signature[5], 10)
    search_time = time.time()
    pprint.pprint(idx)



# for index, i in enumerate(data):
#     idx = hnsw.search(i, 1)
#     pprint.pprint(idx[0][0])
#     pprint.pprint(i)
#     pprint.pprint(hnsw.data[idx[0][0]])
# pprint.pprint('------------------------------')
# pprint.pprint(hnsw.data)
# pprint.pprint('------------------------------')
# pprint.pprint(data)
# pprint.pprint('------------------------------')
# pprint.pprint(hnsw._graphs)
# pprint.pprint(len(hnsw._graphs))
# print(signature_user)
CandidatePair_Classification_Information=[]
Originalorder_Classification_Information=[]
# k__Fungi;p__Fungi_phy_Incertae_sedis;c__Fungi_cls_Incertae_sedis;o__Fungi_ord_Incertae_sedis;f__Fungi_fam_Incertae_sedis;g__Fungi_gen_Incertae_sedis;s__Fungi_sp
def extract_taxonomy_info(fasta_file, sequence_number):
    with open(fasta_file, 'r') as file:
        current_index = 0
        taxonomy_info = None
        for line in file:
            line = line.strip()
            if line.startswith('>'):  #FASTA格式中的描述行以'>'开头
                if current_index == sequence_number:
                    # 分割描述行以提取分类信息
                    description_parts = line.split('|')
                    # 分类信息位于描述行的第五部分，以分号分隔
                    taxonomy_parts = description_parts[1].split(';')
                    # 移除最后一个元素（通常是序列的额外标识符）
                    taxonomy_info = ' '.join(taxonomy_parts)
                    break  # 找到所需序列后停止循环
                current_index += 1
    if taxonomy_info:
        print(f"与原序列最相似的为：第{sequence_number}条序列，其的分类信息是：{taxonomy_info}")
        CandidatePair_Classification_Information.append(taxonomy_info)
    else:
        print(f"未找到第{sequence_number}条序列的分类信息。")
        return taxonomy_info
def extract_taxonomy_info1(fasta_file, sequence_number):
    with open(fasta_file, 'r') as file:
        current_index = 0
        taxonomy_info = None
        for line in file:
            line = line.strip()
            if line.startswith('>'):  #FASTA格式中的描述行以'>'开头
                if current_index == sequence_number:
                    # 分割描述行以提取分类信息
                    description_parts = line.split('|')
                    # 分类信息位于描述行的第五部分，以分号分隔
                    taxonomy_parts = description_parts[1].split(';')
                    # 移除最后一个元素（通常是序列的额外标识符）
                    # taxonomy_info = ' '.join(taxonomy_parts[:-1])
                    taxonomy_info = ' '.join(taxonomy_parts)
                    break  # 找到所需序列后停止循环
                current_index += 1
    if taxonomy_info:
        print(f"原输入序列的信息为：第{sequence_number}条序列，其的分类信息是：{taxonomy_info}")
        Originalorder_Classification_Information.append(taxonomy_info)
    else:
        print(f"未找到第{sequence_number}条序列的分类信息。")
    return taxonomy_info
# 查找序列在列表中的位置

try:
    index = sequences.index(user_input)
    print(f"输入的序列位于序列列表的第 {index} 位。")
except ValueError:
    print("输入的序列在序列列表中未找到。")
sequence_number = index
taxonomy = extract_taxonomy_info1(fasta_file, sequence_number)
# if taxonomy:
#     print(f"第{sequence_number}条序列的分类信息是：{taxonomy}")
# else:
#     print(f"未找到第{sequence_number}条序列的分类信息。")
# sequences=user_input
print("与其相似的序列的索引号以及相似度为",idx)
# 将从提供的列表中提取每个元组的第一位数字，并将它们存储在新的列表hx中。
hx = [tup[0] for tup in idx]
print("与原序列相似的序列的索引号列表为",hx)
print("输入原序列通过检索到的前几条数据信息为：")
for number in hx:
    extract_taxonomy_info(fasta_file, number)
print("原序列分类信息为：",Originalorder_Classification_Information)
print("候选对的分类信息为：",CandidatePair_Classification_Information)

# 定义函数来计算分类信息的匹配概率
def calculate_match_probability(original, candidates):
    # 将原始分类信息分割成列表
    original_taxonomy = original[0].split(' ')
    print(original_taxonomy)
    # 初始化结果列表
    match_results = [0,0,0,0,0,0,0]
    # 遍历每个候选分类信息
    for candidate in candidates:
        # 将候选分类信息分割成列表
        candidate_taxonomy = candidate.split(' ')
        print(candidate_taxonomy)
        # # 计算匹配的分类等级数量
        # match_count = sum(1 for o, c in zip(original_taxonomy, candidate_taxonomy) if o == c)
        i=0
        for i in range(7):
            if candidate_taxonomy[i]==original_taxonomy[i]:
                match_results[i]=match_results[i]+1
    return match_results

# 计算匹配概率
match_probabilities = calculate_match_probability(Originalorder_Classification_Information,CandidatePair_Classification_Information)
print("输入序列和查找到的最相似序列的“界门纲目科属种”的相似列表为:",match_probabilities)

print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

match_results = [0, 0, 0, 0, 0, 0,0]
def calculate_match_probability1(original, candidates):
    # 将原始分类信息分割成列表
    original_taxonomy = original[0].split(' ')
    print("分割完的原序列为：",original_taxonomy)
    # 初始化结果列表
    # 遍历每个候选分类信息
    for candidate in candidates:
        # 将候选分类信息分割成列表
        candidate_taxonomy = candidate.split(' ')
        print("分割完的候选对序列为：",candidate_taxonomy)
        # # 计算匹配的分类等级数量
        # match_count = sum(1 for o, c in zip(original_taxonomy, candidate_taxonomy) if o == c)
        i=0
        for i in range(7):
            if candidate_taxonomy[i]==original_taxonomy[i]:
                match_results[i]=match_results[i]+1
    return match_results

#循环计时
start_search_time = time.time()
for sequence in sequences[sequence_number_train_start:sequence_number_all]:
    Originalorder_Classification_Information = []
    CandidatePair_Classification_Information = []
    try:
        index = sequences.index(sequence)
    except ValueError:
        print("输入的序列在序列列表中未找到。")
    sequence_number = index
    #获得分类信息
    taxonomy = extract_taxonomy_info1(fasta_file, sequence_number)
    print("原序列分类信息为:",taxonomy)
    #查找
    idx = hnsw.search(signature[index], kn)
    print("与其相似的序列的索引号以及相似度为",idx)
    # 将从提供的列表中提取每个元组的第一位数字，并将它们存储在新的列表hx中。
    hx=[]
    hx = [tup[0] for tup in idx]
    print("与原序列相似的序列的索引号列表为",hx)
    print("输入原序列通过检索到的前几条数据信息为：")
    for number in hx:
        extract_taxonomy_info(fasta_file, number)
    print("原序列分类信息为：", Originalorder_Classification_Information)#kon
    print("候选对的分类信息为：", CandidatePair_Classification_Information)#太多
    # 解析分类信息并统计每个级别的出现次数
    classification_counts = defaultdict(lambda: defaultdict(int))
    for info in CandidatePair_Classification_Information:
        parts = info.split(' ')
        for part in parts:
            level, name = part.split('__')
            classification_counts[level][name] += 1
    print(classification_counts)
    # 找出出现次数最多的分类级别
    most_common_levels = {level: max(counts, key=counts.get) for level, counts in classification_counts.items()}
    print(most_common_levels)
    # 将字典转换成列表，格式为：'k__Fungi p__Basidiomycota c__Agaricomycetes o__Russulales f__Amylostereaceae g__Amylostereum s__Amylostereum_areolatum'
    most_common_levels_list = ' '.join(f"{level}__{name}" for level, name in most_common_levels.items())
    # 将字符串转换为列表，列表中只有一个元素
    most_common_levels_list = [most_common_levels_list]
    print(most_common_levels_list)
    # 赋值给CandidatePair_Classification_Information
    CandidatePair_Classification_Information = most_common_levels_list
    result=calculate_match_probability1(Originalorder_Classification_Information,CandidatePair_Classification_Information)
    print("******************************************************************************************************************************************************************************************************************************************************************88")
stop_search_time = time.time()
print("“界门纲目科属种”相似的个数为：")
print(result)



# 使用列表推导式来除以整数
result_list = [x / divisor for x in result]
# 打印结果
print("“界门纲目科属种”的分类准确率为：")
print(result_list)

pprint.pprint("本次提取特征时间为: %f" % (stop_get_feature_time - start_get_feature_time))
pprint.pprint("本次计算相似度时间为: %f" % (stop_calculate_similar_time - start_calculate_similar_time))
pprint.pprint("本次建立HNSW时间为: %f" % (stop_build_hnsw_time - start_build_hnsw_time))
pprint.pprint("本次搜索并且从相似序列找出综合结果的时间为: %f" % (stop_search_time - start_search_time))
pprint.pprint("平均每条搜索并且从相似序列找出综合结果的时间为: %f" % ((stop_search_time - start_search_time)/(sequence_number_all-sequence_number_train_start)))

#CCCTTGCTCTTTCCGACACACCCTGTGCACTCCCGCGTGGGCTGCGGTGACTTCGGTTGCCGTGCCCGCGATTTTATACACTCTTTGTATGTCTCAGAATGTCTTTGCGTGTTGCGCATCTAATACAACTTTCAACAACGGATCTCTTGGCTCTCGCATCGATGAAGAACGCAGCGAAATGCGATAAGTAATGTGAATTGCAGAATTCAGTGAATCATCGAATCTTTGAACGCACCTTGCGCCCCTTGGTATTCCGAGGGGCACACCTGTTTGAGTGTCGTGAATTTCTCAACTCCGCCTCCTTTGCGGGAGCGCGGGGCTTGGAGTTGGAGGCTTGCGGGCGTAAGTCCGCTCCTCTCAAATGCATTAGTGAAGCCCAGGTGGCCTCGGCGTGATAATTGTCTACGTCCAAGGTTGTCTGTAGCG