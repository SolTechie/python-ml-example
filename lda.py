# coding=utf-8
import re

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer

#基础词库 空格分隔
base_words = ["苹果 雪梨 如龙 火龙果 黄瓜 拳皇 樱桃 橙子 柿子 葡萄 西瓜 刺客信条 哈密瓜 超级玛丽 石榴 极品飞车 香蕉 三国志"]

#基础语料库 每条语料空格分隔
corpus = [
    "我 买 了 西瓜 和 橙子",
    "拳皇 和 如龙 都 是 格斗 类 游戏",
    "我 最 喜欢 吃 樱桃 葡萄 柿子 橙子",
    "小明 是 刺客信条 的 忠实 玩家 ， 同时 对于 三国志 类 的 游戏 也 特别 精通"
]

cv_model = CountVectorizer()
cv_model.fit(base_words)
#保存词频计数模型
#joblib.dump(cv_model,"cv_model_path/cv_model.m",compress=0)
#加载之前保存的词频计数模型
#cv_model = joblib.load("cv_model_path/cv_model.m")

corpus_cv = cv_model.transform(corpus)


lda_model = LatentDirichletAllocation(n_topics=2, #预定义2个主题
    learning_method='online', 
    n_jobs=-1, #最大限度利用并行
    max_iter=10,
    max_doc_update_iter=100, 
    verbose=1000)
lda_model.fit(corpus_cv)
#保存训练好的模型
#joblib.dump(lda_model,"lda_model_path/lda_model.m",compress=0)
#加载保存的模型
#lda_model = joblib.load('lda_model_path/lda_model.m')


#lda_model.components_可以查看具体的主题构成分布
# print("-----------------------------------")
# print(lda_model.components_)
# print("-----------------------------------")

#预测主题的语料 列表中每个字符串元素是一个语料 单个语料分词用空格分隔 以下是两个语料
p_corpus = [
    "葡萄 和 樱桃 都 可以 做成 酒",
    "刺客信条 是 ARPG 游戏 ， 三国志 是 策略 类 游戏 ， 如龙 和 拳皇 都是 格斗 类 游戏"
]

#预测 先词频数量化 然后过LDA模型
p_corpus_cv = cv_model.transform(p_corpus)
topic_dis = lda_model.transform(p_corpus_cv)

# 预测结果 二维列表 每一行代表参与预测的每一个语料 每行中的列代表该语料中各个主题的分布 如下是前述预测主题语料的结果
# 葡萄和樱桃都可以做成酒 偏向与第2个主题 0.81248534
# 刺客信条是ARPG游戏，三国志是策略类游戏，如龙和拳皇都是格斗类游戏 偏向第1个主题 0.67633699
# [
#     [ 0.18751466  0.81248534]
#     [ 0.67633699  0.32366301]
# ]




