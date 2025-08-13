import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import re
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import os
import jieba
import fugashi

def removePunctuation(text, iflower=False):
    punctuation = '、！!,;:：，“”?"\''
    text = re.sub(r'[{}]+'.format(punctuation), '', text)
    if iflower:
        return text.lower()
    else:
        return text.strip()

# 使用 fugashi 作为日语分词器
tagger = fugashi.Tagger()
# tagger = fugashi.Tagger(ipadic.MECAB_ARGS)
# dic_dir = os.path.abspath("mecab-ipadic")
# assert os.path.exists(os.path.join(dic_dir, "dicrc"))
# tagger = fugashi.Tagger(f'-d "{dic_dir}"')


# 加载三种语言的文本
english = open('./Iamacat_en.txt', 'r', encoding='utf-8').readlines()
chinese = open('./Iamacat_ch.txt', 'r', encoding='utf-8').readlines()
japanese = open('./Iamacat_jp.txt', 'r', encoding='utf-8').readlines()

# 英文分句 + 分词
english_sentences = []
for s in english:
    s = s.strip()
    if s == '':
        continue
    s = removePunctuation(s, True)
    sl = s.split('.')
    for si in sl:
        if si != '':
            english_sentences.append(si.strip().split(' '))

# 中文分句 + jieba 分词
chinese_sentences = []
for s in chinese:
    s = s.strip()
    if s == '':
        continue
    s = removePunctuation(s)
    sl = s.split('。')
    for si in sl:
        if si != '':
            chinese_sentences.append(list(jieba.cut(si.strip())))

# 日文分句 + fugashi 分词
japanese_sentences = []
for s in japanese:
    s = s.strip()
    if s == '':
        continue
    s = removePunctuation(s)
    sl = s.split('。')
    for si in sl:
        if si != '':
            tokens = [word.surface for word in tagger(si.strip())]
            japanese_sentences.append(tokens)

# 统计词频
def to_tks(sentences):
    tks = {}
    for s in sentences:
        for w in s:
            tks[w] = tks.get(w, 0) + 1
    return tks

# 获取前500个高频词
def get_top500(tokens_dict):
    Z = sorted(tokens_dict.items(), key=lambda item: item[1], reverse=True)
    return [k for k, v in Z[:500]]

En_tks = to_tks(english_sentences)
Ch_tks = to_tks(chinese_sentences)
Jp_tks = to_tks(japanese_sentences)

top500_en = get_top500(En_tks)
top500_ch = get_top500(Ch_tks)
top500_jp = get_top500(Jp_tks)

# 训练 Word2Vec 模型
english_model = Word2Vec(english_sentences, min_count=1, vector_size=50, window=3, sample=1e-5)
chinese_model = Word2Vec(chinese_sentences, min_count=1, vector_size=50, window=3, sample=1e-5)
japanese_model = Word2Vec(japanese_sentences, min_count=1, vector_size=50, window=3, sample=1e-5)

# 可视化
def display_tsnescatterplot(model, words):
    arr = []
    labels = []

    # 只取模型中实际存在的词
    for w in words:
        if w in model.wv:
            arr.append(model.wv[w])
            labels.append(w)

    # 使用 t-SNE 降维到二维平面
    tsne = TSNE(n_components=2, random_state=0, init='pca')
    # Y = tsne.fit_transform(arr)
    Y = tsne.fit_transform(np.array(arr))

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    # 绘图
    plt.figure(figsize=(12, 10))
    plt.scatter(x_coords, y_coords)

    # 添加注释标签
    for label, x, y in zip(labels, x_coords, y_coords):
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(0, 0),
                     textcoords='offset points',
                     fontsize=9,
                     fontname="SimHei")  # 确保你有这字体或换成 SimHei 等

    plt.xlim(x_coords.min() - 1, x_coords.max() + 1)
    plt.ylim(y_coords.min() - 1, y_coords.max() + 1)
    plt.title("t-SNE visualization of Word2Vec embeddings")
    plt.show()
display_tsnescatterplot(english_model, top500_en)
display_tsnescatterplot(chinese_model, top500_ch)
display_tsnescatterplot(japanese_model, top500_jp)
