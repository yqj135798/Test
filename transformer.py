import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import copy

import matplotlib.pyplot as plt
from torch.autograd import Variable


# 定义Embeddings类来实现文本嵌入层
# 这里继承nn.Module，这样就有标准层的一些功能
# --------------词嵌入层--------------------------------------
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        # 类的初始化，有两个参数，d_model：指词嵌入的维度，vocab_size：指词表的大小
        super(Embeddings, self).__init__()
        # 之后就是调用nn中的预定义层Embedding，获得一个词嵌入对象self.embed
        self.embed = nn.Embedding(vocab_size, d_model)
        # 最后就是将d_model传入类中
        self.d_model = d_model

    def forward(self, x):  # 乘以d_model
        # 可以将其理解为该层的前向传播逻辑，所有层中会有此函数
        # 当传给该类的实例化对象参数时，自动调用该函数
        # 参数x：因为Embedding层是首层，所以代表输入给模型的文本通过词汇映射后的张量

        # 将x传给self.embed并与根号下self.d_model相乘作为结果输出
        return self.embed(x) * math.sqrt(self.d_model)


d_model = 512
vocab_size = 1000

x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
emb = Embeddings(d_model, vocab_size)
embr = emb(x)
# print("embr:", embr)
# print(embr.shape)
#
# ----------------------位置编码层---------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout,  max_len=5000):
        # 位置编码类的的参数共有三个，d_model:词嵌入维度， self_dropout：置0比率， max_len：每个句子的最大长度

        super(PositionalEncoding, self).__init__()
        #实例化nn中预定义的dropout层，并将dropout传入其中，获得对象self.dropout
        self.dropout = nn.Dropout(p=dropout)
        # 初始化一个位置编码矩阵，它是一个0阵，矩阵大小是max_len * d_model
        pe = torch.zeros(max_len, d_model)

        # 初始化一个绝对位置矩阵
        # 所以我们首先使用range方法获得一个连续自然数向量
        # 又因为参数传的是1，代表1矩阵扩展的位置，会使向量变成一个max_len * 1 的矩阵
        position = torch.arange(0, max_len).unsqueeze(1).float()
        # 绝对位置矩阵初始化之后，接下来就是考虑如何将这些位置信息加入到位置编码矩阵中
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        #将前面定义的变化矩阵进行奇数，偶数的分别赋值
        pe[:, 0::2] = torch.sin(position * div_term) #偶数维sin
        pe[:, 1::2] = torch.cos(position * div_term) #奇数维sin

        # 这样就得到了位置编码pe，pe现在还只是一个二维矩阵
        # 就必须拓展一个维度，所以这里使用unsqueeze拓展维度
        pe = pe.unsqueeze(0)

        # 把位置矩阵注册为模型buffer，这个buffer不是模型的参数，不跟随优化器同步跟新
        # 注册之后我们就可以在模型保存后重新加载时和模型结构与参数一同被加载
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 参数x表示文本序列的词嵌入表示
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)

        #最后使用self.dropout对象进行丢弃操作，并返回结果
        return x + self.pe[:, :x.size(1)]

# -----------------参数设置----------------------------------

d_model = 512
dropout = 0.1
max_len = 60

x = embr
pe = PositionalEncoding(d_model, dropout, max_len)
pe_result = pe(x)
# print(pe_result)
# print(pe_result.shape)

# -------------------------------------------------------------------

# ------------可视化-------------------------------------------


# 定义一个画布
# plt.figure(figsize=(15, 5))
#
# # 实例化PositionEncoding类对象，词嵌入维度给20，且置零比率设置为0
# pe = PositionalEncoding(20, 0)
#
# # 向pe中传入一个全零初始化的x， 且相当于展示pe
# y = pe(Variable(torch.zeros(1, 100, 20)))
#
# plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
#
# plt.legend(["dim %d"%p for p in [4, 5, 6, 7]])
# plt.show()  # 显式调用show()确保图表显示

# -----------------------------------------------------------------

# print(np.triu(([1, 2, 3], [4, 5, 6], [7, 8, 9]), k=-1))
# print(np.triu(([1, 2, 3], [4, 5, 6], [7, 8, 9]), k=0))
# print(np.triu(([1, 2, 3], [4, 5, 6], [7, 8, 9]), k=1))

# ---------------------------掩码生成-----------------------------
def subsequent_mask(size):
    # 生成向后遮掩的掩码张量,参数size是掩码张量的大小,它的最后两位维形成一个方阵
    #在函数中,首先定义掩码张量的形状
    attn_shape = (1, size, size)

    # 然后使用np.ones方法向这个形状中添加1元素,形成上三角阵,最后为了节省空间
    # 再使其中的数据类型变为无符号整型unit8
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 最后将numpy类型转化为torch中的tensor,内部做一个1-的操作
    # 在这个其实是做了一个三角阵的反转,subsequent_mask中的每一个元素都会被1减
    # 如果是0,subsequent_mask中的该位置由0转为1
    # 如果是1,subsequent_mask中的该位置由1转为0
    return torch.from_numpy(1 - subsequent_mask)

# size = 5
# sm = subsequent_mask(size)
# # print("sm:", sm)
# plt.figure(figsize=(5, 5))
# plt.imshow(subsequent_mask(20)[0])
# plt.show()

# --------------------------------------------------------
# m = nn.Dropout(p=0.2)
# input1 = torch.randn(4, 5)
# output = m(input1)
# # print(output)
# x = torch.tensor([1, 2, 3, 4])
# y = torch.unsqueeze(x, 0)
# print(y)
# z = torch.unsqueeze(x, 1)
# print(z)
# ------------------------------------------------------------

# x = Variable(torch.randn(5, 5))
# print(x)
#
# mask = Variable(torch.zeros(5, 5))
# print(mask)
#
# y = x.masked_fill(mask == 0, -1e9)
# print(y)



# --------------------注意力机制--------------------------------------

def attention(query, key, value, mask=None, dropout=None):

    # 注意力机制的实现.输入分别是query, key, value, mask:掩码张量,
    # dropout是nn.Dropout层的实例化对象,默认为None
    # 在函数中,首先取query的最后一堆的大小,i一般情况下就等同于我们的希嵌入维度,命名为d_k
    d_k = query.size(-1)
    # 按照注意力公式,将query与key的转置相乘,这里面key是将最后两个维度进行转置,再除以缩放系数
    # 得到注意力的分张量scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  #缩放点积注意力

    # 判断是否使用掩码张量
    if mask is not None:
        # 使用tensor的masked_fill方法,将掩码张量和scores张量每个位置一一比较
        # 则对应的scores张量使用-le9这个值来替换,如下演示
        scores = scores.masked_fill(mask.expand_as(scores) == 0, -1e9)

    # 对scores的最后一维进行softmax操作,使用F.softmac方法,第一个参数是softmax对象,第二个是
    # 这样获得最终的注意力张量
    p_attn = F.softmax(scores, dim=-1)

    # 之后判断是否使用dropout进行随机置0
    if dropout is not None:
        # 将attn传入dropout对象中进行'丢弃'处理
        p_attn = dropout(p_attn)
    # 最后,根据公式将attn与value张量相乘获得最终的query注意力表示,返回之一了张量
    return torch.matmul(p_attn, value), p_attn


# query = key = value = pe_result
# mask = Variable(torch.zeros(2, 4, 4))
# attn, p_attn = attention(query, key, value, mask=mask)
# print('attn:', attn)
# print(attn.shape)
# print('p_attn:', p_attn)
# print(p_attn.shape)


# ---------------------多头注意力机制-----------------------------------------

# x = torch.randn(4, 4)
# print(x.size())
# y = x.view(16)
# print(y.size())
# z = x.view(-1, 8)
# print(z.size())
#
# a = torch.randn(1, 2, 3, 4)
# print(a.size())
# print(a)
#
# b = a.transpose(1, 2)
# print(b.size())
# print(b)
#
# c = a.view(1, 3, 2, 4)
# print(c.size())
# print(c)


# 首先定义一个克隆函数，因为在多头注意力机制的实现中要用到多个结构相同的线性层
# 我们将使用clone函数将他们一同初始化在一个网络层列表对象中，之后的结构也会用到该函数
def clones(module, N):
    # 用于生成相同网络层的克隆函数，它的参数modle表示要克隆的目标网络层，N代表需要克隆的数量
    # 在韩式中我们通过for循环对module进行N次深度拷贝，使其每个module成为独立的层
    # 然后将其放在nn.ModuleList类型的列表中存放
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttention(nn.Module):
    # 在类初始化时，会传入三个参数，head代表头数，embedding_dim代表词嵌入的维度，dropout代表进行dropout操作时的置0比率，默认时0.1

    def __init__(self, head, embedding_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        # 在函数中，首先使用了一个测试中常用的assert语句，判断h是否能被d_model整除
        # 这是因为我们之后要给每个头分配词特征，也就是embedding_dim/head个
        assert embedding_dim % head == 0

        # 使得每个头获得分割词向量维度d_k
        self.d_k = embedding_dim // head
        # 传入头数head
        self.head = head

        # 然后获得线性层对象没通过nn的Linear实例化，它的内部变换矩阵时阿，bedding_dim x embedding_dim

        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)

        # self.attn为None，它代表最后得到的注意力张量模型中还没有结果所以为None
        self.attn = None
        # 最后一个就是self.dropout对象，它通过nn中的Dropout实例化而来，置0比率为传进来的dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # 向前逻辑函数，它的输入参数有四个，前三个就是注意力机制需要的Q，K， V
        # 最后一个是注意力机制中肯需要的mask掩码张量，默认为None

        # 如果存在掩码张量mask
        if mask is not None:
            # 使用unsqueeze拓展维度，代表多头中的第n头
            mask = mask.unsqueeze(1)
        # 接着，我们获得一个batch_size的变量，他是query尺寸的第一个数字，代表有多少条样本
        batch_size = query.size(0)
        # 之后进入多头处理环节
        # def transform(x, linear):
        #     x = linear(x)
        #     return x.view(batch_size, -1, self.h, self.d_k).transpose(1,2)
        # query = transform(query, self.linear_q)
        # query = transform(query, self.linear_k)
        # query = transform(query, self.linear_v)
        query, key, value = \
        [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
         for model, x in zip(self.linears, (query, key, value))]
        # 的哀悼每个图的输入后，接下来就是将他们传入带attention中，
        # 这里直接调用我们之前实现的attention函数， 同时也将mask和dropout传入其中
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1,2).contiguous().view(batch_size, -1, self.head * self.d_k)

        return self.linears[-1](x)

# 实例化若干参数
head = 8
embedding_dim = 512
dropout = 0.2

# 若干输入参数的初始化
query = key = value = pe_result
mask = Variable(torch.zeros(2, 4, 4))
mha = MultiHeadAttention(head, embedding_dim, dropout)
mha_result = mha(query, key, value, mask)
# print(mha_result)
# print(mha_result.shape)



#------------------------------------------------------------------------------------

# ----------------------------------前馈全连接层-------------------------

# 通过类PositionwiseFeedForward来实现前馈全连接层
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        # 初始化后汉书有三个输入参数分别是d_model， d_ff和dropout=0.1
        # 第一个是线性层的输入维度也是第二个线性层的输出维度，第二个d_ff是线性层的输入维度和输出维度，最后一个时dropout的置0比率
        super(PositionwiseFeedForward, self).__init__()

        # 首先按照我们与其使用nn实例化了两个线性层对象，self.w1和self.w2
        # 它们的参数分别是d_model，d_ff和d_ff，d_model
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        # 然后使用nn的dropout实例化了对象self.dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 输入参数为x，代表来自上一层的输出
        # 首先经过第一个线性层，然后使用Funtional中的relu函数进行激活
        # 之后在使用dropout进行随机置0，最后通过第二个线性性w2，返回最终结果
        return self.w2(self.dropout(F.relu(self.w1(x))))

d_model = 512
d_ff = 64
dropout = 0.2

x = mha_result
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
ff_result = ff(x)
# print(ff_result)
# print(ff_result.shape)

# ---------------------------规范化层------------------------------------
# 通过LayerNorm实现规范化层的类
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        # 初始化函数有两个参数,一个是features,表示词嵌入的维度
        # 另一个是eps它是一个足够小的数,在规范化公式的分母中出现,防止分母为0,默认是1e-6
        super(LayerNorm, self).__init__()

        # 根据features的形状初始化两个参数张量a2和b2,第一个初始化为1张量
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        # 把eps传入到类中
        self.eps = eps

    def forward(self, x):
        # 输入参数x代表来自上一层的输出
        # 在函数中首先对输入变量求其最后一个维度的均值,并保持输出维度一输入维度的一致
        mean = x.mean(-1, keepdim=True)
        # 接着对x进行最后一个维度上的求标准差操作
        std = x.std(-1, keepdim=True)
        return self.a2 * (x-mean) / (std + self.eps) + self.b2


features = d_model = 512
eps = 1e-6

x = ff_result
ln = LayerNorm(features, eps)
ln_result = ln(x)
# print(ln_result)
# print(ln_result.shape)
#------------------------------------------------------------------


# --------------------子层连接结构-------------------------------
# 使用SublayerConnection来实现子层连接结构的类
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        # 有了两个参数,一个size表示词嵌入维度大小
        # dropout本身是对模型结构中的节点数进行随机抑制的比率
        # 又因为节点被抑制等效于该节点的输出都是0,因此可以把dropout看作是对输出矩阵的随机置0比率
        super(SublayerConnection, self).__init__()
        # 实例化了规范化对象self.norm
        self.norm = LayerNorm(size)
        # 又使用nn中预定义的dropout实例化一个self.dropout对象
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        # 接收上一层或者子层的输入作为第一个参数
        # 将该子层连接中的子层函数作为第二个参数
        # 将x进行规范化，然后送入子层函数中处理，处理结果进入dropout层，最后进行残差连接
        return x + self.dropout(sublayer(self.norm(x)))

size = d_model = 512
head = 8
dropout = 0.2

x = pe_result
mask = Variable(torch.zeros(2, 4, 4))
self_attn = MultiHeadAttention(head, d_model)

sublayer = lambda x: self_attn(x, x, x, mask)

sc = SublayerConnection(size, dropout)
sc_result = sc(x, sublayer)
# print(sc_result)
# print(sc_result.shape)


# -----------------------------------------------------------------------


# ------------------编码器层-----------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        # 四个参数，size就是词嵌入维度的大小
        # 第二个self_attn，之后我们将传入多头自注意力子层实例化对象，并且是自注意力机制
        # 第三个是feed_forward，之后我们将传入前馈全连接层实例化对象
        # 最后一个dropout是置比率
        super(EncoderLayer, self).__init__()

        # 首先将self_attn和feed_forward传入其中
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 把size传入其中
        self.size = size

        # 编码曾中有两个子层连接结构，所以使用colnes函数进行克隆
        self.sublayer = clones(SublayerConnection(size, dropout), 2)




    def forward(self, x, mask):
        # forward函数中有两个输入参数，x和mask，分别表示上一层的输出，和掩码张量mask
        # 首先通过地图个子层连接结构，其中包含多头自注意力子层
        x = self.sublayer[0](x, lambda  x: self.self_attn(x, x, x, mask))
        return  self.sublayer[1](x, self.feed_forward)

size = d_model = 512
head = 8
d_ff = 64
x = pe_result
dropout = 0.2

self_attn = MultiHeadAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
mask = Variable(torch.zeros(2, 4, 4))

el = EncoderLayer(size, self_attn, ff, dropout)
el_result = el(x, mask)
# print(el_result)
# print(el_result.shape)


#--------------------------编码器---------------------------------

# 使用Encoder类来实现编码器
class Encoder(nn.Module):
    def __init__(self, layer, N):
        # 初始化函数两个参数分别嗲表编码器层和编码器层的个数
        super(Encoder, self).__init__()
        # 首先使用clones函数克隆N个编码器层放在self.layers中
        self.layers = clones(layer, N)
        # 再初始化一个规范层，它将用在编码器最后面
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        # 首先就是对我们克隆的编码器层进行循环，每次的带一个新的x
        # 这个循环过程，就相当于输出x经过了N个编码层的处理
        # 最后再通过规范化层的对象self.norm进行处理最后返回结果
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

size = d_model = 512
d_ff = 64
head = 8
c = copy.deepcopy
attn = MultiHeadAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
dropout = 0.2
layer = EncoderLayer(size, c(attn), c(ff), dropout)
N = 8
mask = Variable(torch.zeros(2, 4, 4))

en = Encoder(layer, N)
en_result = en(x, mask)
# print(en_result)
# print(en_result.shape)



# ------------------------------------------------------------------

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        # 五个参数，size代表词嵌入维度，同时也是代表解码器层的尺寸
        # 第二个是self_attn，多头自注意力对象，也就是说这个注意力机制需要Q=K=V
        # 第三个是src_attn，多头注意力对象，这里Q!=K=V，第四个是前馈全连接层对象
        # 最后就是dropout置0比率
        super(DecoderLayer, self).__init__()
        # 在初始化函数中，主要是将这些输入传到类中
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = dropout
        # 按照结构图使用clones函数俄克拉三个子层连接对象
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        # 共有四个参数，分别是来自上一层的输入x
        # 来自编码器层的语义存储变量memory， 以及源数据掩码张量和目标数据掩码张量

        # 将memory便是成m方便使用
        m = memory
        # x进入第一个子层，第一个子层结构的输入分别是x和self_attn，因为是自注意力机制
        # 最后一个参数是目标数据掩码张量，这是要对目标数据进行遮掩，因此模型可能还没有生成任何数据
        # 比如在解码器准备生成第一个字符或词汇时，我们其实已经上传了第一个字符以方便计算损失
        # 但是我们不希望在生成第一个字符时模型能利用这个信息，因此我们会对其进行遮掩，同时生成第二个字符
        # 模型只能使用第一个字符或者词汇信息，第二个字符以及以后的信息都不允许被模型使用
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))

        # 接着进入第二个子层，这个子层中的常规的注意力机制，q是输入x， k，v是编码层输出memory
        # 同样也传入了source_mask，但是进行源数据合演的原因并非抑制信息泄露，而是遮掩掉对结果没有用的信息
        # 以此提升模型效果和训练速度，这样就完成了第二个子层的处理
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))


        # 最后一个子层就是前馈全连接子层，经过它的处理后就可以返回结果，这就是我，1的解码器的结构
        return self.sublayer[2](x, self.feed_forward)

size = d_model = 512
head = 8
d_ff = 64
dropout = 0.1

self_attn = src_attn = MultiHeadAttention(head, d_model, dropout)

ff = PositionwiseFeedForward(d_model, d_ff, dropout)

x = pe_result
memory = en_result

mask = Variable(torch.zeros(2, 4, 4))
source_mask = target_mask = mask

dl = DecoderLayer(size, self_attn, src_attn, ff, dropout)
dl_result = dl(x, memory, source_mask, target_mask)
# print(dl_result)
# print(dl_result.shape)

# -------------------------------解码器------------------------------
class Decoder(nn.Module):
    def __init__(self, layer, N):
        # 初始化函数有两个参数，第一个是解码器层layer， 第二个是解码器层个数N
        super(Decoder, self).__init__()
        # 首先使用clones方法克隆了N个layer，然后实例化了一个规范化层
        # 因为数据走过了所有的解码器层后最后要做规范化处理
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        # 有四个参数，x代表目标数据的嵌入表示，memory是编码层的输出
        # source_mask，target_mask代表源数据和目标数据的掩码张量

        # 然后对每个层进行循环当然这个循环就是变量x通过每一个层的处理
        # 得出最后结构,进行以此规范化返回即可
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


size = d_model = 512
head = 8
d_ff = 64
dropout = 0.2
c = copy.deepcopy
attn = MultiHeadAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)

N = 8
x = pe_result
memory = en_result
mask = Variable(torch.zeros(2, 4, 4))
source_mask = target_mask = mask

de = Decoder(layer, N)
de_result = de(x, memory, source_mask, target_mask)
# print(de_result)
# print(de_result.shape)

#--------------线性层和softmax层------------------------------------------------------------
# 将线性层和softmax层计算一起实现,因为二者的共同目标是生成最后的结构
# 因此芭蕾的名字叫做Generator,生成器类
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        # 两个参数,d_model代表词的嵌入维度,vocab_size代表词表的大小
        super(Generator, self).__init__()
        # 首先就是使用nn中的预定义线性层进行实例化,得到一个对象self.project等待使用
        # 这个线形成的参数有两个就是初始化进来的两个参数。d_model和vocab_size
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # 在函数中，首先使用上一步得到的self.project对x进行线性变化
        # 然后使用F中已经实现的log_softmax进行softmax处理
        # 在这里之所以使用log_softmax是因为我们这个pytorch攀比的损失函数实现有关
        # log_softmax就是对softmax的结果又取了对数，因为对数函数是单调递增函数
        # 因此对最终我们取最大的概率没有影响，最终返回结果
        return F.log_softmax(self.project(x), dim=-1)

d_model = 512
vocab_size = 1000
x = de_result

gen = Generator(d_model, vocab_size)
gen_result = gen(x)
# print(gen_result)
# print(gen_result.shape)
# ----------------------------编码器-解码器---------------------
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        super(EncoderDecoder, self).__init__()
        # 舒适化五个参数分别的encoder：编码器对象，decoder：解码器对象
        # source_embwd：源数据嵌入函数，target_embed:目标数据嵌入函数，generator：以及输出部分生成器对象

        # 将参数传入到类中
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generatoe = generator

    def forward(self, source, target, source_mask, target_mask):
        # source：代表源数据 target代表目标数据
        # 在函数中，将source，source_mask传入编码函数中，得到结果后与source_mask，target和target_mask一同穿个解码函数
        return self.decode(self.encode(source, source_mask), source_mask, target, target_mask)

    def encode(self, source, source_mask):
        # 使用src_embed对source做处理，然后和source_mask一起传给self.encoder
        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        # memory：代表经历编码后的输出张量
        # 使用tgt_embed对target做处理，然后和source_mask，target，memory一起传给self.decoder
        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)


vocab_size = 1000
d_model = 512
encoder = en
decoder = de
source_embed = nn.Embedding(vocab_size, d_model)
target_embed = nn.Embedding(vocab_size, d_model)
generator = gen

source = target = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
source_mask = target_mask = Variable(torch.zeros(2, 4, 4))

ed = EncoderDecoder(encoder, decoder, source_embed, target_embed, generator)
ed_result = ed(source, target, source_mask, target_mask)
# print(ed_result)
# print(ed_result.shape)

# ------------------------Transformer代码分析---------------------
def make_model(source_vocab, target_vocab, N=6,
               d_model=512, d_ff=2048, head=8, dropout=0.1):
    # 该函数用来构建模型，有七个参数，source_vocab:源数据特征（词汇）总数
    # target_vocab：目标数据特征总数，N:编码器和解码器总数，d_model:词向量映射维度，d_ff：前反馈全连接网络中的变换矩阵维度
    # head：多头注意力结构中的多头数，，dropout：置0比率

    # 首先得到一个深度拷贝命令，接下来很多结构都需要进行深度拷贝
    # 来保证他们彼此之间相互独立，不受干扰
    c = copy.deepcopy

    # 实例化了很多多头注意力，得到对象attn
    attn = MultiHeadAttention(head, d_model)
    # 然后实现实例化前反馈全连接类，得到对象ff
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # 实例化位置编码类。得到对象position
    position = PositionalEncoding(d_model, dropout)


    # 根据结构图，最外层是EncoderDecoder，在EncoderDecoder中，分别是编码器层，解码器层，源数据Embedding层1和位置编码组成的有序结构
    # 目标数据Embedding层和位置编码组成有序结构，以及类生成器层
    # 在编码器层中有attention子层以及前反馈全连接子层
    # 在解码器层有两个attention子层和前反馈全连接层
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab))

    # 模型构建完成后，接下来就是初始化模型中的参数，比如线性层中的变化矩阵
    # 这里一旦判断参数维度大于1，则会将其初始化层一个服从均匀分布的矩阵
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


source_vocab = 11
target_vocab = 11
N = 6

if __name__ == '__main__':
    res = make_model(source_vocab, target_vocab, N)
    print(res)



# class Transformer(nn.Module):
#     def __init__(self, src_vocab, tag_vacab, d_model=512, N=6, h=8, d_ff=2048, dropout=0.1):
#         super().__init__()
#
#         self.src_embed = nn.Sequential(
#             Embeddings(src_vocab, d_model),
#             PositionnalEncoding(d_model)
#         )
#
#         self.tag_embed = nn.Sequential(
#             Embeddings(tag_vacab, d_model),
#             PositionnalEncoding(d_model)
#         )
#
#         attn = lambda: MultiHeadAttention(h, d_model, dropout)
#         ff = lambda : FeedForward(d_model, d_ff, dropout)
#
#         self.encoder = nn.ModuleList([
#             EncoderLayer(d_model, attn(), ff(), dropout) for _ in range(N)
#         ])
#
#         self.out = nn.Linear(d_model, tag_vacab)
#
#         def encode(self, src, src_mask):
#             x = self.src_embed(src)
#             for layer in self.encoder:
#                 x = layer(x, src_mask)
#             return x
#
#         def decode(self, tag, memory, src_mask, tag_mask):
#             x = self.tag_embed
#             for layer in self.decoder:
#                 x = layer(x, memory, src_mask, tag_mask)
#             return x
#         def forward(self,src, tag, src_mask, tag_mask=None):
#             memory = self.encode(src, src_mask)
#             out = self.decode(tag, memory, src_mask, tag_mask)
#
#             return self.out(out)