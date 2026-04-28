import jieba


##jieba分词示例
text = "我来到北京清华大学"

#全模式，返回一个生成器
jieba.lcut(text)
for word in jieba.lcut(text):
    print(word)
print("--------")
#精确模式，返回一个list
jieba.lcut(text, cut_all=True)
print(jieba.lcut(text, cut_all=True))
print("--------")
#搜索引擎模式
jieba.lcut_for_search(text)
for word in jieba.lcut_for_search(text):
    print(word)
print("--------")

##自定义词典
jieba.load_userdict("./data/user_dict.txt")

jieba.lcut("随着云计算技术的普及，越来越多企业开始采用云原生架构来部署服务，并借助大模型能力提升智能化水平，实现业务流程的自动化与智能决策。")
for word in jieba.lcut("随着云计算技术的普及，越来越多企业开始采用云原生架构来部署服务，并借助大模型能力提升智能化水平，实现业务流程的自动化与智能决策。"):
    print(word)