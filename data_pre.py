#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


path = './weibo_senti_100k.csv'


# In[3]:


pd_all = pd.read_csv('./data/weibo_senti_100k.csv')

print('评论数目（总体）：%d' % pd_all.shape[0])
print('评论数目（正向）：%d' % pd_all[pd_all.label==1].shape[0])
print('评论数目（负向）：%d' % pd_all[pd_all.label==0].shape[0])


# In[4]:


pd_all.sample(20)


# In[5]:


from sklearn.model_selection import train_test_split


# 假设您的数据集有两列：'text' 和 'label'
X = pd_all['review']  # 特征列
y = pd_all['label']  # 标签列

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=220)

# 查看划分结果
print(f"训练集大小: {len(X_train)}")
print(f"测试集大小: {len(X_test)}")


# In[6]:


# 将训练集和测试集保存为新的 CSV 文件
train_data = pd.DataFrame({'review': X_train, 'label': y_train})
test_data = pd.DataFrame({'review': X_test, 'label': y_test})

train_data.to_csv('./data/train_dataset.csv', index=False)
test_data.to_csv('./data/test_dataset.csv', index=False)

print("训练集和测试集已保存到 CSV 文件中。")


# In[7]:


import jieba


# In[8]:


# 数据读取
def load_tsv(file_path):
   data = pd.read_csv(file_path)
   data_x = data.iloc[:, 0]
   # 选择列名为 'col1' 和 'col3' 的列
   #df_slice = df.loc[:, ['col1', 'col3']]
   data_y = data.iloc[:, -1]
   return data_x, data_y

with open('./hit_stopwords.txt','r',encoding='UTF8') as f:
   stop_words=[word.strip() for word in f.readlines()]
   print('Successfully')
def drop_stopword(datas):
   for data in datas:
       for word in data:
           if word in stop_words:
               data.remove(word)
   return datas

def save_data(datax,path):
   with open(path, 'w', encoding="UTF8") as f:
       for lines in datax:
           for i, line in enumerate(lines):
               f.write(str(line))
               # 如果不是最后一行，就添加一个逗号
               if i != len(lines) - 1:
                   f.write(',')
           f.write('\n')


# In[9]:


if __name__ == '__main__':
    train_x, train_y = load_tsv("./data/train_dataset.csv")
    test_x, test_y = load_tsv("./data/test_dataset.csv")
    train_x = [list(jieba.cut(x)) for x in train_x]
    test_x = [list(jieba.cut(x)) for x in test_x]
    train_x=drop_stopword(train_x)
    test_x=drop_stopword(test_x)
    save_data(train_x,'./train.txt')
    save_data(test_x,'./test.txt')
    print('Successfully')

