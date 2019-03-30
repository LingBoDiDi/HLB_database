import collections
import numpy as np

start_token = 'B'
end_token = 'E'


def process_poems(file_name):
    # poems -> list of numbers
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():         #依次按行读取训练数据
            try:
                title, content = line.strip().split(':')    #将诗词的标题和内容分开
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                        start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 79:    #去掉不适合当作诗词集的训练集
                    continue
                content = start_token + content + end_token  #提出每行训练集中的数据字符串
                poems.append(content)  #在新定义的数组中存储每行提取的诗词
            except ValueError as e:
                pass
    # poems = sorted(poems, key=len)

    all_words = [word for poem in poems for word in poem]
    counter = collections.Counter(all_words)
    words = sorted(counter.keys(), key=lambda x: counter[x], reverse=True)  #倒序排序:按照键值由大到小

    words.append(' ')  #在列表末尾添加新的对象
    L = len(words)
    word_int_map = dict(zip(words, range(L)))  #创造词典
    poems_vector = [list(map(lambda word: word_int_map.get(word, L), poem)) for poem in poems]  #创造训练集诗词的向量

    return poems_vector, word_int_map, words


def generate_batch(batch_size, poems_vector, word_to_int):
    n_chunk = len(poems_vector) // batch_size  #分批次处理训练集向量
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size

        batches = poems_vector[start_index:end_index]
        length = max(map(len, batches))  #防止存储空间不足
        x_data = np.full((batch_size, length), word_to_int[' '], np.int32)  #构造一个64*length的数组
        for row, batch in enumerate(batches):
            x_data[row, :len(batch)] = batch  #分批处理训练集向量（矩阵切割）
        y_data = np.copy(x_data)
        y_data[:, :-1] = x_data[:, 1:]  #训练集标签
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches
