1.主要依赖包
-tensorboard-gpu/tensorboard
-jieba
-pandas
-pickle
-numpy

2.文件结构
-data 数据存储文件
	train.xlsx 训练用数据
	klb.xlsx 测试用，康莱宝数据
	yhwc.xlsx 测试用，樱花卫厨数据
	yxcd.xlsx 测试用，易鑫车贷数据
	glossary 词汇表，pickle文件
-model 模型存储文件
	checkpoint
	last.ckpt.data-00000-of-00001
	last.ckpt.index
	last.ckpt.meta
-interface.py 测试接口
-train.py 训练接口
-model_sent.py 模型
-reader.py 读文件util


3.测试，见文件interface.py
3.1.
test(filepath)
已知标记文本，测试模型预测的准确性，输出各项指标
Args:
	filepath 输入文件路径。输入文件格式形如data/klb.xlsx
    
3.2.
predict(input_path, output_path)
已知无标记文本，输出本模型预测结果
Args:
	input_path 输入文件路径。输入文件格式形如data/klb.xlsx
	output_path 输出文件路径。
	
3.3.
test_onesent(text)
预测输入的一句话的情感结果
Args:
	text 输入一句文本。

	
4.训练，见文件train.py
4.1
train_sent(sess, corpus, test_corpus)
训练model_sent模型，结果保存在model
Args:
	train_data 训练文件路径
	test_data 测试文件路径
	
4.2训练时tensorboard使用
tensorboard --logdir=tensorb	