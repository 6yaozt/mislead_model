import tensorflow as tf
import numpy as np
import os
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import flask
import jieba
import sys
import time

sys.path.append("../../")
import codecs
import os
import re
import sys
import importlib

importlib.reload(sys)

def FenCi(line):
    jieba.initialize()
    # 更高效的字符串替换
    lines = filter(lambda ch: ch not in '0123456789 ', line)
    line2 = "".join(lines)
    newline = jieba.lcut(line2, cut_all=False)
    newline = ' '.join(list(newline)).replace('，', ' ').replace('。', ' ').replace('？', ' ').replace('！', ' ') \
            .replace('（', ' ').replace('）', ' ') \
            .replace('=', ' ').replace('-', ' ') \
            .replace('+', ' ').replace(';', ' ') \
            .replace(')', ' ').replace(')', ' ') \
            .replace('◣', ' ').replace('◢', ' ') \
            .replace('@', ' ').replace('|', ' ') \
            .replace('~', ' ').replace(']', ' ') \
            .replace('●', ' ').replace('★', ' ') \
            .replace('/', ' ').replace('■', ' ') \
            .replace('╪', ' ').replace('☆', ' ') \
            .replace('└', ' ').replace('┘', ' ') \
            .replace('─', ' ').replace('┬', ' ') \
            .replace('：', ' ').replace('‘', ' ') \
            .replace(':', ' ').replace('-', ' ') \
            .replace('、', ' ').replace('.', ' ') \
            .replace('...', ' ').replace('?', ' ') \
            .replace('“', ' ').replace('”', ' ') \
            .replace('《', ' ').replace('》', ' ') \
            .replace('!', ' ').replace(',', ' ') \
            .replace('】', ' ').replace('【', ' ') \
            .replace('·', ' ')
    return newline
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
FLAGS = tf.flags.FLAGS
app = flask.Flask(__name__)

# Load the saved model and vocabulary processor
checkpoint_dir = "./runs/1702306241/checkpoints/"
vocab_path = os.path.join(checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

# Load the saved model
session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
sess = tf.Session(config=session_conf)
saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(checkpoint_dir) + '.meta')
saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))


graph = tf.get_default_graph()
input_x = graph.get_operation_by_name("input_x").outputs[0]
dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
predictions = graph.get_operation_by_name("output/predictions").outputs[0]

# Define the API endpoint
@app.route('/check_misleading', methods=['POST'])
def check_misleading():
    data = flask.request.get_json(force=True)
    x1 = data['text']
    print(x1)
    x1=FenCi(x1)
    x_raw = [x1]
    x_test = np.array(list(vocab_processor.transform(x_raw)))

    # Generate batches for one epoch
    batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

    # 存储模型预测结果
    all_predictions = []
    for x_test_batch in batches:
        batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
        all_predictions = np.concatenate([all_predictions, batch_predictions])
    y = []
    for i in all_predictions:
        if i == 0:
            y.append("[诱导]")
        elif i==1:
            y.append("[非诱导]")
        else:
            y.appeng("无")
    # 把预测的结果保存到本地
    predictions_human_readable = np.column_stack((y, np.array(x_raw)))
    out_path = os.path.join(checkpoint_dir, "..", "prediction.csv")
    print("Saving evaluation to {0}".format(out_path))
    with open(out_path, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(predictions_human_readable)

    #但凡分词里面含有一个0就是诱导性文字
    if 0 in all_predictions:result=True
    else: result=False
    return flask.jsonify({'is_misleading': result})

if __name__ == '__main__':
    app.run(port=5000)
