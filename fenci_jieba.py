# coding:utf-8
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


def FenCi(readfile, outfile):
    line = readfile.readline()
    jieba.initialize()
    while line:
        # 更高效的字符串替换
        lines = filter(lambda ch: ch not in '0123456789 ', line)
        line2 = "".join(lines)
        newline = jieba.lcut(line2, cut_all=False)
        str_out = ' '.join(list(newline)).replace('，', ' ').replace('。', ' ').replace('？', ' ').replace('！', ' ') \
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
        # print (str_out)
        outfile.write(str_out)
        line = readfile.readline()


if __name__ == '__main__':
    fromdir = "./data"
    todir = "./fenci/"
    # 一次只能对一个文档进行分词
    file = "clear.txt"
    #file = "mislead.txt"
    outfile = open(os.path.join(todir, file), 'w+', encoding='utf-8')
    infile = open(os.path.join(fromdir, file), 'r', encoding='utf-8')
    FenCi(infile, outfile)
    infile.close()
    outfile.close()

