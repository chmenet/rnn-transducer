# import numpy as np
# import os
# import random
# location = "C:/Users/chme/Desktop/intern_test/mel_spect/game_mel"
# metas=[]
#
# #for i, file_name in enumerate(os.listdir(location)):
# total_data_number = 100
# start_token = 's'
# stay_token = '0'
# tokens = ['1', '2', '3', '4', '5']
# elements = ['a', 'bb', 'c', 'dd', 'e', 'ff']
# combination = ['abb', 'bbc', 'cdd', 'dde', 'eff']
# all='abbcddeffghhijjk'
# for i in range(total_data_number):
#     input = ''
#     target = ''
#     line = input + ','+ len(input)+','+target+ ',' + len(target) + ',' + "\n"
#     metas.append(line)
#
# with open('C:/Users/chme/Desktop/Voice_AI/kof_mtm.txt','w', encoding='utf-8') as file:
#     file.writelines(metas)
#
# from __future__ import print_function
# import argparse
# import os
# import shutil
# import random
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--dir', help="data directory", default="../data")
# parser.add_argument('--max-len', help="max sequence length", default=10)
# args = parser.parse_args()
#
# def generate_dataset(root, name, size):
#     path = os.path.join(root, name)
#     if not os.path.exists(path):
#         os.mkdir(path)
#
#     # generate data file
#     data_path = os.path.join(path, 'data.txt')
#     with open(data_path, 'w') as fout:
#         for _ in range(size):
#             length = random.randint(1, args.max_len)
#             seq = []
#             for _ in range(length):
#                 seq.append(str(random.randint(0, 9)))
#             fout.write("\t".join([" ".join(seq), " ".join(reversed(seq))]))
#             fout.write('\n')
#
#     # generate vocabulary
#     src_vocab = os.path.join(path, 'vocab.source')
#     with open(src_vocab, 'w') as fout:
#         fout.write("\n".join([str(i) for i in range(10)]))
#     tgt_vocab = os.path.join(path, 'vocab.target')
#     shutil.copy(src_vocab, tgt_vocab)

def toy_gen(chars, times = 2):
    len_chars = len(chars)

    X = []
    Y = []
    for i in range(len_chars*times): # length index
        if(i == 0):
            continue
        for j in range(len_chars): # first char index
            x = ''
            y = ''
            for k in range(i): # length index
                x += chars[(j+k)%len_chars] if (j+k) % 2 == 0 else chars[(j+k)%len_chars]*2
                y += chars[(j+k+1)%len_chars]
            X.append(x)
            Y.append(y)

    return X, Y

if __name__ == '__main__':
    chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    X,Y = toy_gen(chars)
    with open('data/toy/train/feats.txt','w',encoding='utf-8') as f:
        f.writelines(['{},{}\n'.format(i,x) for i, x in enumerate(X[:12]*10)])

    with open('data/toy/train/target.txt','w',encoding='utf-8') as f:
        f.writelines(['{},{}\n'.format(i,y) for i, y in enumerate(Y[:12]*10)])

    with open('data/toy/dev/feats.txt', 'w', encoding='utf-8') as f:
        f.writelines(['{},{}\n'.format(i, x) for i, x in enumerate(X[12:])])

    with open('data/toy/dev/target.txt', 'w', encoding='utf-8') as f:
        f.writelines(['{},{}\n'.format(i, y) for i, y in enumerate(Y[12:])])

    with open('data/toy/test/feats.txt', 'w', encoding='utf-8') as f:
        f.writelines(['{},{}\n'.format(i, x) for i, x in enumerate(X[12:])])

    with open('data/toy/test/target.txt', 'w', encoding='utf-8') as f:
        f.writelines(['{},{}\n'.format(i, y) for i, y in enumerate(Y[12:])])

    chars = list('abcdefghijklnmopqrstuvwxyz')
    X, Y = toy_gen(chars, times=1)
    with open('data/toy2/train/feats.txt', 'w', encoding='utf-8') as f:
        f.writelines(['{},{}\n'.format(i, x) for i, x in enumerate(X[:25])])

    with open('data/toy2/train/target.txt', 'w', encoding='utf-8') as f:
        f.writelines(['{},{}\n'.format(i, y) for i, y in enumerate(Y[:25])])

    with open('data/toy2/dev/feats.txt', 'w', encoding='utf-8') as f:
        f.writelines(['{},{}\n'.format(i, x) for i, x in enumerate(X[25:])])

    with open('data/toy2/dev/target.txt', 'w', encoding='utf-8') as f:
        f.writelines(['{},{}\n'.format(i, y) for i, y in enumerate(Y[25:])])

    with open('data/toy2/test/feats.txt', 'w', encoding='utf-8') as f:
        f.writelines(['{},{}\n'.format(i, x) for i, x in enumerate(X[25:])])

    with open('data/toy2/test/target.txt', 'w', encoding='utf-8') as f:
        f.writelines(['{},{}\n'.format(i, y) for i, y in enumerate(Y[25:])])


    # data_dir = args.dir
    # if not os.path.exists(data_dir):
    #     os.mkdir(data_dir)
    #
    # toy_dir = os.path.join(data_dir, 'toy_reverse')
    # if not os.path.exists(toy_dir):
    #     os.mkdir(toy_dir)
    #
    # generate_dataset(toy_dir, 'train', 10000)
    # generate_dataset(toy_dir, 'dev', 1000)
    # generate_dataset(toy_dir, 'test', 1000)