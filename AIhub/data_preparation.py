from os import listdir
from os.path import isfile, join
import os
import soundfile


def wav_duration(file_name):
    audio, samp_rate = soundfile.read(file_name, dtype='int16')
    nframes = audio.shape[0]
    duration = nframes / samp_rate
    return duration

def meta_to_txt():
    location = "./AIhub"
    reader = open("./AIhub/ASR_metadata.txt", "r", encoding='utf-8').readlines()
    audio_paths = []
    targets = []
    total_len = 0
    train_set_len = 0
    for i, line in enumerate(reader):
        audio_path, target, _ = line.strip().split('|')
        audio_path = audio_path.replace('\\', '/')[2:]
        wavduration = wav_duration(os.path.join(location, audio_path))
        total_len = total_len + wavduration
        if wavduration > 6.68:
            continue
        audio_paths.append(str(i)+'|'+audio_path+'\n')
        targets.append(str(i) + '|' + target + '\n')
        train_set_len = train_set_len + wavduration
    with open(os.path.join(location, 'feats.txt'),'w', encoding='utf-8') as file:
        file.writelines(audio_paths)
    with open(os.path.join(location, 'targets.txt'),'w', encoding='utf-8') as file:
        file.writelines(targets)
    print(total_len, train_set_len)
    return

def split_valset():
    val=[]
    test=[]
    mode = 'targets.txt'
    with open(os.path.join('./AIhub', mode),'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            val.append(line)
            if i == 3610:
                break

        for i, line in enumerate(file):
            test.append(line)
            if i == 3610:
                break

        with open(os.path.join('./data/AIhub/train', mode),'w', encoding='utf-8') as output:
            for line in file:
                output.write(line)

    with open(os.path.join('./data/AIhub/dev', mode),'w', encoding='utf-8') as file:
        file.writelines(val)

    with open(os.path.join('./data/AIhub/test', mode),'w', encoding='utf-8') as file:
        file.writelines(test)

    return


#meta_to_txt()

if __name__ == '__main__':
    split_valset()
