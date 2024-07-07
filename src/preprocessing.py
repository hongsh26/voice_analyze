import os
import random
import shutil
import csv
import librosa
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_features(file_path, sr=16000):
    y, sr = librosa.load(file_path, sr=sr)  # 샘플링 레이트 16000으로 로드
    features = {
        'id': file_path,
        'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y)),
        'rms': np.mean(librosa.feature.rms(y=y)),
        'tempo': librosa.beat.tempo(y=y, sr=sr)[0],
        'amplitude_skew': np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)[0]),
        'amplitude_kurtosis': np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)[1])
    }
    return features


def load_and_label_data(dir, csv):
    try:
        train = pd.read_csv(csv)
    except Exception as e:
        print(f"Error reading CSV file: {e}")

    print("data loading --------")
    count = 0
    length = len(os.listdir(dir))
    for filename in os.listdir(dir):
        count += 1
        print(f"{count}/{length}")
        try:
            result = train[train['id'] == filename.replace('.ogg', '')]
            label = result['label'].values[0]
            print(label)
            if (label == 'real'):
                print('../data/train/' + filename)
                shutil.move('../data/train/' + filename, '../data/real_train/' + filename)
            else:
                print('../data/train/' + filename)
                shutil.move('../data/train/' + filename, '../data/fake_train/' + filename)
        except Exception as e:
            print(f"Skipping file {filename} due to error: {e}")


def division(real_path, fake_path):
    real_files = [os.path.join(real_path, f) for f in os.listdir(real_path) if os.path.isfile(os.path.join(real_path, f))]
    fake_files = [os.path.join(fake_path, f) for f in os.listdir(fake_path) if os.path.isfile(os.path.join(fake_path, f))]
    random.shuffle(real_files)
    random.shuffle(fake_files)
    train_real, valid_real = train_test_split(real_files, test_size=0.2, random_state=40)
    train_fake, valid_fake = train_test_split(fake_files, test_size=0.2, random_state=40)

    train_dir = '../data/train'
    valid_dir = '../data/valid'
    os.makedirs(os.path.join(train_dir, 'real'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'fake'), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, 'real'), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, 'fake'), exist_ok=True)

    # 파일 복사 함수
    def copy_files(file_list, dest_dir):
        for file in file_list:
            shutil.copy(file, dest_dir)

    # 파일 복사
    copy_files(train_real, os.path.join(train_dir, 'real'))
    copy_files(train_fake, os.path.join(train_dir, 'fake'))
    copy_files(valid_real, os.path.join(valid_dir, 'real'))
    copy_files(valid_fake, os.path.join(valid_dir, 'fake'))

def writing_csv(folder):
    f = open('../data/test_feature.csv', 'w')
    writer = csv.writer(f)
    files = os.listdir(folder)
    length = len(files)
    count = 0
    for file in files:
        content = []
        count += 1
        print(f"{count}/{length}")
        data = extract_features(folder+'/'+ file)
        content.append(file)
        content.append(data['spectral_centroid'])
        content.append(data['spectral_bandwidth'])
        content.append(data['spectral_rolloff'])
        content.append(data['zero_crossing_rate'])
        content.append(data['rms'])
        content.append(data['tempo'])
        content.append(data['amplitude_skew'])
        content.append(data['amplitude_kurtosis'])
        writer.writerow(content)
    f.close()

def combine_train(real_path, fake_path):
    fake_train = pd.read_csv(fake_path)
    real_train = pd.read_csv(real_path)
    real_train['label'] = 1
    fake_train['label'] = 0

    combined_data = pd.concat([real_train, fake_train], axis=0).reset_index(drop=True)
    return combined_data
# load_and_label_data('../data/train', '../data/train.csv')
# division('../data/real_train', '../data/fake_train')
# writing_csv('../data/valid/fake')
# writing_csv('../data/train/real')
# writing_csv('../data/test')

