import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np
import pandas as pd
from scipy.ndimage import shift
from scipy.fft import fft2, ifft2, fftshift
from scipy.optimize import curve_fit
import os
from datetime import datetime
import argparse
import configparser
import image_processing


"""
大まかなプログラム説明

補正対象フォルダを指定し、その連続画像をドリフト補正する。
またその画像を動画として出力する。
さらに評価のため、相互相関関数のヒートマップとラインプロファイルを作成する。
ただしそのラインプロファイルは画像間のドリフト量を表しており、ガウシアンでフィッティングして半値全幅も出力する。

また相互相関関数を求める際、参照画像は直前のシフト量分だけ切り取って黒い帯を消しており、
補正対象画像は参照画像とは逆側を切り取った。
これはドリフト方向があまり変化しないものに対しては有効であると考えている。
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.ini', help='Path to configuration file')
    return parser.parse_args()


def main(config_path):
    config = configparser.ConfigParser()
    if not config.read(config_path, encoding='utf-8'):
        print(f"エラー: 設定ファイル {config_path} が読み込めませんでした。")
        return
    target = config.get('Paths', 'Target')
    base_path = config.get('Paths', 'BasePath')
    image_dir = os.path.join(base_path, target)
    corrected_img_dir = os.path.join(base_path, 'corrected')
    heatmap_dir = os.path.join(base_path, 'heatmap')
    profiles_dir = os.path.join(base_path, 'line profiles')
    csv_dir = os.path.join(base_path, 'csv')
    fps = config.getint('Settings', 'FPS')

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"The directory {image_dir} does not exist.")
    if fps <= 0:
        raise ValueError("FPS must be a positive integer.")


    # 初期シフト値を0に
    shift_values = np.array([0, 0])
    total_shift_values = np.array([0, 0])

    # 相互相関関数の最大値を保存するリストを作成
    max_correlation = []

    # FWHMの結果を保存するリストを作成
    fwhm_results = []

    # シフト量を保存するリストを作成
    shift_values_list = [(0, 0)]

    # 最大値とFWHMの結果を保存するDataFrameを作成
    df_results = pd.DataFrame(columns=[
                              'ref_filename', 'filename', 'max_correlation', 'fwhm_x', 'fwhm_y', 'total_shift_x', 'total_shift_y'])

    # 画像ファイル名のリストを作成
    image_filenames = sorted(os.listdir(image_dir))

    # 最初の画像を参照画像として設定、保存
    ref_img = plt.imread(os.path.join(image_dir, image_filenames[0]))
    # ref_img = rgb2gray(ref_img)
    plt.imsave(os.path.join(corrected_img_dir,
               f'00.png'), ref_img, cmap='gray')

    # 各画像に対してループ
    for i in range(1, len(image_filenames)):
        # 画像を読み込む
        img = plt.imread(os.path.join(image_dir, image_filenames[i]))
        # img = rgb2gray(img)

        """
        # 画像をトリミング
        trm_ref_img = trim_image(ref_img, x_start, x_end, y_start, y_end)
        trm_img = trim_image(img, x_start, x_end, y_start, y_end)
        """

        #1つ前のshift_valuesを保存
        previous_shift_values = total_shift_values

        # 画像をトリミング
        trimmed_ref_img = image_processing.trim_image_based_on_shift(
            ref_img, total_shift_values, 0, 1)
        trimmed_img = image_processing.trim_image_based_on_shift(img, total_shift_values, 0, 2)

        # トリミングした画像を元にシフト量を決定し、元サイズの画像に対してドリフト補正を行う
        shift_values = image_processing.compute_shift(
            trimmed_ref_img, trimmed_img, max_correlation, fwhm_results, heatmap_dir, profiles_dir, i)
        shift_values_list.append(shift_values)
        total_shift_values = np.add(previous_shift_values, shift_values)
        corrected_img = shift(img, total_shift_values)

        # 補正後の画像を保存
        # 画像保存先ディレクトリ、ファイル名
        plt.imsave(os.path.join(corrected_img_dir,
                   f'{i:02}.png'), corrected_img, cmap='gray')

        # 結果をDataFrameに保存
        new_data = pd.DataFrame({'ref_filename': [image_filenames[i-1]], 'filename': [image_filenames[i]], 'max_correlation': [max_correlation[i-1]], 'fwhm_x': [fwhm_results[i-1]
            [0]], 'fwhm_y': [fwhm_results[i-1][1]], 'total_shift_x': [total_shift_values[1]], 'total_shift_y': [total_shift_values[0]]})
        df_results = pd.concat([df_results, new_data], ignore_index=True)


        # 補正後の画像を次の参照画像として設定
        ref_img = corrected_img

    # シフト量をプロット
    image_processing.shift_values_plt(shift_values_list, f'{base_path}\\shift_values')

    # 動画にして出力
    # 移動したいディレクトリのパス
    dir_path = corrected_img_dir

    # カレントディレクトリを変更
    os.chdir(dir_path)

    # 画像ファイルのリストを作成
    image_files = sorted(os.listdir('./'))

    # 動画ファイルの設定
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 動画コーデックを指定

    # 最初の画像から動画の解像度を決定
    img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    video = cv2.VideoWriter('video.mp4', fourcc, fps,
                            (width, height))  # 動画ファイルを開く

    # 各画像を動画に追加
    for image_file in image_files:
        img = cv2.imread(image_file)
        video.write(img)  # 画像を動画に書き込む

    # 動画ファイルを閉じる
    video.release()

    # DataFrameをCSVファイルとして保存
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    df_results.to_csv(os.path.join(
        csv_dir, f'{timestamp}_results.csv'), index=False)


if __name__ == "__main__":
    args = get_args()
    main(args.config)
