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


# 画像の四方を切り取る関数
def trim_image(img, x_start, x_end, y_start, y_end):
    """
    画像の指定された範囲をトリミングする

    Parameters:
    - img (np.ndarray): トリミングする画像
    - x_start, x_end, y_start, y_end (int): トリミングする範囲を指定する

    Returns:
    - np.ndarray: トリミングされた画像
    """
    if not isinstance(img, np.ndarray):
        raise ValueError("img must be a numpy ndarray")
    if not (0 <= x_start < x_end <= img.shape[1] and 0 <= y_start < y_end <= img.shape[0]):
        raise ValueError("Invalid trim range")

    return img[y_start:y_end, x_start:x_end]


# 1つ前のシフト量に基づいてトリミングする関数
def trim_image_based_on_shift(img: np.ndarray, shift: tuple, addition, img_type: int) -> np.ndarray:
    """
    画像の上下左右からシフト量に基づいてトリミングする

    Parameters:
    - img (np.ndarray): トリミングする画像
    - shift (tuple): トリミングするピクセル数 (y, x)
    - addition: 追加でトリミングするピクセル数
    - img_type (int): 画像のタイプ (1 または 2)
                    Type1 : Shift>0の時、画像先頭から切り取り、Shift<0の時、画像終端から切り取る(参照用)
                    Type2 : Shift>0の時、画像終端から切り取り、Shift<0の時、画像先頭から切り取る(補正対象用)

    Returns:
    - np.ndarray: トリミングされた画像
    """
    if img_type not in [1, 2]:
        raise ValueError("Invalid image type. Must be 1 or 2.")
    height, width = img.shape
    shift_y, shift_x = shift

    y_positive, y_negative = shift_y + addition, shift_y - addition
    x_positive, x_negative = shift_x + addition, shift_x - addition

    if img_type == 1:
        start_y = y_positive if shift_y > 0 else 0
        end_y = height + y_negative if shift_y < 0 else height
        start_x = x_positive if shift_x > 0 else 0
        end_x = width + x_negative if shift_x < 0 else width
    elif img_type == 2:
        start_y = -y_negative if shift_y < 0 else 0
        end_y = height - y_positive if shift_y > 0 else height
        start_x = -x_negative if shift_x < 0 else 0
        end_x = width - x_positive if shift_x > 0 else width
    else:
        raise ValueError("Invalid image type. Must be 1 or 2.")

    trimmed_img = img[start_y:end_y, start_x:end_x]

    return trimmed_img


# ヒートマップを作成し、保存する関数
def heatmap_plt(matrix, save_filename):
    """
    ヒートマップを作成し、保存する関数

    Parameters:
    - matrix (np.ndarray): 相互相関関数
    - save_filename (str): 保存先パス
    """
    if not isinstance(matrix, np.ndarray) or not isinstance(save_filename, str):
        raise ValueError("Invalid input types. 'matrix' should be a numpy ndarray and 'save_filename' should be a string.")
    # 新たなフィギュアを生成
    plt.figure(figsize=(12, 10))

    # 軸の範囲を計算
    x_max = matrix.shape[1] // 2
    y_max = matrix.shape[0] // 2

    # ヒートマップを描画
    plt.imshow(np.abs(matrix), cmap='gray', vmin=0.40, vmax=0.80,
               extent=[-x_max, x_max, -y_max, y_max], aspect='auto')

    # 軸ラベルを設定
    plt.xlabel('X')
    plt.ylabel('Y')

    # カラーバーを追加
    plt.colorbar(label='correlation', shrink=0.6)

    # 上部に余白を設ける
    plt.subplots_adjust(top=0.7)

    # 保存
    plt.savefig(save_filename, bbox_inches='tight', pad_inches=0)

    # フィギュアをクリア
    plt.close()


# 指定座標における水平・垂直ラインプロファイルを作成し、保存する関数
def profiles_plt(matrix, position, save_filename):
    """
    指定座標における水平・垂直ラインプロファイルを作成し、保存する関数

    Parameters:
    - matrix (np.ndarray): 相互相関関数
    - position (tuple): (x, y)
    - save_filename (str): 保存先パス
    """
    # 水平ラインプロファイル
    horizontal_profile = matrix[position[0], :]
    # 垂直ラインプロファイル
    vertical_profile = matrix[:, position[1]]

    # 軸の範囲を計算
    x_max = len(horizontal_profile) // 2
    y_max = len(vertical_profile) // 2

    plt.figure(figsize=(10, 14))  # グラフのサイズを指定
    plt.plot(range(-x_max, x_max), horizontal_profile,
             color='blue', label='Horizontal Profile')
    plt.plot(range(-y_max, y_max), vertical_profile,
             color='red', label='Vertical Profile')

    plt.title('Line Profiles')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()  # 凡例を表示
    plt.grid(True)
    plt.savefig(save_filename)

    # フィギュアをクリア
    plt.close()


# ガウシアンを定義
def gaussian1D(x, amplitude, xo, sigma, offset):
    xo = float(xo)
    g = offset + amplitude * np.exp(-((x-xo)**2) / (2 * sigma**2))
    return g


# プロファイルにガウシアンをフィットし、FWHMを計算
def fit_gaussian_and_compute_fwhm(profile):
    # フィットのための初期パラメータを定義
    x = np.arange(profile.size)
    amplitude = profile.max()
    xo = profile.argmax()
    sigma = 25
    offset = 0

    try:
        # フィットを実行
        popt, _ = curve_fit(gaussian1D, x, profile, p0=[amplitude, xo, sigma, offset])
    except RuntimeError:
        print("Error - curve_fit failed")
        popt = [0, 0, 0, 0]

    # フィット結果のsigmaからFWHMを計算
    fwhm = 2 * np.sqrt(2 * np.log(2)) * popt[2]

    return popt, fwhm


# ラインプロファイルとFWHMを計算
def compute_line_profiles_and_fwhms(matrix, peak, save_filename):
    """
    ラインプロファイルとFWHMを計算

    Parameters:
    - matrix (np.ndarray): 相互相関関数
    - peak (tuple): ピークの位置 (x, y)
    - save_filename (str): 保存先パス
    """
    # 水平および垂直のラインプロファイルを抽出
    horizontal_profile = matrix[peak[0], :]
    vertical_profile = matrix[:, peak[1]]

    # 軸の範囲を計算
    # 画像の中心を計算しています。画像の幅や高さを2で割った結果を取得
    center_x = len(horizontal_profile) // 2
    center_y = len(vertical_profile) // 2

    # 画像の中心を基準にxおよびyの範囲を計算
    # これにより、プロットの中心が画像の中心と一致
    x_horizontal = np.arange(-center_x, len(horizontal_profile) - center_x)
    x_vertical = np.arange(-center_y, len(vertical_profile) - center_y)

    # ガウシアンフィットとFWHMの計算
    # 水平および垂直のラインプロファイルをガウシアン関数でフィッティング
    # また、フィットの結果からFWHM (Full Width at Half Maximum) を計算
    popt_horizontal, fwhm_horizontal = fit_gaussian_and_compute_fwhm(horizontal_profile)
    popt_vertical, fwhm_vertical = fit_gaussian_and_compute_fwhm(vertical_profile)

    # プロファイルとフィット結果のプロット
    # まず、水平方向のプロファイルとそのガウシアンフィットをプロット
    plt.plot(x_horizontal, horizontal_profile,
             color='blue', label='Horizontal Profile')
    plt.plot(x_horizontal, gaussian1D(np.arange(horizontal_profile.size), *
             popt_horizontal), color='blue', linestyle='--', label='Fit to Horizontal Profile')

    # 次に、垂直方向のプロファイルとそのガウシアンフィットをプロット
    plt.plot(x_vertical, vertical_profile,
             color='red', label='Vertical Profile')
    plt.plot(x_vertical, gaussian1D(np.arange(vertical_profile.size), *popt_vertical),
             color='red', linestyle='--', label='Fit to Vertical Profile')

    # グラフのタイトル、軸のラベル、凡例を設定
    plt.title('Line Profiles and Gaussian Fits')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_filename)
    plt.close()

    return fwhm_horizontal, fwhm_vertical


# フーリエ変換し、2つの画像間の相互相関を計算する関数
def compute_cross_correlation(img1, img2):
    """
    Parameters:
    - img1, img2 (np.ndarray): 画像

        f_img1 * np.conj(f_img2)
        A * B: 行列の要素同士の積
        周波数領域の配列の複素共役(np.conj())をとること = 実空間領域の配列の逆順をとること

    Returns:
    - np.ndarray: 相互相関関数
    """
    # フーリエ変換
    f_img1 = fft2(img1)
    f_img2 = fft2(img2)

    # 画像のフーリエ変換を共役複素数で掛け、逆フーリエ変換
    cross_correlation = ifft2(f_img1 * np.conj(f_img2))

    # 画像中心にシフトさせる
    # 画像中心が左上(配列の[0, 0]位置)に移動してしまうため
    cross_correlation = fftshift(cross_correlation)

    return np.abs(cross_correlation)


# 相互相関関数を規格化する関数
def normalized_cross_correlation(img1, img2):
    # 相互相関関数を計算
    cross_correlation = compute_cross_correlation(img1, img2)

    # 自己相関（自分自身との相互相関）を計算
    autocorr1 = compute_cross_correlation(img1, img1)
    autocorr2 = compute_cross_correlation(img2, img2)

    # 自己相関の最大値で規格化
    cross_correlation /= np.sqrt(np.max(autocorr1) * np.max(autocorr2))

    return cross_correlation


# 相互相関関数の最大値を見つけることによって画像間のシフト量を計算する関数
def compute_shift(img1, img2, max_correlation, fwhm_results, heatmap_dir, profiles_dir, i):
    """
    Parameters:
    - img1, img2 (np.ndarray): 画像
    - max_correlation (list): 相互相関関数の最大値をまとめるリスト
    - fwhm_results (list): 相互相関関数のピークの半値全幅をまとめるリスト

    Returns:
    - np.ndarray: ドリフト量
    """
    # 規格化された相互相関
    correlation = normalized_cross_correlation(img1, img2)

    # 補正精度評価のためにヒートマップ、ラインプロファイル、FWHMを求める
    # ヒートマップを作成
    heatmap_plt(correlation, os.path.join(heatmap_dir, f'heatmap_{i}.png'))

    # 相互相関関数のピークを見つける
    peak = np.unravel_index(correlation.argmax(), correlation.shape)

    # 相互相関関数の最大値をリストに追加
    max_correlation.append(np.max(correlation))

    # FWHMを計算、ラインプロファイルを作成
    fwhm_x, fwhm_y = compute_line_profiles_and_fwhms(
        correlation, peak, os.path.join(profiles_dir, f'profiles_{i}.png'))

    # FWHMの結果をリストに追加
    fwhm_results.append((fwhm_x, fwhm_y))

    # ピークと画像中心との差を計算する
    # 画像中心からピーク座標へのベクトルを補正に用いる
    shift_values = np.array(peak) - np.array(correlation.shape) // 2

    return shift_values


# シフト量をグラフにまとめる関数
def shift_values_plt(shift_list, save_filename):
    Vertical_Shift_Values, Horizontal_Shift_Values = zip(*shift_list)

    # Vertical_Shift_Valuesの正負を逆にする
    Vertical_Shift_Values = np.array(Vertical_Shift_Values) * -1

    # リスト番号（横軸の値）
    list_numbers = range(len(Vertical_Shift_Values))

    # グラフの描画
    plt.figure(figsize=(10, 6))

    plt.plot(list_numbers, Vertical_Shift_Values,
             label='Vertical_Shift_Values', marker='o')
    plt.plot(list_numbers, Horizontal_Shift_Values,
             label='Horizontal_Shift_Values', marker='x')

    plt.xlabel('Picture Number')
    plt.ylabel('Shift_Value')
    plt.title('Vertical and Horizontal Shift Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_filename)

    # フィギュアをクリア
    plt.close()