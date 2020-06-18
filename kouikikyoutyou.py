# -*- coding: utf-8 -*-
import numpy as np
import cv2
import scipy.ndimage.filters as fi
import matplotlib.pyplot as plt
import scipy.stats


def highpass_filter(src):
    # 高速フーリエ変換(2次元)
    src = np.fft.fft2(src)
    
    # 画像サイズ
    h, w = src.shape

    # ガウス分布の高域強調フィルタ
    kouiki_filter = gkern(h, 100)

    # 元画像にフィイルタかける
    fsrc = src * kouiki_filter

    # 高速逆フーリエ変換 
    dst = np.fft.ifft2(fsrc)
   
    # 実部の値のみを取り出し、符号なし整数型に変換して返す
    return  np.uint8(dst.real)
    
# kernlenは得られるガウス分布のサイズ, nsigはガウス分布の広がり
def gkern(kernlen, nsig):
    # kernlen x kernlen のゼロ行列を作成
    inp = np.zeros((kernlen, kernlen))

    # 中央の要素を1にする
    inp[kernlen // 2, kernlen // 2] = 1

    # ガウシアンフィルタを適用
    gaussian = fi.gaussian_filter(inp, nsig)

    # 配列を全部表示するやつ
    #np.set_printoptions(threshold=np.inf)
    
    #(縦分割数、横分割数、ポジション)
    plt.subplot(1,3,1)

    # ガウス分布表示
    plt.imshow(gaussian)

    # -1かけて逆向きにする
    gaussian = gaussian * (-1)
    plt.subplot(1,3,2)
    plt.imshow(gaussian)

    # 2を足して底が1になるようにする
    gaussian = gaussian + 2
    plt.subplot(1,3,3)
    plt.imshow(gaussian)
    plt.show()

    return gaussian

def main():
    # 入力画像を読み込み
    #img = cv2.imread("input.jpeg")
    img = cv2.imread("rena.jpg")

    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("gray.png", gray)

    # ハイパスフィルタ処理
    himg = highpass_filter(gray)

    # 結果をグラフに出力
    cv2.imwrite("output.png", himg)

if __name__ == "__main__":
    main()