# `mem_spectrum` による最大エントロピースペクトル

`src/enzanlab/signal/spectrum/mem_spectrum.py` は複素信号に対する Maximum Entropy Method (MEM)／自己回帰 (AR) スペクトル推定を実装している。`order` 次の AR($p$) モデルを当てはめ、そのパラメータからパワースペクトル密度 (Power Spectral Density; PSD) を評価する。以下では処理手順を数式で説明する。

## 1. 自己相関列の計算
長さ $N$ の複素系列 $x[n]$ に対し、`autocorr_mem` はラグ $p$ までのバイアス付き自己相関 $\hat r[k]$ を計算する。

$$
\hat r[k] = \frac{1}{N-k} \sum_{n=0}^{N-k-1} x^{*}[n] \, x[n+k], \qquad k = 0,\dots,p.
$$

実装では `numpy.vdot` を用いているため、第1引数が自動的に共役される。またエネルギー項が厳密に実数となるように $\hat r[0] \leftarrow \operatorname{Re}(\hat r[0])$ を適用し、以降の再帰計算を安定化させている。

## 2. Levinson–Durbin 再帰
`levinson_durbin` は Yule–Walker 方程式

$$
\begin{bmatrix}
\hat r[0] & \hat r[1] & \dots & \hat r[p-1] \\
\hat r^{*}[1] & \hat r[0] & \dots & \hat r[p-2] \\
\vdots & \vdots & \ddots & \vdots \\
\hat r^{*}[p-1] & \hat r^{*}[p-2] & \dots & \hat r[0]
\end{bmatrix}
\begin{bmatrix}
1 \\
a_1 \\
\vdots \\
a_p
\end{bmatrix}
=
\begin{bmatrix}
\hat r[0] \\
-\hat r[1] \\
\vdots \\
-\hat r[p]
\end{bmatrix}
$$

を逐次的に解く。次数 $m$ での更新は反射係数 (reflection coefficient; Parcor) $\kappa_m$ を

$$
\kappa_m = -\frac{\hat r[m] + \sum_{k=1}^{m-1} a_k^{(m-1)} \, \hat r[m-k]}{e_{m-1}}
$$

と求め、AR 係数および予測誤差エネルギーを

$$
a_m^{(m)} = \kappa_m, \qquad a_k^{(m)} = a_k^{(m-1)} + \kappa_m \; a_{m-k}^{(m-1)*},
$$
$$
e_m = e_{m-1} \left(1 - |\kappa_m|^2\right)
$$

で更新する。丸め誤差で $e_m$ が負になることを防ぐため、実装では `stability_eps` で下限を設けている。最終的に $\boldsymbol{a} = [a_1,\dots,a_p]$ と残差分散 $\sigma_e^2 = e_p$ が得られる。

## 3. スペクトル評価
`mem_spectrum` は得られた AR モデルから一側 PSD を算出する。AR($p$) モデルは

$$
x[n] + \sum_{k=1}^{p} a_k x[n-k] = e[n]
$$

と書ける。ここで $e[n]$ は分散 $\sigma_e^2$ の白色雑音であり、$z^{-1}$ 領域では

$$
X(z) = \frac{E(z)}{1 + \sum_{k=1}^{p} a_k z^{-k}}
$$

となる。よって周波数応答 $H(e^{j\omega})$ は

$$
H(e^{j\omega}) = \frac{1}{1 + \sum_{k=1}^{p} a_k e^{-j\omega k}} = \frac{1}{A(e^{j\omega})}
$$

であり、入力が白色雑音であるため PSD は

$$
P_{xx}(f) = |H(e^{j\omega})|^2 \, \sigma_e^2 = \frac{\sigma_e^2}{\left|A(e^{j\omega})\right|^2}
$$

と導かれる。実装では $0$ から $f_s/2$ までを `n_freq` 分割した周波数グリッド上で、正規化角周波数 $\omega = 2\pi f / f_s$ における分母多項式

$$
A(e^{j\omega}) = 1 + \sum_{k=1}^{p} a_k e^{-j\omega k}
$$

を構成し、

を構成し、上式を用いて PSD を計算する。浮動小数誤差で $\sigma_e^2$ がわずかに負になるケースに備え、分散も `stability_eps` でクリップする。戻り値は周波数配列 `freqs` (Hz) と PSD `Pxx` であり、例えば `10 * log10(Pxx)` で dB 表示できる。

## 4. 調整パラメータ
- `order`: AR モデル次数。大きいほど分解能は上がるが過学習しやすい。
- `n_freq`: PSD を離散評価する点数。
- `fs`: サンプリング周波数。角周波数を Hz に変換するために使用。
- `stability_eps`: 再帰および PSD を正定に保つための小さい定数。

以上が `mem_spectrum` における MEM 推定の数式的ステップであり、ソースコードとの対応づけに役立つ。
