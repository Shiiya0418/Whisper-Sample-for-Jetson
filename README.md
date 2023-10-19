# Whisper-Sample-for-Jetson
Whisperのサンプルコードをまとめたリポジトリです。

## 環境構築 for Jetson
以下のコマンドを順番に実行します。（クリップボードにコピーできます。）
```sh
pip install -U openai-whisper
```
```sh
sudo apt update && sudo apt install ffmpeg
```
```sh
pip install setuptools-rust
```
```sh
pip install numpy --upgrade
```
```sh
sudo apt-get install portaudio19-dev
```
```sh
pip install pyaudio
```

## 環境構築 for GPU server
上記のJetsonの環境構築に加え、fine-tuningで用いたライブラリ群をインストールします。
```sh
pip install evaluate
```

```sh
pip install pytorch-lightning
```

```sh
pip install torchaudio
```

```sh
pip install jiwer
```

リポジトリに含まれる訓練データ `data.zip` を `unzip` で解凍します。
```sh
unzip data.zip
```

## サンプルコードの実行

### 基本的なサンプルコードの動かし方
録音を行うサンプルプログラム
```sh
python3 record.py
```
`.wav` ファイルから音声認識を行うサンプルプログラム
```sh
python3 speechrecog.py [認識を行いたいファイルのファイルパス]
```

### 実験用の雑音付加コードの動かし方
以下のようにして、音声ファイル( `.wav` )に白色雑音を付与することができる。
```sh
python3 add_noise.py [雑音を付与したい音声ファイルのファイルパス]
```
### fine-tuningの動かし方
リポジトリに含まれる訓練用データを解凍します。
```sh
unzip data.zip
```

```sh
python3 finetunig.py
```

fine-tuning後のモデルを使って音声認識を試す場合は、 `data/my_audio_131_noise.wav` 以降のファイルを使用して下しさい。`data/my_audio_0_noise.wav` 〜 `data/my_audio_130_noise_.wav` のファイルは訓練に使用されています。


## scpコマンドの使い方
リモート間でファイルのやり取りを行うコマンドに `scp` というものがある。

基本的な使い方は以下のとおりである。

```sh
scp [オプション] 送信側のパス 受信側のパス
```

この時、リモート側のパスは以下の規則で記述する。

```sh
ユーザ名@IPアドレス:ファイルパス
```

例えば、自分の Jetson から自分の接続しているPCのカレントディレクトリへファイルを送信する場合は、以下のようにする。

```sh
scp jetson@192.168.11.[Jetson番号]:~/work_dir/test.wav ./
```

もしディレクトリごとコピーしたい場合は、 `-r` オプションをつける。

```sh
scp -r jetson@192.168.11.[Jetson番号]:~/work_dir ./
```

## 未マウントの領域を割り当てる方法

次のコマンドを実行してください。
```sh
sudo parted /dev/mmcblk0 resizepart 1 80%
```
もし `Fix/Ignore?` と聞かれたら、 `Fix` と入力して `Enter` キーを入力して下しさい。

もし `Are you sure want to continue... Yes/No?` のように聞かれたら、 `Yes` と入力して `Enter` キーを入力してください。

`End?   [31,3GB]?` と聞かれるので `80%` と入力して `Enter` キーを入力してください。

```sh
sudo resize2fs /dev/mmcblk0p1
```

もしもJetsonにログインできない場合は、直接Jetsonにキーボードとディスプレイを差し、 `ctl` キー、 `alt` キー、 `F2` キーを同時に押してください。 CLIモードで Jetsonを操作できます。

## cache領域の拡大方法
以下のコマンドを順番に実行してください。

こちらが参考になります。（ (参考ページ)[https://www.hiramine.com/physicalcomputing/jetsonnano/swap_check_extend.html] ）

```sh
sudo fallocate -l 4G /swapfile
```

```sh
sudo chmod 600 /swapfile
```

```sh
sudo mkswap/swapfile
```

次に、 `/etc/fstab` を編集します。お好みのエディタで開いてください。

```sh
# 例) vimの場合
sudo vim /etc/fstab
```

ファイルの末尾に以下を追記します。

```
/swapfile none swap defaults 0 0
```

最後に再起動を行います。
```sh
sudo reboot
```

### 補足
fine-tuning用訓練データは [VOICEVOX](https://voicevox.hiroshiba.jp/) を用いて作られました。

©︎VOICEVOX:ずんだもん [VOICEVOX](https://voicevox.hiroshiba.jp/)