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