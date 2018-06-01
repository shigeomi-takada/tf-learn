import glob, time
import cv2 as cv
import numpy as np
import tensorflow as tf


class Reader:

    def __init__(self):
        pass

    def _parse(self, img_path, label=None):
        """
        画像の読み込み手順
        opencvを使うこともできるが、TFに画像処理系の関数が用意されているので
        せっかくなのでTFの機能を使う
        """
        # 1. ファイルを読み込む
        # tf.read_file関数を使う
        img = tf.read_file(img_path)

        # 2. ファイルの種類に応じてデコードする
        # bmp, gif, jpeg, pngが用意されている。ファイルの拡張子を見てif文で使い分ければ良い
        # 自動的に判定してくれるtf.image.decode_image関数もあるが、エラーになったので、
        # decode_xxxシリーズを使うべし
        # 返り値は[height, width, channels]となる
        # チャネルの指定は次の通り
        # 0：そのままで出力
        # 1：グレースケールで出力
        # 3：RGB画像を出力
        # この時点ではデータ型はtf.uint8
        img = tf.image.decode_png(img, channels=1)

        # 3. 画像サイズを調整する
        # サイズが変更された画像は、元の縦横比が縦横比が同じでないと歪んで表示される
        # 歪みを避けるには、tf.image.resize_image_with_crop_or_pad関数を使う
        # この関数を使うと、縦か横かを指定して、はみ出る部分を切り取る形でresizeする
        # resize_images関数はtf.float32型で[new_height, new_width, channels]を返す
        img = tf.image.resize_images(img, [28, 28])

        # TFが扱うことのできるfloat型に変換
        # resize_imagesはtf.float32で返すので必要ない
        # img = tf.cast(img, tf.float32)

        # 4. batch_sizeを追加
        # この時点では[new_height, new_width, channels]という次元になっている
        # layers APIは[batch_size, new_height, new_width, channels]の次元を
        # 必要とするので先頭に1を追加する
        # expand_dimsは指定した箇所に1を追加する。以下のように0であれば、つまりリストの先頭的な
        # 意味合いで先頭に1を追加する
        # [1, height, width, channels] こういう形になる
        img = tf.expand_dims(img, 0)

        return img, label

    def input_fn(self, filenames, labels, batch_size=1):
        ''''''
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

        # 生の画像を読み込んだ状態なので、TFで扱えるように1画像ずつパース
        dataset = dataset.map(self._parse)

        # データをシャッフルする
        # 第1引数: データの数より多いtf.int64を指定すれば十分にシャッフルされる
        # 第2引数: seed値. seed値を指定することで、シャッフルの結果を毎回同じにすることができる
        dataset = dataset.shuffle(1000)

        # ディープラーニングは損失関数が最小化するまで学習を繰り返す
        # 指定した学習のstep数になるまで繰り返すので、それまではデータセットを使い切っても
        # 繰り返す必要がある。そのためにrepeatがある
        # 引数で何回繰り返すかを指定することができる。デフォルトは無制限に繰り返す
        dataset = dataset.repeat()

        # バッチサイズを指定する
        dataset = dataset.batch(batch_size)

        # Iterator型に変換
        iterator = dataset.make_one_shot_iterator()

        # batch_size分ずつ取得して返す
        return iterator.get_next()


image_path = 'tmp/*.png'

# 指定したパス内のファイル名をパス無しでlistで返す。
# 特定のフォルダ内のファイル名をファイル名だけ取得したい場合に使う
filenames = glob.glob(image_path)

# 0 = Cat, 1 = Dog
labels = [0 if None else 1 for img in filenames]
r = Reader()
r.input_fn(filenames, labels)

with tf.Session() as sess:
    print(sess.run(r.input_fn(filenames, labels)))
    time.sleep(1)
    print(sess.run(r.input_fn(filenames, labels)))


























#
