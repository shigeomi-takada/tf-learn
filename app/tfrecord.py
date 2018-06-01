from random import shuffle
import glob
import cv2 as cv
import tensorflow as tf
import numpy as np


class Tfrecord:
    '''
    画像ファイルからTFRecordを作成する

    '''

    def __init__(self):
        '''initialize'''
        self.SAVE_TO = 'tmp/train_1.tfrecords'

    def _int64_feature(self, value):
        '''お作法として覚える'''
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        '''お作法として覚える'''
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _load_image(self, path, row=28, col=28):
        '''
        row: 横, 行, width
        col: 縦, 列, height
        '''
        img = cv.imread(path)
        # INTER_CUBICは処理時間かかるが滑らかにresizeする
        img = cv.resize(img, (row, col), interpolation=cv.INTER_CUBIC)
        # opencvはBGRで画像を読み込むので、RGBに変換する
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # TFが扱えるようにするためにfloat32に変換する
        img = img.astype(np.float32)

        return img

    def _write(self, images, labels):
        '''
        データとラベルのセットのTFRecordを作成する
        '''
        # open the TFRecords file
        with tf.python_io.TFRecordWriter(self.SAVE_TO) as w:
            for i in range(len(images)):

                # 100の倍数の場合に経過報告
                if not i % 100:
                    print('write: {}/{}'.format(i, len(images)))

                # Load the image
                image = self._load_image(images[i])
                label = labels[i]

                # Create an example protocol buffer
                # 以下のプログラムはお作法として覚えてしまえばいい
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'image': self._bytes_feature(
                                tf.compat.as_bytes(image.tostring())),
                            'label': self._int64_feature(label),
                        }))

                # Serialize to string and write on the file
                w.write(example.SerializeToString())

    def input_fn():
        ########################################
        # 画像の読み込み手順
        ########################################

        # 1. ファイルを読み込む
        # tf.read_file関数を使う
        img = tf.read_file('tmp/d.png')

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

        # 5. Dataset型に変換
        # すでにテンソル型になっているので、from_tensor_slicesを使う
        dataset = tf.data.Dataset.from_tensor_slices(img)

        # 6.
        dataset = dataset.batch(1)

        # 7. Iterator型に変換
        return dataset.make_one_shot_iterator().get_next()

    def write(self):
        ''''''
        shuffle_data = True  # shuffle the addresses before saving

        image_path = 'tmp/*.png'

        # 指定したパス内のファイル名をパス無しでlistで返す。
        # 特定のフォルダ内のファイル名をファイル名だけ取得したい場合に使う
        image_names = glob.glob(image_path)

        # 0 = Cat, 1 = Dog
        labels = [0 if None else 1 for img in image_names]

        # シャッフルする
        if shuffle_data:
            c = list(zip(image_names, labels))
            shuffle(c)
            images, labels = zip(*c)

        tfr = Tfrecord()
        tfr._write(images, labels)

    def _parse_func(self, example_proto):
        '''
        TFRecordに変換した画像ファイルを読み込む際に元に戻すメソッド
        仕組みがどうというよりもお決まりのパターンなので覚えてしまえばよい
        '''
        # 書き込んだ際のデータ構造に合わせる
        features = {
            "image": tf.FixedLenFeature((), tf.string, default_value=""),
            "label": tf.FixedLenFeature((), tf.int64, default_value=0),
        }

        # 第1引数を第2引数の形にパースする
        parsed_features = tf.parse_single_example(example_proto, features)

        return parsed_features["image"], parsed_features["label"]


def main(unused_argv):

    tfr = Tfrecord()

    dataset = tf.data.TFRecordDataset(['tmp/train_1.tfrecords'])
    dataset = dataset.map(tfr._parse_func)
    dataset = dataset.repeat()  # Repeat the input indefinitely.
    dataset = dataset.batch(1)
    iterator = dataset.make_initializable_iterator()

if __name__ == '__main__':
    tf.app.run()
