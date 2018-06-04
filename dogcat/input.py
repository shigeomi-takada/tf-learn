import glob, random
import tensorflow as tf

class Input:
    '''
    入力系の処理をまとめる
    '''
    def __init__(self,
        channels=3,
        height=128,
        width=128,
        batch_size=32):
        ''''''
        self.channels = channels
        self.height = height
        self.width = width
        self.batch_size = batch_size

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
        img = tf.image.decode_png(img, channels=self.channels)

        # 3. 画像サイズを調整する
        # サイズが変更された画像は、元の縦横比が縦横比が同じでないと歪んで表示される
        # 歪みを避けるには、tf.image.resize_image_with_crop_or_pad関数を使う
        # この関数を使うと、縦か横かを指定して、はみ出る部分を切り取る形でresizeする
        # resize_images関数はtf.float32型で[new_height, new_width, channels]を返す
        img = tf.image.resize_images(img, [self.height, self.width])

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
        #img = tf.expand_dims(img, 0)

        return img, label

    def input_fn(self, filenames, labels):
        ''''''
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

        # 生の画像を読み込んだ状態なので、TFで扱えるように1画像ずつパース
        dataset = dataset.map(self._parse)

        # データをシャッフルする
        # 第1引数: データの数より多いtf.int64を指定すれば十分にシャッフルされる
        # 第2引数: seed値. seed値を指定することで、シャッフルの結果を毎回同じにすることができる
        dataset = dataset.shuffle(20000)

        # ディープラーニングは損失関数が最小化するまで学習を繰り返す
        # 指定した学習のstep数になるまで繰り返すので、それまではデータセットを使い切っても
        # 繰り返す必要がある。そのためにrepeatがある
        # 引数で何回繰り返すかを指定することができる。デフォルトは無制限に繰り返す
        dataset = dataset.repeat()

        # バッチサイズを指定する
        dataset = dataset.batch(self.batch_size)

        # Iterator型に変換
        iterator = dataset.make_one_shot_iterator()

        # batch_size分ずつ取得して返す
        return iterator.get_next()

    def eval_input_fn(self, filenames, labels):
        '''
        evaluationとpredictの場合はeval_input_fnを呼び出す
        evaluationとpredictにおいてはshuffleとrepeatはいらない
        '''
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

        # 生の画像を読み込んだ状態なので、TFで扱えるように1画像ずつパース
        dataset = dataset.map(self._parse)

        # バッチサイズを指定する
        dataset = dataset.batch(self.batch_size)

        # Iterator型に変換
        iterator = dataset.make_one_shot_iterator()

        # batch_size分ずつ取得して返す
        return iterator.get_next()

    def get(self):
        '''
        Dog, cat
        http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html
        https://guillaumebrg.wordpress.com/2016/02/06/dogs-vs-cats-project-first-results-reaching-87-accuracy/
        '''
        image_path = '../datasets/dogcat/*.jpg'

        # 指定したパス内のファイル名をパス付きでlistで返す。
        filenames = glob.glob(image_path)

        # 参照渡しでシャッフル
        random.shuffle(filenames)

        # 0 = Cat, 1 = Dog
        labels = [0 if 'cat' in fname.split('/')[-1] else 1 for fname in filenames]

        # train, val, testに分ける
        return {
            'train_fnames': filenames[0:int(0.6 * len(filenames))],
            'train_labels': labels[0:int(0.6 * len(labels))],
            'val_fnames': filenames[int(0.6 * len(filenames)):int(0.8 * len(filenames))],
            'val_labels': labels[int(0.6 * len(filenames)):int(0.8 * len(filenames))],
            'test_fnames': filenames[int(0.8 * len(filenames)):],
            'test_labels': labels[int(0.8 * len(labels)):]
        }
