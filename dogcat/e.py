'''
畳み込みニューラルネットワークは画像分類タスクのための現在の最先端のモデルアーキテクチャである。

CNNには3つのコンポーネントがある。この3つのコンポーネントが含まれるDNNがCNNと呼ばれる
畳み込みレイヤー: 画像を畳み込む。ReLUをアクティベーション関数に用いることで、
モデルに非線形性を導入する
プールレイヤー: 画像データを縮小する。ダウンサンプリングと呼ぶ
全結合レイヤー: 畳み込みレイヤによって抽出され、プールレイヤによってダウンサンプリングされた特徴を全て連結する

TensorFlowのドキュメントで紹介されているCNN例
畳み込みレイヤ1
プールレイヤ1
畳み込みレイヤ2
プールレイヤ2
全結合レイヤ1
全結合レイヤ2（Logitsレイヤ）
'''
import sys, glob, random
import cv2 as cv
import numpy as np
import tensorflow as tf

class Reader:
    '''

    '''

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
        img = tf.image.decode_png(img, channels=3)

        # 3. 画像サイズを調整する
        # サイズが変更された画像は、元の縦横比が縦横比が同じでないと歪んで表示される
        # 歪みを避けるには、tf.image.resize_image_with_crop_or_pad関数を使う
        # この関数を使うと、縦か横かを指定して、はみ出る部分を切り取る形でresizeする
        # resize_images関数はtf.float32型で[new_height, new_width, channels]を返す
        img = tf.image.resize_images(img, [128, 128])

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

    def input_fn(self, filenames, labels, batch_size=32):
        ''''''
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

        # 生の画像を読み込んだ状態なので、TFで扱えるように1画像ずつパース
        dataset = dataset.map(self._parse)

        # データをシャッフルする
        # 第1引数: データの数より多いtf.int64を指定すれば十分にシャッフルされる
        # 第2引数: seed値. seed値を指定することで、シャッフルの結果を毎回同じにすることができる
        dataset = dataset.shuffle(10000)

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

    def eval_input_fn(self, filenames, labels, batch_size=2):
        ''''''
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

        # 生の画像を読み込んだ状態なので、TFで扱えるように1画像ずつパース
        dataset = dataset.map(self._parse)

        # eval or predictにおいてはshuffleとrepeatはいらない

        # バッチサイズを指定する
        dataset = dataset.batch(batch_size)

        # Iterator型に変換
        iterator = dataset.make_one_shot_iterator()

        # batch_size分ずつ取得して返す
        return iterator.get_next()

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    # 入力層
    # MNISTデータセットは、モノクロ28x28画素画像からなる
    # 先頭の-1はバッチサイズの指定で、入力されたデータサイズに合わせて動的に決定されるという意味になる。
    # 最後の1はチャネルのこと。モノクロだから1
    input_layer = tf.reshape(features, [-1, 128, 128, 3])

    # 畳み込みレイヤ1
    # ReLUアクティベーション機能を使用して32個の5×5フィルターを入力レイヤーに適用する
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)

    # プールレイヤー1
    # 最初のプールレイヤーを作成したばかりの畳み込みレイヤに接続する
    # 2ピクセルずつストライドさせるので、画像サイズは半分に縮小できる
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # 畳み込みレイヤ2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)

    # プールレイヤー2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # プールレイヤーを2次元にする
    pool2_flat = tf.reshape(pool2, [-1, 32 * 32 * 64])

    # 全結合レイヤー1
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # 精度向上のためにドロップアウト正則化を適用
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # 全結合レイヤー2(Logitsレイヤ)
    # 0-9の10通りの分類を行いたいので、ユニット(ノード)数を10にする
    logits = tf.layers.dense(inputs=dropout, units=2)

    # logitsはつまり予測結果の数値である。0-9それぞれに対しての予測の数値が割り当てられている
    # 最も高い数値を抽出するにはtf.argmax関数を使えば良い
    # logitsを確率に変換するためにsoftmax関数を用いる
    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    # PREDICTの場合は予測を含むEstimatorSpecオブジェクトを返す必要がある
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # TRAINとEVALUATIONの場合はモデルの損失を計算する必要がある
    # モデルの予測が目標クラスにどれだけ近づくかを測定するために損失関数を用いる
    # 多クラス分類問題の場合は、クロスエントロピーを用いることが一般的である

    # labelsをワンホットエンコーディングに変換する
    # tf.castは型変換関数, labelsをtf.int32型に変換する
    # depthは要するにクラス数のこと
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)

    # 多クラス分類の場合はsoftmax_cross_entropyが良い結果が出る
    # TFにはsoftmax_cross_entropyとsparse_softmax_cross_entropyの似たような
    # softmaxシリーズが用意されているが、違いは、ワンホットエンコーディング処理を内包しているか否かだけ。
    # 次のloss1-3は全て同じ値を返す
    # self.layers['cls_score']: logits
    # self.y: integer labels such as [1, 2, 0, 0, 1]
    # loss1 = tf.losses.sparse_softmax_cross_entropy(logits=self.layers['cls_score'], labels=self.y)
    # y_onehot = tf.one_hot(self.y, depth=3)
    # loss2 = tf.reduce_mean(-tf.reduce_sum(y_onehot * tf.log(tf.nn.softmax(self.layers['cls_score'])), [1]))
    # loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.layers['cls_score'], labels=y_onehot))

    # loss = tf.reduce_mean(
    #    tf.losses.sparse_softmax_cross_entropy(labels, logits))

    # 2値分類の場合はhinge_lossが良い
    loss = tf.reduce_mean(
        tf.losses.hinge_loss(tf.cast(onehot_labels, tf.float32), logits))

    if mode == tf.estimator.ModeKeys.TRAIN:
        # オプティマイザー。最適なモデルのパラメータを見つけるためのアルゴリズム
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)


    # EVALUATION
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    print('train, eval, predictのいづれかを指定すべし')

def main(unused_argv=None):
    '''
    Dog, cat
    http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html
    https://guillaumebrg.wordpress.com/2016/02/06/dogs-vs-cats-project-first-results-reaching-87-accuracy/
    '''
    image_path = 'datasets/dogcat/*.jpg'

    # 指定したパス内のファイル名をパス無しでlistで返す。
    # 特定のフォルダ内のファイル名をファイル名だけ取得したい場合に使う
    filenames = glob.glob(image_path)

    random.shuffle(filenames)

    # 0 = Cat, 1 = Dog
    labels = [0 if 'cat' in fname else 1 for fname in filenames]
    r = Reader()

    # train, val, testに分ける
    train_fnames = filenames[0:int(0.6 * len(filenames))]
    train_labels = labels[0:int(0.6 * len(labels))]
    val_fnames = filenames[int(0.6 * len(filenames)):int(0.8 * len(filenames))]
    val_labels = labels[int(0.6 * len(filenames)):int(0.8 * len(filenames))]
    test_fnames = filenames[int(0.8 * len(filenames)):]
    test_labels = labels[int(0.8 * len(labels)):]

    # Estimatorのインスタンス化
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir='models/dogcat')

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # train
    classifier.train(
        input_fn=lambda:r.input_fn(train_fnames, train_labels),
        steps=20000,
        hooks=[logging_hook])

    # eval
    eval_results = classifier.evaluate(
        input_fn=lambda:r.eval_input_fn(val_fnames, val_labels))

    print(eval_results)


if __name__ == '__main__':
    tf.app.run()
