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
import sys
import cv2 as cv
import numpy as np
import tensorflow as tf

#tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    # 入力層
    # MNISTデータセットは、モノクロ28x28画素画像からなる
    # 先頭の-1はバッチサイズの指定で、入力されたデータサイズに合わせて動的に決定されるという意味になる。
    # 最後の1はチャネルのこと。モノクロだから1
    input_layer = tf.reshape(features, [-1, 96, 96, 3])

    # 畳み込みレイヤ1
    # ReLUアクティベーション機能を使用して32個の5×5フィルターを入力レイヤーに適用する
    conv1 = tf.layers.conv2d(
        inputs=features,
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
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # 全結合レイヤー1
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # 精度向上のためにドロップアウト正則化を適用
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # 全結合レイヤー2(Logitsレイヤ)
    # 0-9の10通りの分類を行いたいので、ユニット(ノード)数を10にする
    logits = tf.layers.dense(inputs=dropout, units=10)

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
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)

    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # オプティマイザー。最適なモデルのパラメータを見つけるためのアルゴリズム
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


    # EVALUATION

    eval_metric_ops = {
    "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv=None):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Estimatorのインスタンス化
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir='mnist_model')

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

      # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    eval_results = classifier.evaluate(input_fn=eval_input_fn)

    print(eval_results)

def eval_input_fn(features, batch_size):
    """An input function for evaluation or prediction"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(dict(features))

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

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

if __name__ == '__main__':
    # tf.app.run()

    # 画像においてはfeature_columnsはない模様
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir='mnist_model')

    # predictメソッドはgenerator objectを返す
    prediction = classifier.predict(input_fn=input_fn)

    for i in prediction:
        print(i['classes'])























#
