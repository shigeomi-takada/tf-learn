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

import cv2 as cv
import uuid
import numpy as np
import tensorflow as tf
from input import Input

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

    # 2値分類の場合はhinge_lossまたは sigmoid_cross_entropy が良い
    loss = tf.reduce_mean(
        tf.losses.sigmoid_cross_entropy(onehot_labels, logits))

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

    # 毎回チェックポイントデータを消すのは面倒なので、保存ディレクトリを実行ごとに生成する
    model_dir = '../models/dogcat/' + str(uuid.uuid1()).split('-')[0]

    input = Input()

    data = input.get()
    model_dir = '../models/dogcat/'
    # Estimatorのインスタンス化
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=model_dir)

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    '''
    # train
    classifier.train(
        input_fn=lambda:input.input_fn(data['train_fnames'], data['train_labels']),
        steps=20000,
        hooks=[logging_hook])

    # eval
    eval_results = classifier.evaluate(
        input_fn=lambda:input.eval_input_fn(data['val_fnames'], data['val_labels']),
        steps=1000,
        hooks=[logging_hook])

    print(eval_results)
    '''

    # Predict
    '''
    predictメソッドの返り値例
    74行目くらいで指定した通りの値が返る
    {'classes': 1, 'probabilities': array([0.04826169, 0.9517383 ], dtype=float32)}
    '''

    data = {
        'test_fnames': ['../datasets/tmp/c.jpg'],
        'test_labels': [0]
    }

    predictions = classifier.predict(
        input_fn=lambda:input.eval_input_fn(data['test_fnames']))

    print('0: Cat, 1: Dog')
    for i, pred in enumerate(predictions):
        print('File: {0}, 予想: {1}, 正解: {2}, 確率: {3}%, {4}'.format(
            data['test_fnames'][i].split('/')[-1],
            pred['classes'],
            data['test_labels'][i],
            round(np.max(pred['probabilities']) * 100),
            '正' if pred['classes'] == data['test_labels'][i] else '誤'))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
