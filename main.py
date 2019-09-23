import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.python.data.experimental.ops import resampling
import pandas as pd
import datetime
from datetime import timedelta
# from tensorflow.contrib.distribute import MirroredStrategy

# assert label+ = 1 label- = 0

bz = 128
maxstep = 1000000
colums_num = 65
# dim = 125
cores = 4

lr = 0.01
eval_bz = 512

tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

tf.app.flags.DEFINE_string("lr", "", "lr")

tf.app.flags.DEFINE_string("exporter_dir", '', "Directory for save model")
tf.app.flags.DEFINE_string("data_path", "hdfs://default/home/hdp_lbg_ectech/resultdata/strategy/ads/linJQ_test/dssm_sample/*/part*", "data path")

tf.app.flags.DEFINE_integer("batch_size", 128, "batch size")
tf.app.flags.DEFINE_integer("eval_batch_size", 512, "eval batch size")
tf.app.flags.DEFINE_integer("max_step", 10000, "max step")
tf.app.flags.DEFINE_integer("vali_num", 1, "vail num")

FLAGS = tf.app.flags.FLAGS

train_hook_list = []
eval_hook_list = []
export_hook_list = []



columns = []
columns.append(tf.feature_column.numeric_column(key='feats',shape=[64]))
lab_column = []
lab_column.append(tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity('label', num_buckets=2)))

record_defaults = []
record_defaults.append([0])
record_defaults.append([""])

record_numeric = [[0.0] for i in range(0, colums_num-1)]

def parseLine(line):
    tmp = tf.decode_csv(line, field_delim='\t', record_defaults=record_defaults)
    res = []
    res.append(tmp[0])
    nums = tf.decode_csv(tmp[1], field_delim=',', record_defaults=record_numeric)
    res.append(nums)
    return res

def get_data(filename_reg, vali_num, epoch_num, bz):
    files = tf.data.Dataset.list_files(filename_reg)
    files = files.shuffle(2048)

    lines_dataset = files.apply(tf.data.experimental.parallel_interleave(
        tf.data.TextLineDataset, cycle_length=cores))
    # lines_dataset.cache()
    if epoch_num == 1:
        # lines_dataset = lines_dataset.shard(10, worker_index)
        lines_dataset = lines_dataset.take(vali_num)
    else:
        lines_dataset = lines_dataset.skip(vali_num)
        lines_dataset = lines_dataset.repeat()

    # lines_dataset = lines_dataset.repeat(epoch_num)
    lines_dataset = lines_dataset.map(parseLine, num_parallel_calls=cores)

    lines_dataset = lines_dataset.shuffle(32768)

    lines_dataset = lines_dataset.batch(batch_size=bz).prefetch(16)
    iterator = lines_dataset.make_initializable_iterator()
    # print(iterator.initializer.name)
    dataset_init = iterator.make_initializer(lines_dataset, name='dataset_init')
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, dataset_init)

    btData = iterator.get_next()

    feat_select_format = {}
    labs = {}
    feat_select_format["feats"] = btData[1]
    labs['label'] = btData[0]
    return feat_select_format, labs

def buildModel(features, labels, mode, params):
    tf.add_to_collection('maxstep', params['maxstep'])
    tf.add_to_collection("features", features)

    X = tf.feature_column.input_layer(features, columns)
    tf.add_to_collection("X", X)

    x_usr, x_ad = tf.split(X,[32,32],axis=1)
    # x_usr_mod1 = x_usr * (1 / tf.sqrt(tf.reduce_sum(tf.multiply(x_usr, x_usr),axis=1)))
    x_usr_mod1 = tf.nn.l2_normalize(x_usr,axis=1)
    x_ad_mod1 = tf.nn.l2_normalize(x_ad, axis=1)

    # x_usr_mod1_h0 = tf.layers.dense(x_usr_mod1,32, "tanh", kernel_initializer=tf.initializers.truncated_normal(), name='h0')
    x_usr_mod1_h1 = tf.layers.dense(x_usr_mod1, 32, "tanh", kernel_initializer=tf.initializers.truncated_normal(), name='h1')

    x_usr_mod1_h1_mod1 = tf.nn.l2_normalize(x_usr_mod1_h1, axis=1)
    tf.add_to_collection("x_usr_mod1_h1_mod1", x_usr_mod1_h1_mod1)

    y_hat = tf.divide((tf.reduce_sum(tf.multiply(x_usr_mod1_h1_mod1, x_ad_mod1),axis=1)+1.0), 2.0, name='binary_logistic_head/predictions/probabilities')
    tf.add_to_collection("y_hat", y_hat)

    # if mode == tf.estimator.ModeKeys.PREDICT:
    #     return tf.estimator.EstimatorSpec(
    #         mode=mode,
    #         predictions=y_hat)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=x_usr_mod1_h1_mod1,
            export_outputs={
                'output_dssm_vector': tf.estimator.export.PredictOutput(x_usr_mod1_h1_mod1)
            })
    y = tf.feature_column.input_layer(labels, lab_column)
    tf.add_to_collection("y_sum", tf.reduce_sum(y, axis=0))
    y_1 = tf.split(tf.cast(y,tf.float32), 2, axis=1)[1]
    loss = tf.losses.cosine_distance(labels=x_ad_mod1*(y_1-0.5)*2, predictions=x_usr_mod1_h1_mod1, axis=1)

    auc, aucUpdate = tf.metrics.auc(y_1, y_hat, num_thresholds=2000)

    tf.add_to_collection("loss", loss)
    tf.add_to_collection("auc", auc)
    tf.add_to_collection("aucUpdate", aucUpdate)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops={'eval_accuracy': (auc, aucUpdate)},
            evaluation_hooks=None)

    tf.summary.scalar('auc_train', aucUpdate)
    lr_plhd = tf.placeholder(dtype=tf.float32)
    tf.add_to_collection('lr_plhd', lr_plhd)
    train_op = tf.train.GradientDescentOptimizer(learning_rate=lr_plhd).minimize(loss,
                                                                                 global_step=tf.train.get_global_step(),
                                                                                 name='opti')
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=y_hat,
            loss=loss,
            train_op=train_op)

    else:
        print("error: no mode name: "+mode+" !")
        return None


class MyHookTrain(tf.train.SessionRunHook):
    """Logs loss and runtime."""

    def begin(self):
        self._global_step_tensor = tf.train.get_or_create_global_step()
        self.step_cur = 0
        self.maxstep = tf.get_collection('maxstep')[0]

    def before_run(self, run_context):
        lr_plhd = tf.get_collection('lr_plhd')[0]
        lr_cur = lr

        if self.step_cur % 10 == 0:
            print('lr: %f'%lr_cur)
            return tf.train.SessionRunArgs(
                [
                    self._global_step_tensor,
                    tf.get_collection('loss')[0],
                    tf.get_collection('aucUpdate')[0],
                    tf.get_collection('auc')[0],
                    # tf.get_collection('y_sum')[0],
                    # tf.get_collection('X')[0],
                    # tf.get_collection('x_usr_mod1_h1_mod1')[0],
                    tf.get_collection('y_hat')[0]
                ],  # Asks for global step value.

                feed_dict={lr_plhd: lr_cur}
            )  # Sets learning rate
        else:
            return tf.train.SessionRunArgs(
                [
                    self._global_step_tensor
                ],  # Asks for global step value.
                feed_dict={lr_plhd: lr_cur}
            )  # Sets learning rate

    def after_run(self, run_context, run_values):
        # global step_cur
        res = run_values.results
        if self.step_cur % 10 == 0:
            print("#############################################################################")
            print(' self.step_cur: %d'%self.step_cur)
            print(res)
        self.step_cur = res[0]
        if self.step_cur >= self.maxstep:
            run_context.request_stop()


class MyHookEval(tf.train.SessionRunHook):
    """Logs loss and runtime."""
    def __init__(self,FM,serving_input_receiver_fn):
        self.serving_input_receiver_fn = serving_input_receiver_fn
        self.FM = FM
        # self.ischief = ischief

    def begin(self):
        self.baseLine = int(0.700*1000)
        self.ress=[]

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(
            [
                tf.get_collection('aucUpdate')[0],
                tf.get_collection('auc')[0]
            ],
        )

    def after_run(self, run_context, run_values):
        res = run_values.results
        self.ress.append(res[1])
        print("**********************************************************************")
        print(res)

    def end(self,session):
        res = int(np.sum(self.ress) / len(self.ress) * 1000)
        # if res > self.baseLine:
        self.FM.export_savedmodel(FLAGS.exporter_dir, self.serving_input_receiver_fn)
        print("************************@@@@@@@@@@@@@  eval end  @@@@@@@@@@@@**************************")
        print(res)

class EvalResultsExporter(tf.estimator.Exporter):
    def __init__(self, name):
        assert name, '"name" argument is required.'
        self._name = name

    @property
    def name(self):
        return self._name

    def export(self, estimator, export_path, checkpoint_path, eval_result, is_the_final_export):
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(export_path)
        print(checkpoint_path)
        print(eval_result)
        # if is_the_final_export:


def main(_):
    global bz, maxstep, lr, lambda_w, lambda_v, k,  FM, serving_input_receiver_fn,eval_bz

    data_path = FLAGS.data_path

    print(data_path)
    bz = FLAGS.batch_size
    eval_bz = FLAGS.eval_batch_size

    maxstep = FLAGS.max_step

    # lr = float(FLAGS.lr)

    # ps_hosts = FLAGS.ps_hosts.split(",")
    # worker_hosts = FLAGS.worker_hosts.split(",")
    # jobtype = FLAGS.job_name
    # taskid = FLAGS.task_index
    #
    # if FLAGS.job_name == 'worker' and FLAGS.task_index == 0:
    #     jobtype = 'chief'
    #
    # if FLAGS.job_name == 'worker' and FLAGS.task_index != 0 and FLAGS.task_index != len(worker_hosts) - 1:
    #     taskid = taskid - 1
    #
    # if FLAGS.job_name == 'worker' and FLAGS.task_index == len(worker_hosts) - 1:
    #     jobtype = 'evaluator'
    #     taskid = 0
    #
    # cluster = {'chief': [worker_hosts[0]],
    #            "ps": ps_hosts, "worker": worker_hosts[1:]}
    # os.environ['TF_CONFIG'] = json.dumps(
    #     {'cluster': cluster,
    #      'task': {'type': jobtype, 'index': taskid}})

    config = tf.estimator.RunConfig(save_checkpoints_secs=300)
    params = {'maxstep': maxstep}
    params['train_bz'] = bz
    params['eval_bz'] = eval_bz

    FM = tf.estimator.Estimator(model_fn=buildModel, model_dir=FLAGS.exporter_dir, config=config, params=params)

    feature_spec = tf.feature_column.make_parse_example_spec(columns)
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

    myhooktrain = MyHookTrain()
    train_hook_list.append(myhooktrain)

    myhookeval = MyHookEval(FM=FM,serving_input_receiver_fn=serving_input_receiver_fn)
    eval_hook_list.append(myhookeval)
    myhookexportor = EvalResultsExporter(name='myExportor')
    export_hook_list.append(myhookexportor)

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: get_data(data_path, FLAGS.vali_num, epoch_num=None, bz=bz),
                                        hooks=train_hook_list)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: get_data(data_path, FLAGS.vali_num, epoch_num=1, bz=eval_bz), steps=200,
                                      start_delay_secs=60, throttle_secs=300, hooks=eval_hook_list,exporters=export_hook_list)

    tf.estimator.train_and_evaluate(FM, train_spec, eval_spec)
    # tf.estimator.train_and_evaluate(FM, train_spec, eval_spec)

if __name__ == "__main__":
    tf.app.run()
