import tensorflow as tf

from multi_classifier.multi_classifier import MultiClassifier
# from multi_classifier import Model
# from data_utils import download_dbpedia, build_word_dict, build_dataset, batch_iter
from utils.FLAGS_helper import parse_training_parameters


def train(FLAGS):

    # Starting Tensorflow session
    with tf.Session() as sess:
        multi_classifier = MultiClassifier(num_labels=2, FLAGS=FLAGS)

        global_step = tf.Variable(0, trainable=False)
        parameters = tf.trainable_variables()
        grads = tf.gradients(multi_classifier.Loss, parameters)
        clipped_grads, _ = tf.clip_by_global_norm(grads, FLAGS.clip_norm)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        training_opt = optimizer.apply_gradients(zip(clipped_grads, parameters), global_step=global_step)
        
        t1_loss = tf.summary.scalar("Loss 1", multi_classifier.loss_t1)
        t2_loss = tf.summary.scalar("Loss 2", multi_classifier.loss_t2)
        total_loss = tf.summary.scalar("Total Loss", multi_classifier.Loss)
        op = tf.summary.merge_all()
        writer = tf.summary.FileWriter("summary", sess.graph)

        sess.run(tf.global_variables_initializer())

        def train_step(batch_in, batch_y1, batch_y2):
            feed_dict = {multi_classifier.inp: batch_in, multi_classifier.lm_y: batch_y1, multi_classifier.clf_y: batch_y2,
                         multi_classifier.keep_prob: FLAGS.keep_prob}
            _, step, summaries, total_loss, lm_loss, clf_loss = \
                sess.run([training_opt, global_step, op, multi_classifier.total_loss, multi_classifier.lm_loss, multi_classifier.clf_loss],
                         feed_dict=feed_dict)
            writer.add_summary(summaries, step)

            if step % 100 == 0:
                print("step {0}: loss={1} (lm_loss={2}, clf_loss={3})".format(step, total_loss, lm_loss, clf_loss))

        def eval(test_x, test_lm_y, test_clf_y):
            test_batches = batch_iter(test_x, test_lm_y, test_clf_y, FLAGS.batch_size, 1)
            losses, accuracies, iters = 0, 0, 0

            for batch_x, batch_lm_y, batch_clf_y in test_batches:
                feed_dict = {multi_classifier.x: batch_x, multi_classifier.lm_y: batch_lm_y, multi_classifier.clf_y: batch_clf_y, multi_classifier.keep_prob: 1.0}
                lm_loss, accuracy = sess.run([multi_classifier.lm_loss, multi_classifier.accuracy], feed_dict=feed_dict)
                losses += lm_loss
                accuracies += accuracy
                iters += 1

            print("\ntest perplexity = {0}".format(np.exp(losses / iters)))
            print("test accuracy = {0}\n".format(accuracies / iters))

        batches = batch_iter(train_x, train_lm_y, train_clf_y, FLAGS.batch_size, FLAGS.num_epochs)
        for batch_x, batch_lm_y, batch_clf_y in batches:
            train_step(batch_x, batch_lm_y, batch_clf_y)
            step = tf.train.global_step(sess, global_step)

            if step % 1000 == 0:
                eval(test_x, test_lm_y, test_clf_y)


if __name__ == '__main__':
    FLAGS = parse_training_parameters(tf.flags)
    train(FLAGS)
