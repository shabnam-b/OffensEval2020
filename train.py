import tensorflow as tf

from model.multi_classifier import MultiClassifier
from utils.args_helper import parse_training_parameters
from utils.utils import batch_iter


def train(train_x, train_y1, train_y2, test_x, test_y1, test_y2, FLAGS):

    # Starting Tensorflow session
    with tf.Session() as sess:
        multi_classifier = MultiClassifier(num_labels=2, FLAGS=FLAGS)

        global_step = tf.Variable(0, trainable=False)
        parameters = tf.trainable_variables()
        grads = tf.gradients(multi_classifier.Loss, parameters)
        clipped_grads, _ = tf.clip_by_global_norm(grads, FLAGS.clip_norm)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        training_opt = optimizer.apply_gradients(zip(clipped_grads, parameters), global_step=global_step)

        # t1_loss = tf.summary.scalar("Loss 1", multi_classifier.loss_t1)
        # t2_loss = tf.summary.scalar("Loss 2", multi_classifier.loss_t2)
        # Loss = tf.summary.scalar("Total Loss", multi_classifier.Loss)
        op = tf.summary.merge_all()
        writer = tf.summary.FileWriter("summary", sess.graph)

        sess.run(tf.global_variables_initializer())

        def train_step(batch_in, batch_y1, batch_y2):

            feed_dict = {multi_classifier.inp: batch_in, multi_classifier.y_t1: batch_y1,
                         multi_classifier.y_t2: batch_y2,
                         multi_classifier.dropout: FLAGS.keep_prob}
            _, step, summaries, Loss, loss_y1, loss_y2 = \
                sess.run([training_opt, global_step, op, multi_classifier.Loss, multi_classifier.loss_t1,
                          multi_classifier.loss_t2],
                         feed_dict=feed_dict)
            writer.add_summary(summaries, step)

            if step % 100 == 0:
                print("Step {0}: Total Loss={1} ( Loss(Y1)={2}, Loss(Y2) = {3} )".format(step, Loss, loss_y1, loss_y2))

        def eval_step(test_x, test_y1, test_y2):
            pass

        batches = batch_iter(train_x, train_y1, train_y2, FLAGS.batch_size, FLAGS.num_epochs)
        for batch_x, batch_y1, batch_y2 in batches:
            train_step(batch_x, batch_y1, batch_y2)
            step = tf.train.global_step(sess, global_step)

            if step % 1000 == 0:
                # TODO to be imolemented
                eval_step(test_x, test_y1, test_y2)


if __name__ == '__main__':
    FLAGS = parse_training_parameters(tf.flags)

    """
 
        TODO : loading dataset...
    
    """
    train(FLAGS)
