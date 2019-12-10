import argparse

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from model.multi_classifier import Model
from utils.utils import load_train_data_nn, load_test_data_nn, build_word_dict, batch_iter, batch_iter_eval


def train(train_x, train_y1, train_y2, test_x1, test_x2, test_y1, test_y2, word_dict, args):
    with tf.Session() as sess:
        model = Model(len(word_dict), num_class=2, args=args, vocab=word_dict)

        # Define training procedure
        global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()
        gradients = tf.gradients(model.Loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer = tf.train.AdamOptimizer(0.001)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

        # Summary
        y1_loss_summary = tf.summary.scalar("y1_loss", model.subtaska_loss)
        y2_loss_summary = tf.summary.scalar("y2_loss", model.subtaskb_loss)
        Loss_summary = tf.summary.scalar("Loss", model.Loss)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("summary", sess.graph)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(batch_x, batch_y1, batch_y2):
            feed_dict = {model.X: batch_x, model.Y1: batch_y1, model.Y2: batch_y2, model.dropout: args.dropout}
            _, step, summaries, Loss, y1_loss, y2_loss = \
                sess.run([train_op, global_step, summary_op, model.Loss, model.subtaska_loss, model.subtaskb_loss],
                         feed_dict=feed_dict)
            summary_writer.add_summary(summaries, step)

            if step % 10 == 0:
                print("step {0}: loss={1} (y1_loss={2}, y2_loss={3})".format(step, Loss, y1_loss, y2_loss))

        def evalA(test_x, test_y):

            test_batches = batch_iter_eval(test_x, test_y, args.batch_size)
            lossesA, accuraciesA, itersA, f1sA = 0, 0, 0, 0
            pred = []
            for batch_x, batch_y, in test_batches:
                feed_dict = {model.X: batch_x, model.Y1: batch_y, model.Y2: batch_y, model.dropout: args.dropout}
                y_lossA, accuracyA, f1A, preds = sess.run(
                    [model.subtaska_loss, model.suba_accuracy, model.f1_suba, model.subtaska_predictions],
                    feed_dict=feed_dict)
                lossesA += y_lossA
                accuraciesA += accuracyA
                f1sA += f1A
                itersA += 1
                pred = np.concatenate((pred, preds))
            print("test perplexity = {0}".format(np.exp(lossesA / itersA)))
            print("Test Accuracy = {0}".format(accuraciesA / itersA))
            print("Test F1 = {0}\n".format(f1_score(test_y, pred, average='macro')))

        def evalB(test_x, test_y):
            test_batches = batch_iter_eval(test_x, test_y, args.batch_size)
            lossesA, accuraciesA, itersA, f1sA, preS, recA = 0, 0, 0, 0, 0, 0
            pred = []
            for batch_x, batch_y, in test_batches:
                feed_dict = {model.X: batch_x, model.Y1: batch_y, model.Y2: batch_y, model.dropout: args.dropout}
                y_lossA, accuracyA, f1A, prec, recall, preds = sess.run(
                    [model.subtaskb_loss, model.subb_accuracy, model.f1_subb, model.precision, model.recall,
                     model.subtaskb_predictions],
                    feed_dict=feed_dict)
                lossesA += y_lossA
                accuraciesA += accuracyA
                f1sA += f1A
                itersA += 1
                pred = np.concatenate((pred, preds))
            print("test perplexity = {0}".format(np.exp(lossesA / itersA)))
            print("Test Accuracy = {0}".format(accuraciesA / itersA))
            # print("Test F1 = {0}\n".format(f1sA/itersA))
            print("Test F1 = {0}\n".format(f1_score(test_y, pred, average='macro')))

            # import pdb;pdb.set_trace()

        batches = batch_iter(train_x, train_y1, train_y2, args.batch_size, args.num_epochs)
        for batch_x, batch_y1, batch_y2 in batches:
            train_step(batch_x, batch_y1, batch_y2)
        print("\n-------------Training Ended--------------")
        print("Subtask A (Glove):")
        evalA(test_x1, test_y1)
        print("Subtask B (Glove):")
        evalB(test_x2, test_y2)
        # evalB(test_x2, test_y2, 'B')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_size", type=int, default=300, help="embedding size.")
    parser.add_argument("--num_layers", type=int, default=1, help="RNN network depth.")
    parser.add_argument("--num_hidden", type=int, default=100, help="RNN network size.")

    parser.add_argument("--dropout", type=float, default=0.3, help="dropout keep prob.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate.")
    parser.add_argument("--batch_size", type=int, default=200, help="batch size.")
    parser.add_argument("--num_epochs", type=int, default=15, help="number of epochs.")
    parser.add_argument("--max_document_len", type=int, default=150, help="max document length.")
    args = parser.parse_args()

    print("\nBuilding dictionary..")
    word_dict = build_word_dict()

    print("Preprocessing dataset..")
    train_x, train_y1, train_y2 = load_train_data_nn(word_dict, args.max_document_len)
    test_x1, test_y1 = load_test_data_nn('dataset/testset-levela-processed.tsv', 'dataset/labels-levela.csv', word_dict,
                                         args.max_document_len)
    test_x2, test_y2 = load_test_data_nn('dataset/testset-levelb-processed.tsv', 'dataset/labels-levelb.csv', word_dict,
                                         args.max_document_len)
    train(train_x, train_y1, train_y2, test_x1, test_x2, test_y1, test_y2, word_dict, args)
