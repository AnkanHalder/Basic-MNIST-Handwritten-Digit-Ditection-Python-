import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
print_tensors_in_checkpoint_file(file_name="./models2/try.ckpt", tensor_name='', all_tensors=True)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_input = mnist.train.images.shape[1]  # 28x28 = 784
n_hidden = 700  # hidden layer n neurons
n_hidden2 = 700
n_classes = 10  # digits 0-9

h1 = tf.Variable(tf.random_normal([n_input, n_hidden]), dtype=tf.float32, name='hidden1')
h2 = tf.Variable(tf.random_normal([n_hidden, n_hidden2]), dtype=tf.float32, name='hidden2')
out = tf.Variable(tf.random_normal([n_hidden2, n_classes]), dtype=tf.float32, name='output')

b1 = tf.Variable(tf.ones([n_hidden]), dtype=tf.float32, name='bias1')
b2 = tf.Variable(tf.ones([n_hidden2]), dtype=tf.float32, name='bias2')
b3 = tf.Variable(tf.ones([n_classes]), dtype=tf.float32, name='bias3')

r_h1 = tf.Variable(tf.random_normal([n_input, n_hidden]), dtype=tf.float32, name='hidden1')
r_h2 = tf.Variable(tf.random_normal([n_hidden, n_hidden2]), dtype=tf.float32, name='hidden2')
r_out = tf.Variable(tf.random_normal([n_hidden2, n_classes]), dtype=tf.float32, name='output')

r_b1 = tf.Variable(tf.ones([n_hidden]), dtype=tf.float32, name='bias1')
r_b2 = tf.Variable(tf.ones([n_hidden2]), dtype=tf.float32, name='bias2')
r_b3 = tf.Variable(tf.ones([n_classes]), dtype=tf.float32, name='bias3')


def mlp_config(n_input, n_classes):
    x = tf.placeholder("float", [None, n_input], name='x')
    y = tf.placeholder("float", [None, n_classes], name='y')

    return x, y


def mlp_model(x, h1, h2, out,b1,b2,b3):
    hidden = tf.nn.relu(tf.matmul(x, h1)+b1)
    hidden2 = tf.nn.relu(tf.matmul(hidden, h2)+b2)
    logits = tf.add(tf.matmul(hidden2, out) , b3,name='prediction')
    # pred   = tf.one_hot(tf.cast(tf.argmax(logits, 1), tf.int32), depth=10)
    return logits


def get_loss(logits, y):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
    return loss


def get_accuracy(pred, y):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def optimizer(loss):
    return tf.train.GradientDescentOptimizer(learning_rate=.01).minimize(loss)

def to_pb():
    saver = tf.train.import_meta_graph('./models2/try.ckpt.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    sess = tf.Session()
    saver.restore(sess, "./models2/try.ckpt")

    output_node_names = "x,prediction"
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,  # The session
        input_graph_def,  # input_graph_def is useful for retrieving the nodes
        output_node_names.split(",")
    )

    output_graph = "./models2/try.pb"
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    sess.close()


def main():

    with tf.Session() as sess:
        saver = tf.train.Saver()
        chk = tf.train.checkpoint_exists("./models2/try.ckpt")
        print('Restore able ? ', chk)
        if chk == True:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, "models2/try.ckpt")
            r_h1, r_h2, r_out, r_b1, r_b2, r_b3 = sess.run([h1, h2, out, b1, b2, b3])
            print('PROBABLY Restored')
        else:
            sess.run(tf.global_variables_initializer())

        r_h1 = h1
        r_h2 = h2
        r_out = out
        r_b1 = b1
        r_b2 = b2
        r_b3 = b3

        x, y = mlp_config(n_input, n_classes)
        logits = mlp_model(x, r_h1, r_h2, r_out, r_b1, r_b2, r_b3)
        loss = get_loss(logits, y)
        accuracy = get_accuracy(logits, y)

        train_step = optimizer(loss)

        for i in range(100):
            batch = mnist.train.next_batch(100)  # fetch batch of size 1000
            acc = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1]})
            print('test accuracy at step %s: %s' % (i, acc))

            sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

        saver.save(sess, 'models3/my-weights')
        g = sess.graph
        gdef = g.as_graph_def()
        tf.train.write_graph(gdef, "tmp1", "graph.pb", False)

        # sv = saver.save(sess, "models2/try.ckpt")
        # print('Saved in Location ', sv)
        print("Accuracy using tensorflow is: ")
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

        #to_pb()


main()