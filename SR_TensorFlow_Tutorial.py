from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

import tensorflow as tf

learning_rate = 0.008
training_iteration = 30
batch_size = 100
display_step = 2

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None,10])

w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

w_h = tf.summary.histogram("weights", w)
b_h = tf.summary.histogram("biases",b)

with tf.name_scope("wx_b") as scope:
    model = tf.nn.softmax(tf.matmul(x,w) + b)

with tf.name_scope("Cost_Function") as scope:
    cost_function = -tf.reduce_sum(y*tf.log(model))
    tf.summary.scalar("Cost Function", cost_function)

with tf.name_scope("train") as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

init = tf.initialize_all_variables()

merge_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)

    summary_writer = tf.summary.FileWriter("/home/tejan/PycharmProjects/Machine_Learning_Practice/venv/logs"
                                           ,graph_def=sess.graph_def)

    for iteration in range(training_iteration):
        avg_cost = 0.0
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            sess.run(optimizer,feed_dict={x:batch_x, y:batch_y})

            avg_cost += sess.run(cost_function, feed_dict={x:batch_x, y:batch_y})/total_batch

            summary_str = sess.run(merge_summary_op, feed_dict={x:batch_x, y:batch_y})
            summary_writer.add_summary(summary_str,iteration*total_batch*i)

        if iteration % display_step == 0:
            print("Iterations", '%04d' % (iteration + 1), "cost=", '{:.9f}'.format(avg_cost))

    print("Tuning Complete")

    prediction = tf.equal(tf.argmax(model,1),tf.argmax(y,1))

    accuracy = tf.reduce_mean(tf.cast(prediction,"float"))
    print("Accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))