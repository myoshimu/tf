import tensorflow as tf

#    c1 c2 c3 c4 c5 c6 c7
X = [[2.5, 3.0, 2.5,  -1, 3.0, 3.0,  -1], #'fabric'
    [ 3.5, 3.5, 3.0, 3.5, 4.0, 4.0, 4.5], #'cutlery'
    [ 3.0, 1.5,  -1, 3.0, 2.0,  -1,  -1], #'curtain'
    [ 3.5, 5.0, 3.5, 4.0, 3.0, 5.0, 4.0], #'stationary'
    [ 2.5, 3.5,  -1, 2.5, 2.0, 3.5, 1.0], #'table'
    [ 3.0, 3.0, 4.0, 4.5, 3.0, 3.0,  -1]] #'chair'
Z = [[  1,   1,   1,   0,   1,   1,  0], #'fabric'
    [   1,   1,   1,   1,   1,   1,  1], #'cutlery'
    [   1,   1,   0,   1,   1,   0,  0], #'curtain'
    [   1,   1,   1,   1,   1,   1,  1], #'stationary'
    [   1,   1,   0,   1,   1,   1,  1], #'table'
    [   1,   1,   1,   1,   1,   1,  0]] #'chair'

x = tf.placeholder(tf.float32, [6, 7])
y = tf.placeholder(tf.float32, [6, 7])

w1 = tf.Variable(tf.random_normal([6, 2], mean = 0.0, stddev = 0.05))
w2 = tf.Variable(tf.random_normal([7, 2], mean = 0.0, stddev = 0.05))
result = tf.matmul(w1, tf.transpose(w2))

loss = tf.square(tf.multiply(tf.matmul(w1, tf.transpose(w2)) - x, y)) + (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)) * 0.01
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(10000):
    sess.run(train_step, feed_dict = {
        x: X,
        y: Z
    })
    if i % 100 == 0:
        print (sess.run(loss, feed_dict = {
            x: X,
            y: Z
        }))

expected_evals = sess.run(result, feed_dict = {
    x: X,
    y: Z
})
for eval in expected_evals:
    print("%s" % eval)

