import tensorflow as tf

logits_list = [tf.Variable([[7., 3., 0.]]), tf.Variable([[6.0, 0.0, 90.0]])]
labels = tf.Variable([[1, 1, 0]])
loss_list = [tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
             for logits in logits_list]

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(loss_list))
