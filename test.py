import tensorflow as tf

g = tf.Graph()
with g.as_default():
    w = tf.constant(2)
s = tf.Session(graph=g)
with s:
    print(s.graph)
    print(tf.get_default_graph())
    print(g)
    with g.as_default():
        print(s.graph)
        print(tf.get_default_graph())
        print(s.run(w))
s.close()
print(tf.get_default_session())
print(tf.get_default_session().run(w))
