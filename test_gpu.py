import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()

# if torch.cuda.is_available():
#     print('it works')
# else:
#     print('it does not work')