import tensorflow as tf
import numpy as np
from model import Model


def create_model(session, feature_shape, words_number, batch_size, max_time, train_dir, train=True):
    model = Model(feature_shape=feature_shape,
                      words_number=words_number,
                      batch_size=batch_size,
                      max_time=max_time)
    ckpt = tf.train.get_checkpoint_state(train_dir)
    saver = tf.train.Saver(tf.global_variables())
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s\n" % ckpt.model_checkpoint_path)
        saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.\n")
        session.run(tf.global_variables_initializer())
    return model, saver

def train(g, save_path=None):
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        model, saver = create_model(session, g.feature_shape, g.words_number, g.batch_size, g.max_sentence_length, save_path)
        for epoch in range(2):
            print("epoch : " + str(epoch + 1))
            state = None
            for i in range(0, g.sample_number, g.batch_size):
                data, length, feature = g.next()
                feed_dict = {model.input: data,
                             model.length: length,
                             model.video_feature_pool: feature}
                if state is not None:
                    feed_dict[model.init_state]= state
                fetch_dict = {"loss": model.ppl, "state":model.final_state}
                val = session.run(fetch_dict, feed_dict)
                print(val['loss'])
        saver.save(session, save_path+"/model")

def generation(g, save_path, videos, idx_to_word):
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        model, _ = create_model(session, g.feature_shape, g.words_number,1,1, save_path)
        state = None
        # TODO: Beam Search
        for video in videos:
            video_feature = np.load('features' + '/' + video + '.npy')
            video_feature = [video_feature]
            current_word = 0
            caption = [current_word]
            for i in range(20):
                input = np.zeros([1, 1, g.words_number])
                input[0][0][current_word] = 1
                length = np.array([1])
                feed_dict = {model.input: input,
                             model.length: length,
                             model.video_feature_pool: np.stack(video_feature)}
                if state is not None:
                    feed_dict[model.init_state] = state
                fetch_dict = {'prob':model.next_words,
                              'state':model.final_state}
                vals = session.run(fetch_dict, feed_dict)
                p = np.squeeze(vals['prob'])
                current_word = np.argmax(p)
                caption.append(current_word)
            caption = map(lambda x: idx_to_word[x], caption)
            print(" ".join(caption))