# models.py

import tensorflow as tf
import numpy as np
import random
from sentiment_data import *


# Returns a new numpy array with the data from np_arr padded to be of length length. If length is less than the
# length of the base array, truncates instead.
def pad_to_length(np_arr, length, constant_value=0):
    result = constant_value * np.ones(length)
    result[0:np_arr.shape[0]] = np_arr
    return result

def pre_pad_to_length(np_arr, length, constant_value=0):
    '''
    we actually should pad the index with "UNK" i.e. zero embedding
    '''
    result = constant_value * np.ones(length)
    result[-np_arr.shape[0]:] = np_arr
    return result

# Train a feedforward neural network on the given training examples, using dev_exs for development and returning
# predictions on the *blind* test_exs (all test_exs have label 0 as a dummy placeholder value). Returned predictions
# should be SentimentExample objects with predicted labels and the same sentences as input (but these won't be
# read for evaluation anyway)
def train_ffnn(train_exs, dev_exs, test_exs, word_vectors):
    '''
    The feed forward neural network takes the average of the word embeddings in each sentence, 
    and then go through two layers of fully connected layer.
    So each sentence should be represented as a list of vectors or a matrix, then take average.
    
    Input Params
    train_exs: training examples, list of SentimentExample, each SentimentExample contains indexed_words,
                a list of indices and a label
    dev_exs: similar
    test_exs: similar
    word_vectors: WordEmbeddings object, contains word_indexer and a list of word vectors, can fetch 
                word vectors through self.get_embedding(word)                
    '''
    vocab_size = word_vectors.vectors.shape[0]
    print('vocabulary size = %d'%vocab_size)
    
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    
    # feature vector size is the same as the word embedding size
    feat_vec_size = word_vectors.get_embedding("UNK").shape[0]
    
    # hidden vector size
    hidden_vec_size = min(feat_vec_size, 100)
    

    
    num_train_samples = len(train_exs)
    num_dev_samples = len(dev_exs)
    num_test_samples = len(test_exs)
    
    num_classes = 2    
    
    print('hidden_vec_size = %d'%hidden_vec_size) 

    
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len, vocab_size-1) 
                            for ex in train_exs], dtype=np.int32)
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs], dtype=np.float32)
    
    # print('train sequence lengths = %d'%np.amin(train_seq_lens))
    #--------------Labels--------------
    train_labels_arr = np.array([ex.label for ex in train_exs], dtype=np.int32)
    
    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len, vocab_size-1) 
                          for ex in dev_exs], dtype=np.int32)
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs], dtype=np.float32)
    dev_labels_arr = np.array([ex.label for ex in dev_exs], dtype=np.int32)
    
    test_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len, vocab_size-1) 
                           for ex in test_exs], dtype=np.int32)
    test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs], dtype=np.float32)
    
    # So we basically, need to convert training examples into tensors, get batches from it feed to 
    # the neural network for training. Then we need to check the accuracy on the development set.
    # If the performance is good, we would also get the prediction on the test set.
    
    # Convert the training examples into two tensors, a tensor of list of index, a tensor of labels
    # train_sentences_tensor = tf.convert_to_tensor(train_mat, dtype=tf.int32)
    # train_labels_tensor = tf.convert_to_tensor(train_labels_arr, dtype=tf.int32)
    
    # --------------Build graph--------------
    word_embeddings = tf.convert_to_tensor(word_vectors.vectors, dtype=tf.float32)
    
    input_word_indices = tf.placeholder(tf.int32, shape=[None, seq_max_len])
    
    input_word_vectors = tf.nn.embedding_lookup(word_embeddings, input_word_indices)
    
    input_seq_lens = tf.placeholder(tf.float32, shape=[None])

    # size of x_sum [None, feat_vec_size]
    x_sum = tf.reduce_sum(input_word_vectors, axis=1)
    
    # size of x_sum [None, feat_vec_size]
    x = tf.pad(tf.divide(x_sum, tf.expand_dims(input_seq_lens, axis=1)), [[0,0],[0,1]], constant_values=1.0)
    
    weights_in_1 = tf.get_variable("w_in_1", [feat_vec_size+1, hidden_vec_size],
                                   initializer=tf.contrib.layers.xavier_initializer(seed=0))
    
    z = tf.pad(tf.sigmoid(tf.tensordot(x, weights_in_1, 1)), [[0,0],[0,1]], constant_values=1.0)
    
    weights_in_2 = tf.get_variable("w_in_2", [hidden_vec_size+1, num_classes])
    
    h = tf.tensordot(z, weights_in_2, 1)
    
    probs = tf.nn.softmax(h)
    
    one_best = tf.argmax(probs, axis=1)
    
    label = tf.placeholder(tf.int32, shape = [None])
    
    label_onehot = tf.one_hot(label, num_classes, axis=-1)
    loss = tf.negative(tf.reduce_sum(tf.log(tf.tensordot(probs, tf.transpose(label_onehot), 1))))
    
    # --------------TRAINING ALGORITHM CUSTOMIZATION--------------

    
    num_epochs = 10    
    
    mini_batch_size = 1
    
    validate_every_n_epoch = 1
        
    # Decay the learning rate by a factor of 0.99 every 10 gradient steps (for larger datasets you'll want a slower
    # weight decay schedule
    
    decay_steps = 1000
    learning_rate_decay_factor = 0.99
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    initial_learning_rate = 0.005
    
    print('num_epochs = %d'%num_epochs)
    print('mini_batch_size = %d'%mini_batch_size)
    print('initial_learning_rate = %.3f'%initial_learning_rate)
    print('initial_learning_rate = %.3f'%initial_learning_rate)
    print('learning_rate_decay_factor = %.2f'%learning_rate_decay_factor)
    print('decay_steps = %d'%decay_steps)    
    
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    # Logging with Tensorboard
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)
    
    #--------------OPTIMIZER--------------
    # Plug in any first-order method here! We'll use Adam, which works pretty well, but SGD with momentum, Adadelta,
    # and lots of other methods work well too
    opt = tf.train.AdamOptimizer(lr)
    # Loss is the thing that we're optimizing
    grads = opt.compute_gradients(loss)
    # Now that we have gradients, we operationalize them by defining an operator that actually applies them.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)    
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')
    
    # --------------RUN TRAINING AND TEST--------------
    # Initializer; we need to run this first to initialize variables
    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()  # merge all the tensorboard variables
    # The computation graph must be run in a particular Tensorflow "session". Parameters, etc. are localized to the
    # session (unless you pass them around outside it). All runs of a computation graph with certain values are relative
    # to a particular session
    with tf.Session() as sess:      
        # Write a logfile to the logs/ directory, can use Tensorboard to view this
        train_writer = tf.summary.FileWriter('logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in range(1, num_epochs+1):
            
            # --------------Training--------------
            
            loss_this_iter = 0.0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            permute = np.random.permutation(num_train_samples)
            for ex_idx in xrange(0, num_train_samples, mini_batch_size):
                if ex_idx + mini_batch_size > num_train_samples:
                    continue
                batch_indices = permute[ex_idx:ex_idx + mini_batch_size]
                
                train_xs_batch = train_mat[batch_indices,:]
                train_ys_batch = train_labels_arr[batch_indices]
                train_seq_lens_batch = train_seq_lens[batch_indices]
                
                # sess.run generally evaluates variables in the computation graph given inputs. "Evaluating" train_op
                # causes training to happen
                [_, loss_this_batch, summary] = sess.run([train_op, loss, merged], 
                                                         feed_dict = {input_word_indices: train_xs_batch,
                                                                      label: train_ys_batch,
                                                                      input_seq_lens: train_seq_lens_batch})
                train_writer.add_summary(summary, step_idx)
                step_idx += 1
                loss_this_iter += loss_this_batch
                # print "Loss for this batch " + repr(loss_this_batch)
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter/num_train_samples)
            
            # --------------Evaluate on the dev set--------------
            
            if i % validate_every_n_epoch == 0:
                dev_correct = 0
                dev_loss = 0.0
                for ex_idx in xrange(0, num_dev_samples, mini_batch_size):
                    next_ex_idx = ex_idx + mini_batch_size
                    next_ex_idx = next_ex_idx if next_ex_idx < num_dev_samples else num_dev_samples

                    # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
                    # quantities from the running of the computation graph, namely the probabilities, prediction, and z
                    [pred_this_batch, loss_this_batch] = sess.run([one_best, loss],
                                                                   feed_dict={input_word_indices:dev_mat[ex_idx:next_ex_idx,:],
                                                                              label:dev_labels_arr[ex_idx:next_ex_idx],
                                                                              input_seq_lens:dev_seq_lens[ex_idx:next_ex_idx]})
                    dev_correct += np.sum(pred_this_batch == dev_labels_arr[ex_idx:next_ex_idx])
                    dev_loss += loss_this_batch
                    #print "Example " + repr(train_xs[ex_idx]) + "; gold = " + repr(train_ys[ex_idx]) + "; pred = " +\
                    #      repr(pred_this_instance) + " with probs " + repr(probs_this_instance)
                    # print "  Hidden layer activations for this example: " + repr(z_this_instance)
                print repr(dev_correct) + "/" + repr(num_dev_samples) + " correct after training"
                print "Loss for dev " + repr(dev_loss/num_dev_samples)
        
        # --------------Evaluate on the test set--------------
        
        test_pred = np.empty(num_test_samples, dtype=np.int32)
        for ex_idx in xrange(0, num_test_samples, mini_batch_size):
            next_ex_idx = ex_idx + mini_batch_size
            next_ex_idx = next_ex_idx if next_ex_idx < num_test_samples else num_test_samples

            # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
            # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            [pred_this_batch] = sess.run([one_best],
                                         feed_dict={input_word_indices:test_mat[ex_idx:next_ex_idx,:],
                                                    input_seq_lens:test_seq_lens[ex_idx:next_ex_idx]})
            test_pred[ex_idx:next_ex_idx] = pred_this_batch
            
    for i, ex in enumerate(test_exs):
        ex.label = test_pred[i]
    return test_exs
    # raise Exception("Not implemented")


# Analogous to train_ffnn, but trains your fancier model.
def train_fancy(train_exs, dev_exs, test_exs, word_vectors):
    return train_brnn(train_exs, dev_exs, test_exs, word_vectors)
    #return train_rnn(train_exs, dev_exs, test_exs, word_vectors)
    
def train_brnn(train_exs, dev_exs, test_exs, word_vectors):
    '''
    The recurrent neural network basically recurrently goes through a the list of vectors and 
    get an output. For bidirectional RNN, we can have two recurrent neural networks, one goes
    forward and the other goes backwards, we get two outputs.
    The the concatenation of the outputs becomes the input of the second layer.
    So each sentence should be represented as a list of vectors or a matrix.
    
    Input Params
    train_exs: training examples, list of SentimentExample, each SentimentExample contains indexed_words,
                a list of indices and a label
    dev_exs: similar
    test_exs: similar
    word_vectors: WordEmbeddings object, contains word_indexer and a list of word vectors, can fetch 
                word vectors through self.get_embedding(word)                
    '''
    # vocabulary size
    vocab_size = word_vectors.vectors.shape[0]
    
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    
    # feature vector size is the same as the word embedding size
    feat_vec_size = word_vectors.get_embedding("UNK").shape[0]
    
    # hidden vector size
    hidden_vec_size = min(feat_vec_size, 100)    
    
    
    num_train_samples = len(train_exs)
    num_dev_samples = len(dev_exs)
    num_test_samples = len(test_exs)
    
    num_classes = 2
    
    
    print('hidden_vec_size = %d'%hidden_vec_size)  
    
    
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    # To get the proper output, we probably should do prepadding and pad with the UNK idx
    train_mat_fwd = np.asarray([pre_pad_to_length(np.array(ex.indexed_words), seq_max_len, vocab_size-1) 
                                for ex in train_exs], dtype=np.int32)
    train_mat_bwd = np.asarray([pre_pad_to_length(np.array(ex.get_indexed_words_reversed()), seq_max_len, vocab_size-1) 
                                for ex in train_exs], dtype=np.int32)
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs], dtype=np.float32)
    
    print(np.amin(train_seq_lens))
    
    # --------------Labels--------------
    
    train_labels_arr = np.array([ex.label for ex in train_exs], dtype=np.int32)
    
    dev_mat_fwd = np.asarray([pre_pad_to_length(np.array(ex.indexed_words), seq_max_len, vocab_size-1) 
                              for ex in dev_exs], dtype=np.int32)
    dev_mat_bwd = np.asarray([pre_pad_to_length(np.array(ex.get_indexed_words_reversed()), seq_max_len, vocab_size-1) 
                              for ex in dev_exs], dtype=np.int32)
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs], dtype=np.float32)
    dev_labels_arr = np.array([ex.label for ex in dev_exs], dtype=np.int32)
    
    test_mat_fwd = np.asarray([pre_pad_to_length(np.array(ex.indexed_words), seq_max_len, vocab_size-1) 
                               for ex in test_exs], dtype=np.int32)
    test_mat_bwd = np.asarray([pre_pad_to_length(np.array(ex.get_indexed_words_reversed()), seq_max_len, vocab_size-1) 
                               for ex in test_exs], dtype=np.int32)
    test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs], dtype=np.float32)
    
    # So we basically, need to convert training examples into tensors, get batches from it feed to 
    # the neural network for training. Then we need to check the accuracy on the development set.
    # If the performance is good, we would also get the prediction on the test set.
    
    # Convert the training examples into two tensors, a tensor of list of index, a tensor of labels
    # train_sentences_tensor = tf.convert_to_tensor(train_mat, dtype=tf.int32)
    # train_labels_tensor = tf.convert_to_tensor(train_labels_arr, dtype=tf.int32)
    # Build graph
    
    # --------------Build graph--------------
    
    word_embeddings = tf.convert_to_tensor(word_vectors.vectors, dtype=tf.float32)
    
    input_word_indices_fwd = tf.placeholder(tf.int32, shape=[None, seq_max_len])
    
    input_word_indices_bwd = tf.placeholder(tf.int32, shape=[None, seq_max_len])
    
    # size of input_word_vectors is [batch_size, seq_max_len, feat_vec_size]
    # batch_size is unknown, so set as None
    input_word_vectors_fwd = tf.nn.embedding_lookup(word_embeddings, input_word_indices_fwd)
    
    input_word_vectors_bwd = tf.nn.embedding_lookup(word_embeddings, input_word_indices_bwd)
    
    input_seq_lens = tf.placeholder(tf.float32, shape=[None])
    
    # unstack tensor to get a list of tensors of shape: [batch_size, feat_vec_size]
    x_fwd = tf.unstack(input_word_vectors_fwd, axis=1)
    x_bwd = tf.unstack(input_word_vectors_bwd, axis=1)
    
    #lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_vec_size, forget_bias=1.0)
    
    # outputs should now be of the shape [max_seq_lens, batch_size, hidden_vec_size]
    #outputs_fwd, states_fwd = tf.contrib.rnn.static_rnn(lstm_cell, x_fwd, dtype=tf.float32)
    
    #outputs_bwd, states_bwd = tf.contrib.rnn.static_rnn(lstm_cell, x_bwd, dtype=tf.float32)
    with tf.variable_scope('lstm1'):
        lstm_cell_fwd = tf.contrib.rnn.BasicLSTMCell(hidden_vec_size, forget_bias=1.0)
        outputs_fwd, states_fwd = tf.contrib.rnn.static_rnn(lstm_cell_fwd, x_fwd, dtype=tf.float32)
    with tf.variable_scope('lstm2'):
        lstm_cell_bwd = tf.contrib.rnn.BasicLSTMCell(hidden_vec_size, forget_bias=1.0)    
        outputs_bwd, states_bwd = tf.contrib.rnn.static_rnn(lstm_cell_bwd, x_bwd, dtype=tf.float32)
    
    
    
    # now we just use the last output as the input of next layter
    # final_outputs = outputs[]
    
    z = tf.pad(tf.concat([outputs_fwd[-1], outputs_bwd[-1]], axis=1),
               [[0,0],[0,1]], constant_values=1.0)
    
    # need to modifiy the shape if use both directions
    weights_in_2 = tf.get_variable("w_in_2", [2*hidden_vec_size+1, num_classes])

    h = tf.tensordot(z, weights_in_2, 1)
    
    probs = tf.nn.softmax(h)
    
    one_best = tf.argmax(probs, axis=1)
    
    label = tf.placeholder(tf.int32, shape = [None])
    
    label_onehot = tf.one_hot(label, num_classes, axis=-1)
    loss = tf.negative(tf.reduce_sum(tf.log(tf.tensordot(probs, tf.transpose(label_onehot), 1))))
    
    # --------------TRAINING ALGORITHM CUSTOMIZATION--------------
    num_epochs = 3    
    
    mini_batch_size = 1
    
    validate_every_n_epoch = 1
    
    # Decay the learning rate by a factor of 0.99 every 10 gradient steps (for larger datasets you'll want a slower
    # weight decay schedule
    
    decay_steps = 1000
    learning_rate_decay_factor = 0.99
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    initial_learning_rate = 0.001
    
    print('num_epochs = %d'%num_epochs)
    print('mini_batch_size = %d'%mini_batch_size)
    print('initial_learning_rate = %.3f'%initial_learning_rate)
    print('learning_rate_decay_factor = %.2f'%learning_rate_decay_factor)
    print('decay_steps = %d'%decay_steps)
    
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    # Logging with Tensorboard
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)
    
    #--------------OPTIMIZER--------------
    
    # Plug in any first-order method here! We'll use Adam, which works pretty well, but SGD with momentum, Adadelta,
    # and lots of other methods work well too
    opt = tf.train.AdamOptimizer(lr)
    # Loss is the thing that we're optimizing
    grads = opt.compute_gradients(loss)
    # Now that we have gradients, we operationalize them by defining an operator that actually applies them.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)    
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')
    
        # RUN TRAINING AND TEST
    # Initializer; we need to run this first to initialize variables
    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()  # merge all the tensorboard variables
    # The computation graph must be run in a particular Tensorflow "session". Parameters, etc. are localized to the
    # session (unless you pass them around outside it). All runs of a computation graph with certain values are relative
    # to a particular session
    with tf.Session() as sess:
        # Write a logfile to the logs/ directory, can use Tensorboard to view this
        train_writer = tf.summary.FileWriter('logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in range(1, num_epochs+1):
            
            # --------------Training--------------
            
            loss_this_iter = 0.0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            permute = np.random.permutation(num_train_samples)
            for ex_idx in xrange(0, num_train_samples, mini_batch_size):
                if ex_idx + mini_batch_size > num_train_samples:
                    continue
                batch_indices = permute[ex_idx:ex_idx+mini_batch_size]
                
                train_xs_fwd_batch = train_mat_fwd[batch_indices,:]
                train_xs_bwd_batch = train_mat_bwd[batch_indices,:]
                train_ys_batch = train_labels_arr[batch_indices]
                train_seq_lens_batch = train_seq_lens[batch_indices]
                
                # sess.run generally evaluates variables in the computation graph given inputs. "Evaluating" train_op
                # causes training to happen
                [_, loss_this_batch, summary] = sess.run([train_op, loss, merged], 
                                                         feed_dict = {input_word_indices_fwd: train_xs_fwd_batch,
                                                                      input_word_indices_bwd: train_xs_bwd_batch,
                                                                      label: train_ys_batch,
                                                                      input_seq_lens: train_seq_lens_batch})
                train_writer.add_summary(summary, step_idx)
                step_idx += 1
                loss_this_iter += loss_this_batch
                # print "Loss for this batch " + repr(loss_this_batch)
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter/num_train_samples)
            
            # --------------Evaluate on the dev set--------------
            
            if i % validate_every_n_epoch == 0:
                dev_correct = 0
                dev_loss = 0.0
                for ex_idx in xrange(0, num_dev_samples, mini_batch_size):
                    next_ex_idx = ex_idx + mini_batch_size
                    next_ex_idx = next_ex_idx if next_ex_idx < num_dev_samples else num_dev_samples

                    # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
                    # quantities from the running of the computation graph, namely the probabilities, prediction, and z
                    [pred_this_batch, loss_this_batch] = sess.run([one_best, loss],
                                                           feed_dict={input_word_indices_fwd:dev_mat_fwd[ex_idx:next_ex_idx,:],
                                                                      input_word_indices_bwd:dev_mat_bwd[ex_idx:next_ex_idx,:],
                                                                      label:dev_labels_arr[ex_idx:next_ex_idx],
                                                                      input_seq_lens:dev_seq_lens[ex_idx:next_ex_idx]})
                    dev_correct += np.sum(pred_this_batch == dev_labels_arr[ex_idx:next_ex_idx])
                    dev_loss += loss_this_batch
                    #print "Example " + repr(train_xs[ex_idx]) + "; gold = " + repr(train_ys[ex_idx]) + "; pred = " +\
                    #      repr(pred_this_instance) + " with probs " + repr(probs_this_instance)
                    # print "  Hidden layer activations for this example: " + repr(z_this_instance)1
                print repr(dev_correct) + "/" + repr(num_dev_samples) + " correct after training"
                print "Loss for dev " + repr(dev_loss/num_dev_samples)
                
        # --------------Evaluate on the test set--------------
        
        test_pred = np.empty(num_test_samples, dtype=np.int32)
        for ex_idx in xrange(0, num_test_samples, mini_batch_size):
            next_ex_idx = ex_idx + mini_batch_size
            next_ex_idx = next_ex_idx if next_ex_idx < num_test_samples else num_test_samples

            # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
            # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            [pred_this_batch] = sess.run([one_best],
                                         feed_dict={input_word_indices_fwd:test_mat_fwd[ex_idx:next_ex_idx,:],
                                                    input_word_indices_bwd:test_mat_bwd[ex_idx:next_ex_idx,:],
                                                    input_seq_lens:test_seq_lens[ex_idx:next_ex_idx]})
            test_pred[ex_idx:next_ex_idx] = pred_this_batch
    for i, ex in enumerate(test_exs):
        ex.label = test_pred[i]
    return test_exs
    

def train_rnn(train_exs, dev_exs, test_exs, word_vectors):
    '''
    The recurrent neural network basically recurrently goes through a the list of vectors and 
    get an output. 
    
    Input Params
    train_exs: training examples, list of SentimentExample, each SentimentExample contains indexed_words,
                a list of indices and a label
    dev_exs: similar
    test_exs: similar
    word_vectors: WordEmbeddings object, contains word_indexer and a list of word vectors, can fetch 
                word vectors through self.get_embedding(word)                
    '''
    # vocabulary size
    vocab_size = word_vectors.vectors.shape[0]
    
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    
    # feature vector size is the same as the word embedding size
    feat_vec_size = word_vectors.get_embedding("UNK").shape[0]
    
    # hidden vector size
    hidden_vec_size = min(feat_vec_size, 100)    
    
    
    num_train_samples = len(train_exs)
    num_dev_samples = len(dev_exs)
    num_test_samples = len(test_exs)
    
    num_classes = 2
    
    
    print('hidden_vec_size = %d'%hidden_vec_size)  
    
    
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    # To get the proper output, we probably should do prepadding and pad with the UNK idx
    train_mat_fwd = np.asarray([pre_pad_to_length(np.array(ex.indexed_words), seq_max_len, vocab_size-1) 
                                for ex in train_exs], dtype=np.int32)
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs], dtype=np.float32)
    
    print(np.amin(train_seq_lens))
    
    # --------------Labels--------------
    
    train_labels_arr = np.array([ex.label for ex in train_exs], dtype=np.int32)
    
    dev_mat_fwd = np.asarray([pre_pad_to_length(np.array(ex.indexed_words), seq_max_len, vocab_size-1) 
                              for ex in dev_exs], dtype=np.int32)

    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs], dtype=np.float32)
    dev_labels_arr = np.array([ex.label for ex in dev_exs], dtype=np.int32)
    
    test_mat_fwd = np.asarray([pre_pad_to_length(np.array(ex.indexed_words), seq_max_len, vocab_size-1) 
                               for ex in test_exs], dtype=np.int32)

    test_seq_lens = np.array([len(ex.indexed_words) for ex in test_exs], dtype=np.float32)
    
    # So we basically, need to convert training examples into tensors, get batches from it feed to 
    # the neural network for training. Then we need to check the accuracy on the development set.
    # If the performance is good, we would also get the prediction on the test set.
    
    # Convert the training examples into two tensors, a tensor of list of index, a tensor of labels
    # train_sentences_tensor = tf.convert_to_tensor(train_mat, dtype=tf.int32)
    # train_labels_tensor = tf.convert_to_tensor(train_labels_arr, dtype=tf.int32)
    # Build graph
    
    # --------------Build graph--------------
    
    word_embeddings = tf.convert_to_tensor(word_vectors.vectors, dtype=tf.float32)
    
    input_word_indices_fwd = tf.placeholder(tf.int32, shape=[None, seq_max_len])
    
    
    # size of input_word_vectors is [batch_size, seq_max_len, feat_vec_size]
    # batch_size is unknown, so set as None
    input_word_vectors_fwd = tf.nn.embedding_lookup(word_embeddings, input_word_indices_fwd)
        
    input_seq_lens = tf.placeholder(tf.float32, shape=[None])
    
    # unstack tensor to get a list of tensors of shape: [batch_size, feat_vec_size]
    x_fwd = tf.unstack(input_word_vectors_fwd, axis=1)
    
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_vec_size, forget_bias=1.0)
    
    # outputs should now be of the shape [max_seq_lens, batch_size, hidden_vec_size]
    outputs_fwd, states_fwd = tf.contrib.rnn.static_rnn(lstm_cell, x_fwd, dtype=tf.float32)
    
    # now we just use the last output as the input of next layter
    # final_outputs = outputs[]
    
    z = tf.pad(outputs_fwd[-1], [[0,0],[0,1]], constant_values=1.0)    
    weights_in_2 = tf.get_variable("w_in_2", [hidden_vec_size+1, num_classes]) 

    h = tf.tensordot(z, weights_in_2, 1)
    
    probs = tf.nn.softmax(h)
    
    one_best = tf.argmax(probs, axis=1)
    
    label = tf.placeholder(tf.int32, shape = [None])
    
    label_onehot = tf.one_hot(label, num_classes, axis=-1)
    loss = tf.negative(tf.reduce_sum(tf.log(tf.tensordot(probs, tf.transpose(label_onehot), 1))))
    
    # --------------TRAINING ALGORITHM CUSTOMIZATION--------------
    num_epochs = 3   
    
    mini_batch_size = 1
    
    validate_every_n_epoch = 1
    
    # Decay the learning rate by a factor of 0.99 every 10 gradient steps (for larger datasets you'll want a slower
    # weight decay schedule
    
    decay_steps = 1000
    learning_rate_decay_factor = 0.99
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    initial_learning_rate = 0.001
    
    print('num_epochs = %d'%num_epochs)
    print('mini_batch_size = %d'%mini_batch_size)
    print('initial_learning_rate = %.3f'%initial_learning_rate)
    print('learning_rate_decay_factor = %.2f'%learning_rate_decay_factor)
    print('decay_steps = %d'%decay_steps)
    
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    # Logging with Tensorboard
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)
    
    #--------------OPTIMIZER--------------
    
    # Plug in any first-order method here! We'll use Adam, which works pretty well, but SGD with momentum, Adadelta,
    # and lots of other methods work well too
    opt = tf.train.AdamOptimizer(lr)
    # Loss is the thing that we're optimizing
    grads = opt.compute_gradients(loss)
    # Now that we have gradients, we operationalize them by defining an operator that actually applies them.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)    
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')
    
        # RUN TRAINING AND TEST
    # Initializer; we need to run this first to initialize variables
    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()  # merge all the tensorboard variables
    # The computation graph must be run in a particular Tensorflow "session". Parameters, etc. are localized to the
    # session (unless you pass them around outside it). All runs of a computation graph with certain values are relative
    # to a particular session
    with tf.Session() as sess:
        # Write a logfile to the logs/ directory, can use Tensorboard to view this
        train_writer = tf.summary.FileWriter('logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in range(1, num_epochs+1):
            
            # --------------Training--------------
            
            loss_this_iter = 0.0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            permute = np.random.permutation(num_train_samples)
            for ex_idx in xrange(0, num_train_samples, mini_batch_size):
                if ex_idx + mini_batch_size > num_train_samples:
                    continue
                batch_indices = permute[ex_idx:ex_idx+mini_batch_size]
                
                train_xs_fwd_batch = train_mat_fwd[batch_indices,:]
                train_ys_batch = train_labels_arr[batch_indices]
                train_seq_lens_batch = train_seq_lens[batch_indices]
                
                # sess.run generally evaluates variables in the computation graph given inputs. "Evaluating" train_op
                # causes training to happen
                [_, loss_this_batch, summary] = sess.run([train_op, loss, merged], 
                                                         feed_dict = {input_word_indices_fwd: train_xs_fwd_batch,
                                                                      label: train_ys_batch,
                                                                      input_seq_lens: train_seq_lens_batch})
                train_writer.add_summary(summary, step_idx)
                step_idx += 1
                loss_this_iter += loss_this_batch
                # print "Loss for this batch " + repr(loss_this_batch)
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter/num_train_samples)
            
            # --------------Evaluate on the dev set--------------
            
            if i % validate_every_n_epoch == 0:
                dev_correct = 0
                dev_loss = 0.0
                for ex_idx in xrange(0, num_dev_samples, mini_batch_size):
                    next_ex_idx = ex_idx + mini_batch_size
                    next_ex_idx = next_ex_idx if next_ex_idx < num_dev_samples else num_dev_samples

                    # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
                    # quantities from the running of the computation graph, namely the probabilities, prediction, and z
                    [pred_this_batch, loss_this_batch] = sess.run([one_best, loss],
                                                           feed_dict={input_word_indices_fwd:dev_mat_fwd[ex_idx:next_ex_idx,:],
                                                                      label:dev_labels_arr[ex_idx:next_ex_idx],
                                                                      input_seq_lens:dev_seq_lens[ex_idx:next_ex_idx]})
                    dev_correct += np.sum(pred_this_batch == dev_labels_arr[ex_idx:next_ex_idx])
                    dev_loss += loss_this_batch
                    #print "Example " + repr(train_xs[ex_idx]) + "; gold = " + repr(train_ys[ex_idx]) + "; pred = " +\
                    #      repr(pred_this_instance) + " with probs " + repr(probs_this_instance)
                    # print "  Hidden layer activations for this example: " + repr(z_this_instance)1
                print repr(dev_correct) + "/" + repr(num_dev_samples) + " correct after training"
                print "Loss for dev " + repr(dev_loss/num_dev_samples)
                
        # --------------Evaluate on the test set--------------
        
        test_pred = np.empty(num_test_samples, dtype=np.int32)
        for ex_idx in xrange(0, num_test_samples, mini_batch_size):
            next_ex_idx = ex_idx + mini_batch_size
            next_ex_idx = next_ex_idx if next_ex_idx < num_test_samples else num_test_samples

            # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
            # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            [pred_this_batch] = sess.run([one_best],
                                         feed_dict={input_word_indices_fwd:test_mat_fwd[ex_idx:next_ex_idx,:],
                                                    input_seq_lens:test_seq_lens[ex_idx:next_ex_idx]})
            test_pred[ex_idx:next_ex_idx] = pred_this_batch
    for i, ex in enumerate(test_exs):
        ex.label = test_pred[i]
    return test_exs
