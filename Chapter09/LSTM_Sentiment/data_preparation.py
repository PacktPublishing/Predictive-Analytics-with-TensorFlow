# data_manager.py: Loads and preprocesses data
import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
import os
import html
import re
import string
import pickle

class Preprocessing(object):
    def __init__(self, data_dir, stopwords_file=None, sequence_len=None, n_samples=None, test_size=0.2, val_samples=100, random_state=0, ensure_preprocessed=False):
        """
        Initiallizes data manager. DataManager provides an interface to load, preprocess and split data into train,
        validation and test sets
        :param data_dir: Data directory containing the dataset file 'data.csv' with columns 'SentimentText' and
                         'Sentiment'
        :param stopwords_file: Optional. If provided, discards each stopword from original data
        :param sequence_len: Optional. Let m be the maximum sequence length in the dataset. Then, it's required that
                          sequence_len >= m. If sequence_len is None, then it'll be automatically assigned to m.
        :param n_samples: Optional. Number of samples to load from the dataset (useful for large datasets). If n_samples
                          is None, then the whole dataset will be loaded (be careful, if dataset is large it may take a
                          while to preprocess every sample)
        :param test_size: Optional. 0<test_size<1. Represents the proportion of the dataset to included in the test
                          split. Default is 0.2
        :param val_samples: Optional. Represents the absolute number of validations samples. Default is 100
        :param random_state: Optional. Random seed used for splitting data into train, test and validation sets. Default is 0.
        :param ensure_preprocessed: Optional. If ensure_preprocessed=True, ensures that the dataset is already
                          preprocessed. Default is False
        """
        self._stopwords_file = stopwords_file
        self._n_samples = n_samples
        self.sequence_len = sequence_len
        self._input_file = os.path.join(data_dir, 'data.csv')
        self._preprocessed_file = os.path.join(data_dir, "preprocessed_" + str(n_samples) + ".npz")
        self._vocab_file = os.path.join(data_dir, "vocab_" + str(n_samples) + ".pkl")
        self._tensors = None
        self._sentiments = None
        self._lengths = None
        self._vocab = None
        self.vocab_size = None

        # Prepare data
        if os.path.exists(self._preprocessed_file) and os.path.exists(self._vocab_file):
            print('Loading preprocessed files ...')
            self.__load_preprocessed()
        else:
            if ensure_preprocessed:
                raise ValueError('Unable to find preprocessed files.')
            print('Reading data ...')
            self.__preprocess()

        # Split data in train, validation and test sets
        indices = np.arange(len(self._sentiments))
        x_tv, self._x_test, y_tv, self._y_test, tv_indices, test_indices = train_test_split(
            self._tensors,
            self._sentiments,
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=self._sentiments[:, 0])
        self._x_train, self._x_val, self._y_train, self._y_val, train_indices, val_indices = train_test_split(
            x_tv,
            y_tv,
            tv_indices,
            test_size=val_samples,
            random_state=random_state,
            stratify=y_tv[:, 0])
        self._val_indices = val_indices
        self._test_indices = test_indices
        self._train_lengths = self._lengths[train_indices]
        self._val_lengths = self._lengths[val_indices]
        self._test_lengths = self._lengths[test_indices]
        self._current_index = 0
        self._epoch_completed = 0

    def __preprocess(self):
        """
        Loads data from data_dir/data.csv, preprocesses each sample loaded and stores intermediate files to avoid
        preprocessing later.
        """
        # Load data
        data = pd.read_csv(self._input_file, nrows=self._n_samples)
        self._sentiments = np.squeeze(data.as_matrix(columns=['Sentiment']))
        self._sentiments = np.eye(2)[self._sentiments]
        samples = data.as_matrix(columns=['SentimentText'])[:, 0]

        # Cleans samples text
        samples = self.__clean_samples(samples)

        # Prepare vocabulary dict
        vocab = dict()
        vocab[''] = (0, len(samples))  # add empty word
        for sample in samples:
            sample_words = sample.split()
            for word in list(set(sample_words)):  # distinct words in list
                value = vocab.get(word)
                if value is None:
                    vocab[word] = (-1, 1)
                else:
                    encoding, count = value
                    vocab[word] = (-1, count + 1)

        # Remove the most uncommon words (they're probably grammar mistakes), encode samples into tensors and
        # store samples' lengths
        sample_lengths = []
        tensors = []
        word_count = 1
        for sample in samples:
            sample_words = sample.split()
            encoded_sample = []
            for word in list(set(sample_words)):  # distinct words in list
                value = vocab.get(word)
                if value is not None:
                    encoding, count = value
                    if count / len(samples) > 0.0001:
                        if encoding == -1:
                            encoding = word_count
                            vocab[word] = (encoding, count)
                            word_count += 1
                        encoded_sample += [encoding]
                    else:
                        del vocab[word]
            tensors += [encoded_sample]
            sample_lengths += [len(encoded_sample)]

        self.vocab_size = len(vocab)
        self._vocab = vocab
        self._lengths = np.array(sample_lengths)

        # Pad each tensor with zeros according self.sequence_len
        self.sequence_len, self._tensors = self.__apply_to_zeros(tensors, self.sequence_len)

        # Save intermediate files
        with open(self._vocab_file, 'wb') as f:
            pickle.dump(self._vocab, f)
        np.savez(self._preprocessed_file, tensors=self._tensors, lengths=self._lengths, sentiments=self._sentiments)

    def __load_preprocessed(self):
        """
        Loads intermediate files, avoiding data preprocess
        """
        with open(self._vocab_file, 'rb') as f:
            self._vocab = pickle.load(f)
        self.vocab_size = len(self._vocab)
        load_dict = np.load(self._preprocessed_file)
        self._lengths = load_dict['lengths']
        self._tensors = load_dict['tensors']
        self._sentiments = load_dict['sentiments']
        self.sequence_len = len(self._tensors[0])

    def __clean_samples(self, samples):
        """
        Cleans samples.
        :param samples: Samples to be cleaned
        :return: cleaned samples
        """
        print('Cleaning samples ...')
        # Prepare regex patterns
        ret = []
        reg_punct = '[' + re.escape(''.join(string.punctuation)) + ']'
        if self._stopwords_file is not None:
            stopwords = self.__read_stopwords()
            sw_pattern = re.compile(r'\b(' + '|'.join(stopwords) + r')\b')

        # Clean each sample
        for sample in samples:
            # Restore HTML characters
            text = html.unescape(sample)

            # Remove @users and urls
            words = text.split()
            words = [word for word in words if not word.startswith('@') and not word.startswith('http://')]
            text = ' '.join(words)

            # Transform to lowercase
            text = text.lower()

            # Remove punctuation symbols
            text = re.sub(reg_punct, ' ', text)

            # Replace CC(C+) (a character occurring more than twice in a row) for C
            text = re.sub(r'([a-z])\1{2,}', r'\1', text)

            # Remove stopwords
            if stopwords is not None:
                text = sw_pattern.sub('', text)
            ret += [text]

        return ret

    def __apply_to_zeros(self, lst, sequence_len=None):
        """
        Pads lst with zeros according to sequence_len
        :param lst: List to be padded
        :param sequence_len: Optional. Let m be the maximum sequence length in lst. Then, it's required that
                          sequence_len >= m. If sequence_len is None, then it'll be automatically assigned to m.
        :return: padding_length used and numpy array of padded tensors.
        """
        # Find maximum length m and ensure that m>=sequence_len
        inner_max_len = max(map(len, lst))
        if sequence_len is not None:
            if inner_max_len > sequence_len:
                raise Exception('Error: Provided sequence length is not sufficient')
            else:
                inner_max_len = sequence_len

        # Pad list with zeros
        result = np.zeros([len(lst), inner_max_len], np.int32)
        for i, row in enumerate(lst):
            for j, val in enumerate(row):
                result[i][j] = val
        return inner_max_len, result

    def __read_stopwords(self):
        """
        :return: Stopwords list
        """
        if self._stopwords_file is None:
            return None
        with open(self._stopwords_file, mode='r') as f:
            stopwords = f.read().splitlines()
        return stopwords

    def next_batch(self, batch_size):
        """
        :param batch_size: batch_size>0. Number of samples that'll be included
        :return: Returns batch size samples (text_tensor, text_target, text_length)
        """
        start = self._current_index
        self._current_index += batch_size
        if self._current_index > len(self._y_train):
            # Complete epoch and randomly shuffle train samples
            self._epoch_completed += 1
            ind = np.arange(len(self._y_train))
            np.random.shuffle(ind)
            self._x_train = self._x_train[ind]
            self._y_train = self._y_train[ind]
            self._train_lengths = self._train_lengths[ind]
            start = 0
            self._current_index = batch_size
        end = self._current_index
        return self._x_train[start:end], self._y_train[start:end], self._train_lengths[start:end]

    def get_val_data(self, original_text=False):
        """
        :param original_text. Optional. Whether to return original samples or not.
        :return: Returns the validation data. If original_text returns (original_samples, text_tensor, text_target,
                 text_length), otherwise returns (text_tensor, text_target, text_length)
        """
        if original_text:
            data = pd.read_csv(self._input_file, nrows=self._n_samples)
            samples = data.as_matrix(columns=['SentimentText'])[:, 0]
            return samples[self._val_indices], self._x_val, self._y_val, self._val_lengths
        return self._x_val, self._y_val, self._val_lengths

    def get_test_data(self, original_text=False):
        """
        :param original_text. Optional. Whether to return original samples or not.
        :return: Returns the test data. If original_text returns (original_samples, text_tensor, text_target,
                 text_length), otherwise returns (text_tensor, text_target, text_length)
        """
        if original_text:
            data = pd.read_csv(self._input_file, nrows=self._n_samples)
            samples = data.as_matrix(columns=['SentimentText'])[:, 0]
            return samples[self._test_indices], self._x_test, self._y_test, self._test_lengths
        return self._x_test, self._y_test, self._test_lengths
