import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN_Model(nn.Module):
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        """
        :param rnn_type:
        :param ntoken: num of tokens, vocab size
        :param ninp: dimension of input tokens
        :param nhid: hidden size
        :param nlayers: num of layers
        :param dropout:
        :param tie_weights:
        """
        super(RNN_Model, self).__init__()
        self.ntoken = ntoken
        self.dropout_layer = nn.Dropout(dropout)
        self.encoder = nn.Embedding(num_embeddings=ntoken, embedding_dim=ninp)
        # choose the type of RNN:
        if rnn_type in ['GRU','LSTM']:
            # Docs: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
            # `getattr` is quite useful!
            self.rnn = getattr(nn, rnn_type)(input_size=ninp, hidden_size=nhid, num_layers=nlayers)
        else:
            try:
                 self.nonlinearity = {'RNN_RELU':'relu', 'RNN_TANH':'tanh'}[rnn_type]
            except KeyError:
                raise ValueError("""only support ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=self.nonlinearity, dropout=dropout)
        # decoder, a simple linear layer:
        # ntoken就是vocab size，所以输出维度要这么定。实际上，训练LM，就是next word prediction
        self.decoder = nn.Linear(in_features=nhid, out_features=ntoken)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        # Q: why we should manually initialize encoder and decoder?
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)  # Why?
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        """
        :param input:
        :param hidden: init hidden state to RNN (h_0)
        :return:
        """
        emb = self.dropout_layer(self.encoder(input))
        # output: the hiddens of n tokens; hidden: the last hidden state (h_n)
        output, hidden = self.rnn(emb, hidden)
        # 注意输入decoder的是n个time steps的hidden，所以整个模型的输入的seq_len是多长，输出就会有多长
        # 这也是seq2seq一般的做法，每个timestep的loss加起来组成整体的loss。
        # 只不过这里的decoder就是一个简单的Linear，所以长度必须跟输入保持相同。如果单独一个RNN作为decoder，就可以长度不同了。
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden  # Why log(softmax(x))?

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)




