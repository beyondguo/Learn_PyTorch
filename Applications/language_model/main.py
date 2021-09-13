import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx as onnx

from data import Corpus
import model

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='../../data/三国small',
                    help='location of the data corpus')
parser.add_argument('--lang', type=str, default='zh',
                    help='language fo the corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length, backprop through time(bptt)')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')

args = parser.parse_args()

# 设置随机种子便于复现
torch.manual_seed(1)
# 设置cuda
if torch.cuda.is_available():
    if not args.cuda:
        print("Hey, You have a CUDA device! Why not using it??")
device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = Corpus(args.data, lang=args.lang)
"""
Starting from sequential data, batchify arranges the dataset into columns.
For instance, with the alphabet as the sequence and batch size 4, we'd get
┌ a g m s ┐
│ b h n t │
│ c i o u │
│ d j p v │
│ e k q w │
└ f l r x ┘.
These columns are treated as independent by the model, which means that the
dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
batch processing.
解释一下：
上面那个矩阵为什么batch维在竖着那一维？因为torch中RNN默认的输入中，sequence_length是第一维，
也就是行，batch在第二维。所以是这么个形状。
然后按照batch=4，把'abcdefg.....xyz'分成4份，每一份就是一个独立的字符串了，就可以并行处理。
"""

def batchify(data, bsz):
    """按照batch size来分割文本，所以bsz越大，用于训练的每条文本就越短"""
    nbatch = data.shape[0] // bsz
    data = data.narrow(0, 0, nbatch * bsz)  # 剪裁，(dimension, start, length)
    data = data.view(bsz, -1).t().contiguous()  # 这里的转置是为了满足RNN的输入，把seq_len放在第一维
    # 但.contiguous()啥用，还不知道
    return data.to(device)


eval_batch_size = 20
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, args.batch_size)
test_data = batchify(corpus.test, args.batch_size)


###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNN_Model(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout).to(device)
loss_func = nn.NLLLoss()


###############################################################################
# Training code
###############################################################################

"""
get_batch subdivides the source data into chunks of length args.bptt.
If source is equal to the example output of the batchify function, 
┌ a g m s ┐
│ b h n t │
│ c i o u │
│ d j p v │
│ e k q w │
└ f l r x ┘.
with a bptt-limit of 2, we'd get the following two Variables for i = 0:
┌ a g m s ┐ ┌ b h n t ┐
└ b h n t ┘ └ c i o u ┘
Note that despite the name of the function, the subdivison of data is not
done along the batch dimension (i.e. dimension 1), since that was handled
by the batchify function. The chunks are along dimension 0, corresponding
to the seq_len dimension in the LSTM.

就是说，原本在没有seq_len的限制下，就是上面第一个矩阵，然后有了seq_len之后，应该去划分
一个个的输入呢，就是按照seq_len去纵向滑动，得到一个个chunk.
"""

def get_batch(source, i):
    """
    从source中第i位置开始取出seq_len长度的数据。
    首先source data已经有了batch维，这里就是按照seq_len做一个切片；
    然后target这里的都往后挪一个index，这实际上就是一个batch的所有target，
    最后需要view(-1)变形成一维的，这样才能直接输入到NLLLoss损失函数中。
    """
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def repackage_hidden(h):  # 这个玩意儿到底干嘛的？
    """Wraps hidden states in new Tensors, to detach them from their history.
    在网上查了查，相关的解释可以参考：
    https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)  # 还是个递归函数，更不懂了

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    # .eval()是nn.Module的函数，用户转换成evaluation模式，主要针对Dropout,BatchNorm这些组件
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':  # 不是Transformer，就有hidden的概念
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):  # 每bptt的
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * loss_func(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.  # 记录一个epoch的loss
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        """
        这里的设计也是挺"奇特的"。不管bptt多大，这里一个迭代都是batch size大小的数据；
        i是一系列间隔seq_len的值，
        所以bptt的作用就是告诉get_batch函数我一个batch中的文本是多长。
        """
        data, targets = get_batch(train_data, i)  # 以seqlen来取一个个batch
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            # 每一次新的反向传播，都得先把hidden给清理一次
            output, hidden = model(data, hidden)
        loss = loss_func(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():  # 为啥不用optim？？
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
