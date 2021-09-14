###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch

import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='../../data/三国small',
                    help='location of the data corpus')
parser.add_argument('--lang', type=str, default='zh',
                    help='language fo the corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default=1000,
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()
join_token = '' if args.lang == 'zh' else ' '
# torch.manual_seed(1)
device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

# 这里需要用到跟训练集相同的词典，用来输出真实的词
corpus = data.Corpus(args.data, lang=args.lang)
ntokens = len(corpus.dictionary)

# load model
with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)
model.eval()

is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
if not is_transformer_model:
    hidden = model.init_hidden(1)

# RNN的输入可以是不定长的，所以理论上我用来trigger的可以是一句话
# input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
trigger_words = [w for w in '诸葛']
print("trigger words:", trigger_words)
input_idx_list = [corpus.dictionary.word2idx[w] for w in trigger_words]
input = torch.tensor(input_idx_list, dtype=torch.long).view(len(input_idx_list), 1).to(device)


with open(args.outf, 'w') as outf:
    print("trigger word:", trigger_words, file=outf)
    print(join_token.join(trigger_words), end=join_token)
    print(join_token.join(trigger_words), end=join_token, file=outf)
    with torch.no_grad():
        for i in range(args.words):  # generate how many words
            if is_transformer_model:
                raise NotImplementedError()
            else:
                output, hidden = model(input, hidden)
                # 这里[-1]是为了取出最后一个time step的输出
                # 否则，如果input是多个词，可能生成的就会不连贯，比方输入"诸葛"，后面却生成不了"亮"。
                word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()  # 这里的temperature啥作用？
                word_idx = torch.multinomial(word_weights, 1)[0]  # randomly sample
                # input.fill_(word_idx)  #  用于输入只有一个词
                # input = word_idx.view(1, 1)   #  用于输入为多个词，但是后面的迭代都是只用前一个词
                # 这里让每次都读取前面N个词用于生成下个词，是不是更好？
                input_idx_list = input_idx_list[-34:] + [(int(word_idx.item()))]  # 每次都用前35个词来预测
                input = torch.tensor(input_idx_list, dtype=torch.long).view(len(input_idx_list), 1).to(device)
            word = corpus.dictionary.idx2word[word_idx]
            if word == '<eos>':
                break
            print(word + ('\n' if i % 20 == 19 else join_token), end=join_token)
            outf.write(word + ('\n' if i % 20 == 19 else join_token))

            # if i % args.log_interval == 0:
            #     print('| Generated {}/{} words'.format(i, args.words))
