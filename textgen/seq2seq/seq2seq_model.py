# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
from loguru import logger

sys.path.append('..')
from textgen.seq2seq.data_reader import (
    gen_examples, read_vocab, create_dataset,
    one_hot, save_word_dict, load_word_dict,
    SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN
)

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, enc_hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)

    def forward(self, x, lengths):
        sorted_len, sorted_idx = lengths.sort(0, descending=True)
        x_sorted = x[sorted_idx.long()]
        embedded = self.dropout(self.embed(x_sorted))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len.long().cpu().data.numpy(),
                                                            batch_first=True)
        packed_out, hid = self.rnn(packed_embedded)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        out = out[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()

        hid = torch.cat([hid[-2], hid[-1]], dim=1)
        hid = torch.tanh(self.fc(hid)).unsqueeze(0)

        return out, hid


class Attention(nn.Module):
    """
    Luong Attention,根据context vectors和当前的输出hidden states，计算输出
    """

    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(Attention, self).__init__()

        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.linear_in = nn.Linear(enc_hidden_size * 2, dec_hidden_size, bias=False)
        self.linear_out = nn.Linear(enc_hidden_size * 2 + dec_hidden_size, dec_hidden_size)

    def forward(self, output, context, mask):
        # output: batch_size, output_len, dec_hidden_size
        # context: batch_size, context_len, 2*enc_hidden_size

        batch_size = output.size(0)
        output_len = output.size(1)
        input_len = context.size(1)

        context_in = self.linear_in(context.view(batch_size * input_len, -1)).view(
            batch_size, input_len, -1)  # batch_size, context_len, dec_hidden_size

        # context_in.transpose(1,2): batch_size, dec_hidden_size, context_len
        # output: batch_size, output_len, dec_hidden_size
        attn = torch.bmm(output, context_in.transpose(1, 2))
        # batch_size, output_len, context_len

        attn.data.masked_fill(mask, -1e6)

        attn = F.softmax(attn, dim=2)
        # batch_size, output_len, context_len

        context = torch.bmm(attn, context)
        # batch_size, output_len, enc_hidden_size

        output = torch.cat((context, output), dim=2)  # batch_size, output_len, hidden_size*2

        output = output.view(batch_size * output_len, -1)
        output = torch.tanh(self.linear_out(output))
        output = output.view(batch_size, output_len, -1)
        return output, attn


class Decoder(nn.Module):
    """
    decoder会根据已经翻译的句子内容，和context vectors，来决定下一个输出的单词
    """

    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(enc_hidden_size, dec_hidden_size)
        self.rnn = nn.GRU(embed_size, enc_hidden_size, batch_first=True)
        self.out = nn.Linear(dec_hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_mask(self, x_len, y_len):
        # a mask of shape x_len * y_len
        max_x_len = x_len.max()
        max_y_len = y_len.max()
        x_mask = torch.arange(max_x_len, device=x_len.device)[None, :] < x_len[:, None]
        y_mask = torch.arange(max_y_len, device=x_len.device)[None, :] < y_len[:, None]
        mask = ~ x_mask[:, :, None] * y_mask[:, None, :]
        return mask

    def forward(self, ctx, ctx_lengths, y, y_lengths, hid):
        sorted_len, sorted_idx = y_lengths.sort(0, descending=True)
        y_sorted = y[sorted_idx.long()]
        hid = hid[:, sorted_idx.long()]

        y_sorted = self.dropout(self.embed(y_sorted))  # batch_size, output_length, embed_size

        packed_seq = nn.utils.rnn.pack_padded_sequence(y_sorted, sorted_len.long().cpu().data.numpy(), batch_first=True)
        out, hid = self.rnn(packed_seq, hid)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        output_seq = unpacked[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()

        mask = self.create_mask(y_lengths, ctx_lengths)

        output, attn = self.attention(output_seq, ctx, mask)
        output = F.log_softmax(self.out(output), -1)

        return output, hid, attn


class Seq2Seq(nn.Module):
    """
    Seq2Seq, 最后我们构建Seq2Seq模型把encoder, attention, decoder串到一起
    """

    def __init__(
            self,
            encoder_vocab_size,
            decoder_vocab_size,
            embed_size,
            enc_hidden_size,
            dec_hidden_size,
            dropout,
    ):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(vocab_size=encoder_vocab_size,
                               embed_size=embed_size,
                               enc_hidden_size=enc_hidden_size,
                               dec_hidden_size=dec_hidden_size,
                               dropout=dropout)
        self.decoder = Decoder(vocab_size=decoder_vocab_size,  # len(trg_2_ids),
                               embed_size=embed_size,
                               enc_hidden_size=enc_hidden_size,
                               dec_hidden_size=dec_hidden_size,
                               dropout=dropout)

    def forward(self, x, x_lengths, y, y_lengths):
        encoder_out, hid = self.encoder(x, x_lengths)
        output, hid, attn = self.decoder(ctx=encoder_out,
                                         ctx_lengths=x_lengths,
                                         y=y,
                                         y_lengths=y_lengths,
                                         hid=hid)
        return output, attn

    def translate(self, x, x_lengths, y, max_length=128):
        encoder_out, hid = self.encoder(x, x_lengths)
        preds = []
        batch_size = x.shape[0]
        attns = []
        for i in range(max_length):
            output, hid, attn = self.decoder(ctx=encoder_out,
                                             ctx_lengths=x_lengths,
                                             y=y,
                                             y_lengths=torch.ones(batch_size).long().to(y.device),
                                             hid=hid)
            y = output.max(2)[1].view(batch_size, 1)
            preds.append(y)
            attns.append(attn)
        return torch.cat(preds, 1), torch.cat(attns, 1)


class LanguageModelCriterion(nn.Module):
    """
    masked cross entropy loss
    """

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # input: (batch_size * seq_len) * vocab_size
        input = input.contiguous().view(-1, input.size(2))
        # target: batch_size * 1
        target = target.contiguous().view(-1, 1)
        mask = mask.contiguous().view(-1, 1)
        output = -input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


class Seq2SeqModel:
    def __init__(
            self, embed_size=128, hidden_size=128,
            dropout=0.25, epochs=10, batch_size=32,
            model_dir="outputs/", max_length=128,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.max_length = max_length
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.model = None
        self.model_path = os.path.join(self.model_dir, 'seq2seq.pth')
        logger.debug(f"Device: {device}")
        self.loss_fn = LanguageModelCriterion().to(device)
        self.src_vocab_path = os.path.join(self.model_dir, "src_vocab.txt")
        self.trg_vocab_path = os.path.join(self.model_dir, "trg_vocab.txt")
        if os.path.exists(self.src_vocab_path):
            self.src_2_ids = load_word_dict(self.src_vocab_path)
            self.trg_2_ids = load_word_dict(self.trg_vocab_path)
            self.id_2_trgs = {v: k for k, v in self.trg_2_ids.items()}
        else:
            self.src_2_ids = None
            self.trg_2_ids = None
            self.id_2_trgs = None

    def train_model(self,train_data,eval_data=None):
        """
        Trains the model using 'train_data'

        Args:
            train_data: Pandas DataFrame containing the 2 columns - `input_text`, `target_text`.
                        - `input_text`: The input text sequence.
                        - `target_text`: The target text sequence
                        If `use_hf_datasets` is True, then this may also be the path to a TSV file with the same columns.
        Returns:
            training_details: training loss 
        """  # noqa: ignore flake8"
        logger.info("Training model...")
        os.makedirs(self.model_dir, exist_ok=True)
        source_texts, target_texts = create_dataset(train_data)

        self.src_2_ids = read_vocab(source_texts)
        self.trg_2_ids = read_vocab(target_texts)
        save_word_dict(self.src_2_ids, self.src_vocab_path)
        save_word_dict(self.trg_2_ids, self.trg_vocab_path)
        train_src, train_trg = one_hot(source_texts, target_texts, self.src_2_ids, self.trg_2_ids, sort_by_len=True)

        id_2_srcs = {v: k for k, v in self.src_2_ids.items()}
        id_2_trgs = {v: k for k, v in self.trg_2_ids.items()}
        logger.debug(f'train src: {[id_2_srcs[i] for i in train_src[0]]}')
        logger.debug(f'train trg: {[id_2_trgs[i] for i in train_trg[0]]}')

        self.model = Seq2Seq(
            encoder_vocab_size=len(self.src_2_ids),
            decoder_vocab_size=len(self.trg_2_ids),
            embed_size=self.embed_size,
            enc_hidden_size=self.hidden_size,
            dec_hidden_size=self.hidden_size,
            dropout=self.dropout
        )
        self.model.to(device)
        logger.debug(self.model)
        optimizer = torch.optim.Adam(self.model.parameters())

        train_data = gen_examples(train_src, train_trg, self.batch_size, self.max_length)
        train_losses = []
        best_loss = 1e3
        for epoch in range(self.epochs):
            self.model.train()
            total_num_words = 0.
            total_loss = 0.
            for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(train_data):
                mb_x = torch.from_numpy(mb_x).to(device).long()
                mb_x_len = torch.from_numpy(mb_x_len).to(device).long()
                mb_input = torch.from_numpy(mb_y[:, :-1]).to(device).long()
                mb_output = torch.from_numpy(mb_y[:, 1:]).to(device).long()
                mb_y_len = torch.from_numpy(mb_y_len - 1).to(device).long()
                mb_y_len[mb_y_len <= 0] = 1

                mb_pred, attn = self.model(mb_x, mb_x_len, mb_input, mb_y_len)

                mb_out_mask = torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
                mb_out_mask = mb_out_mask.float()

                loss = self.loss_fn(mb_pred, mb_output, mb_out_mask)

                num_words = torch.sum(mb_y_len).item()
                total_loss += loss.item() * num_words
                total_num_words += num_words

                # update optimizer
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                optimizer.step()

                if it % 100 == 0:
                    logger.debug("Epoch :{}/{}, iteration :{}/{} loss:{:.4f}".format(epoch, self.epochs,
                                                                                     it, len(train_data),
                                                                                     loss.item()))
            cur_loss = total_loss / total_num_words
            train_losses.append(cur_loss)
            logger.debug("Epoch :{}/{}, Training loss:{:.4f}".format(epoch, self.epochs, cur_loss))
            if epoch % 1 == 0:
                # find best model
                is_best = cur_loss < best_loss
                best_loss = min(cur_loss, best_loss)
                if is_best:
                    self.save_model()
                    logger.info('Epoch:{}, save new bert model:{}'.format(epoch, self.model_path))
                if eval_data:
                    self.eval_model(eval_data)


        return train_losses

    def eval_model(self, eval_data):
        """
        Evaluates the model on eval_data. Saves results to output_dir.

        Args:
            eval_data: Pandas DataFrame containing the 2 columns - `input_text`, `target_text`.
                        - `input_text`: The input text sequence.
                        - `target_text`: The target text sequence.
                        If `use_hf_datasets` is True, then this may also be the path to a TSV file with the same columns.
        Returns:
            results: Dictionary containing evaluation results.
        """  # noqa: ignore flake8"
        os.makedirs(self.model_dir, exist_ok=True)
        source_texts, target_texts = create_dataset(eval_data)
        logger.info("Evaluating the model...")
        logger.info("Number of examples: {}".format(len(source_texts)))

        if self.src_2_ids is None:
            self.src_2_ids = load_word_dict(self.src_vocab_path)
            self.trg_2_ids = load_word_dict(self.trg_vocab_path)
        if self.model is None:
            if os.path.exists(self.model_path):
                self.model = Seq2Seq(
                    encoder_vocab_size=len(self.src_2_ids),
                    decoder_vocab_size=len(self.trg_2_ids),
                    embed_size=self.embed_size,
                    enc_hidden_size=self.hidden_size,
                    dec_hidden_size=self.hidden_size,
                    dropout=self.dropout
                )
                self.load_model()
                self.model.to(device)
            else:
                raise ValueError("Model not found at {}".format(self.model_path))
        self.model.eval()

        train_src, train_trg = one_hot(source_texts, target_texts, self.src_2_ids, self.trg_2_ids, sort_by_len=True)

        id_2_srcs = {v: k for k, v in self.src_2_ids.items()}
        id_2_trgs = {v: k for k, v in self.trg_2_ids.items()}
        logger.debug(f'evaluate src: {[id_2_srcs[i] for i in train_src[0]]}')
        logger.debug(f'evaluate trg: {[id_2_trgs[i] for i in train_trg[0]]}')
        eval_data = gen_examples(train_src, train_trg, self.batch_size, self.max_length)

        total_num_words = 0.
        total_loss = 0.
        with torch.no_grad():
            for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(eval_data):
                mb_x = torch.from_numpy(mb_x).to(device).long()
                mb_x_len = torch.from_numpy(mb_x_len).to(device).long()
                mb_input = torch.from_numpy(mb_y[:, :-1]).to(device).long()
                mb_output = torch.from_numpy(mb_y[:, 1:]).to(device).long()
                mb_y_len = torch.from_numpy(mb_y_len - 1).to(device).long()
                mb_y_len[mb_y_len <= 0] = 1

                mb_pred, attn = self.model(mb_x, mb_x_len, mb_input, mb_y_len)

                mb_out_mask = torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
                mb_out_mask = mb_out_mask.float()

                loss = self.loss_fn(mb_pred, mb_output, mb_out_mask)

                num_words = torch.sum(mb_y_len).item()
                total_loss += loss.item() * num_words
                total_num_words += num_words
        loss = total_loss / total_num_words
        logger.info(f"Evaluation loss: {loss}")
        return {'loss': loss}

    def predict(self, sentence_list):
        """
        Performs predictions on a list of text.

        Args:
            sentence_list: A python list of text (str) to be sent to the model for prediction. 

        Returns:
            preds: A python list of the generated sequences.
        """  # noqa: ignore flake8"

        if self.src_2_ids is None:
            self.src_2_ids = load_word_dict(self.src_vocab_path)
            self.trg_2_ids = load_word_dict(self.trg_vocab_path)
        if self.model is None:
            if os.path.exists(self.model_path):
                self.model = Seq2Seq(
                    encoder_vocab_size=len(self.src_2_ids),
                    decoder_vocab_size=len(self.trg_2_ids),
                    embed_size=self.embed_size,
                    enc_hidden_size=self.hidden_size,
                    dec_hidden_size=self.hidden_size,
                    dropout=self.dropout
                )
                self.load_model()
                self.model.to(device)
            else:
                raise ValueError("Model not found. Please train the model first.")
        self.model.eval()
        result = []
        for query in sentence_list:
            out = []
            tokens = [token.lower() for token in query]
            tokens = [SOS_TOKEN] + tokens + [EOS_TOKEN]
            src_ids = [self.src_2_ids[i] for i in tokens if i in self.src_2_ids]

            sos_idx = self.trg_2_ids[SOS_TOKEN]
            src_tensor = torch.from_numpy(np.array(src_ids).reshape(1, -1)).long().to(device)
            src_tensor_len = torch.from_numpy(np.array([len(src_ids)])).long().to(device)
            sos_tensor = torch.Tensor([[self.trg_2_ids[SOS_TOKEN]]]).long().to(device)
            translation, attn = self.model.translate(src_tensor, src_tensor_len, sos_tensor,
                                                     self.max_length)
            translation = [self.id_2_trgs[i] for i in translation.data.cpu().numpy().reshape(-1) if
                           i in self.id_2_trgs]
            for word in translation:
                if word != EOS_TOKEN:
                    out.append(word)
                else:
                    break
            result.append(''.join(out))
        return result

    def save_model(self):
        logger.info(f"Saving model into {self.model_path}")
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        logger.info(f"Loading model from {self.model_path}")
        self.model.load_state_dict(torch.load(self.model_path))
