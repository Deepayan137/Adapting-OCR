from pytorch_pretrained_bert import (OpenAIGPTTokenizer, 
OpenAIGPTModel, OpenAIGPTLMHeadModel)
import torch
import pdb
from tqdm import *

class LM(object):
    def __init__(self):
        self.lm_model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
        self.lm_model.eval()
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.lm_model = self.lm_model.cuda()
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

    def score_sentences(self, sentences):
        scores = []
        for sentence in tqdm(sentences):
            if len(sentence) > 2:
                tokenize_input = self.tokenizer.tokenize(sentence)
                try:
                    tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenize_input)]).cuda()
                    sent_scores = self.lm_model(tensor_input, lm_labels=tensor_input).detach().cpu().numpy()
                except:
                    pdb.set_trace()
            else:
                sent_scores = 0
            scores.append(sent_scores)
        max_score = max(scores)
        norm_scores = [1 - score/max_score for score in scores]
        return norm_scores