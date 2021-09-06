from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForSequenceClassification
import torch 
import nltk
import os
import argparse

from service import TransformerService
from saliency_grad import saliency_map

if __name__ == "__main__":
    
    ts = TransformerService()
    model = AutoModelForSequenceClassification.from_pretrained("qdenisq/BertFormalityClassificiation")
    tokenizer = AutoTokenizer.from_pretrained("qdenisq/BertFormalityClassificiation")
    
    # Pack using dictionary (recommended)
    artifact = {"model": model, "tokenizer": tokenizer}
    ts.pack("bertModel", artifact)
    os.makedirs("./bento_bundles/bundle", exist_ok=True)
    ts.save_to_dir("./bento_bundles/bundle")