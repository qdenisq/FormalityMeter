import bentoml
from bentoml.adapters import JsonInput
from bentoml.frameworks.transformers import TransformersModelArtifact
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForSequenceClassification
import nltk
import torch
import numpy as np

from saliency_grad import saliency_map

@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([TransformersModelArtifact("bertModel")])
class TransformerService(bentoml.BentoService):
    @bentoml.utils.cached_property
    def _get_nltk(self):
        nltk.download('punkt')
        return None
    
    @bentoml.api(input=JsonInput(), batch=False)
    def predict(self, parsed_json):
        src_text = parsed_json.get("text")
        model = self.artifacts.bertModel.get("model")
        tokenizer = self.artifacts.bertModel.get("tokenizer")
        self._get_nltk
        src_text_split = nltk.tokenize.sent_tokenize(src_text)

        response = {}
        formality_scores = []
        word_scores = []
        num_words = 0
        for text in src_text_split:
            input_ids = tokenizer.encode(text, return_tensors="pt")
            salience_scores, logits = saliency_map(model, input_ids)
            decoded = []
            for input_id in input_ids[0]:
                decoded.append(tokenizer.decode(input_id))
            scores = list(zip(decoded, salience_scores))
            probs = torch.nn.functional.softmax(logits, dim=0)
            word_scores.append({
                "informality_score": probs[0].item(),
                "word_scores": scores[1:-1]
            })
            num_words += len(scores) - 2 
            formality_scores.append(probs[1].item() * (len(scores) - 2 ))
        
        response["score"] = np.sum(formality_scores) / num_words
        response["word_scores"] = word_scores
        return response

