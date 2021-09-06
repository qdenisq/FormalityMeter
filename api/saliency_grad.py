import torch
import numpy as np
from transformers import AutoModelForSequenceClassification
from typing import Tuple, List, Dict, Optional

def _register_embedding_list_hook(model: AutoModelForSequenceClassification, embeddings_list: List):
    """ Helper function to handle output from bert embedding layer.
    """
    def forward_hook(module, inputs, output):
        embeddings_list.append(output.squeeze(0).clone().cpu().detach().numpy())
    embedding_layer = model.bert.embeddings.word_embeddings
    handle = embedding_layer.register_forward_hook(forward_hook)
    return handle

def _register_embedding_gradient_hooks(model: AutoModelForSequenceClassification, embeddings_gradients: List):
    """ Helper function to register the gradient hook from bert embedding layer.
    """
    def hook_layers(module, grad_in, grad_out):
        embeddings_gradients.append(grad_out[0])
    embedding_layer = model.bert.embeddings.word_embeddings
    hook = embedding_layer.register_backward_hook(hook_layers)
    return hook

def saliency_map(model: AutoModelForSequenceClassification, input_ids: List[int]) -> Tuple[List[float], List[float]]:
    """Compute the gradient-based saliency map. 

    See https://arxiv.org/pdf/2009.13295.pdf for details.

    Args:
        model (AutoModelForSequenceClassification): BERT model
        input_ids (List[int]): tokenized input to the model

    Returns:
        Tuple[List[float], List[float]]: [description]
    """
    torch.enable_grad()
    model.eval()
    # register gradient hook
    embeddings_list = []
    handle = _register_embedding_list_hook(model, embeddings_list)
    embeddings_gradients = []
    hook = _register_embedding_gradient_hooks(model, embeddings_gradients)

    # forward pass
    model.zero_grad()
    model_pred = model(input_ids)
    model_pred.logits[0][0].backward()
    handle.remove()
    hook.remove()

    # normalize gradients
    saliency_grad = embeddings_gradients[0].detach().cpu().numpy()
    saliency_grad = np.sqrt(np.sum((saliency_grad[0] * embeddings_list[0])**2, axis=1))
    norm = np.sum(saliency_grad)
    saliency_grad = [e / norm for e in saliency_grad] 
    return saliency_grad, model_pred.logits[0]