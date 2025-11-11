"""
Retrieval model implementations for M-BEIR evaluation.
"""

import logging
from typing import Dict, List, Optional, Union
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class BaseRetrievalModel:
    """Base class for retrieval models."""
    
    def __init__(self, model_name: str, device: str = "cuda", **kwargs):
        """
        Initialize base retrieval model.
        
        Args:
            model_name: Name or path of the model
            device: Device to run model on
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model = None
        self.tokenizer = None
    
    def encode_texts(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """
        Encode list of texts into embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            **kwargs: Additional encoding parameters
            
        Returns:
            Array of text embeddings
        """
        raise NotImplementedError
    
    def encode_images(self, images: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """
        Encode list of images into embeddings.
        
        Args:
            images: List of image paths or image data
            batch_size: Batch size for encoding
            **kwargs: Additional encoding parameters
            
        Returns:
            Array of image embeddings
        """
        raise NotImplementedError("Image encoding not implemented for this model")


class SentenceTransformerModel(BaseRetrievalModel):
    """Sentence Transformer based retrieval model."""
    
    def __init__(self, model_name: str, device: str = "cuda", **kwargs):
        super().__init__(model_name, device, **kwargs)
        logger.info(f"Loading Sentence Transformer model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
    
    def encode_texts(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode texts using Sentence Transformer."""
        return self.model.encode(texts, batch_size=batch_size, **kwargs)


class HuggingFaceModel(BaseRetrievalModel):
    """HuggingFace Transformers based retrieval model."""
    
    def __init__(self, model_name: str, device: str = "cuda", max_length: int = 512, **kwargs):
        super().__init__(model_name, device, **kwargs)
        logger.info(f"Loading HuggingFace model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.max_length = max_length
    
    def encode_texts(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode texts using HuggingFace model with mean pooling."""
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=self.max_length, 
                    return_tensors="pt"
                ).to(self.device)
                
                # Get embeddings
                outputs = self.model(**inputs)
                
                # Mean pooling
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)


class LocalModel(BaseRetrievalModel):
    """Local model loader for pre-downloaded models."""
    
    def __init__(self, model_path: str, device: str = "cuda", model_type: str = "sentence_transformer", **kwargs):
        """
        Initialize local model from local path.
        
        Args:
            model_path: Local path to the model directory
            device: Device to run model on
            model_type: Type of model (sentence_transformer, huggingface, multimodal)
            **kwargs: Additional model-specific parameters
        """
        super().__init__(model_path, device, **kwargs)
        self.model_type = model_type
        logger.info(f"Loading local model from: {model_path}")
        
        # Check if model path exists
        import os
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        # Load model based on type
        if model_type == "sentence_transformer":
            self.model = SentenceTransformer(model_path, device=self.device)
        elif model_type == "huggingface":
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path).to(self.device)
            self.max_length = kwargs.get('max_length', 512)
        else:
            # Default to sentence transformer
            self.model = SentenceTransformer(model_path, device=self.device)
    
    def encode_texts(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode texts using the local model."""
        if self.model_type == "sentence_transformer":
            return self.model.encode(texts, batch_size=batch_size, **kwargs)
        elif self.model_type == "huggingface":
            return self._encode_huggingface_texts(texts, batch_size, **kwargs)
        else:
            # Default to sentence transformer encoding
            return self.model.encode(texts, batch_size=batch_size, **kwargs)
    
    def _encode_huggingface_texts(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode texts using HuggingFace model with mean pooling."""
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=self.max_length, 
                    return_tensors="pt"
                ).to(self.device)
                
                # Get embeddings
                outputs = self.model(**inputs)
                
                # Mean pooling
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)


class MultiModalModel(BaseRetrievalModel):
    """Multi-modal retrieval model supporting both text and image."""
    
    def __init__(self, model_name: str, device: str = "cuda", **kwargs):
        super().__init__(model_name, device, **kwargs)
        logger.info(f"Loading multi-modal model: {model_name}")
        # This would be implemented for models like CLIP, ALBEF, etc.
        # For now, we'll use a placeholder
        self.text_model = SentenceTransformerModel(model_name, device)
    
    def encode_texts(self, texts: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode texts using the text component."""
        return self.text_model.encode_texts(texts, batch_size, **kwargs)
    
    def encode_images(self, images: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Encode images using the vision component."""
        # Placeholder implementation
        logger.warning("Image encoding not fully implemented for multi-modal model")
        return np.random.randn(len(images), 768)


def get_retrieval_model(model_config: Dict) -> BaseRetrievalModel:
    """
    Factory function to get retrieval model based on configuration.
    
    Args:
        model_config: Model configuration dictionary
        
    Returns:
        Initialized retrieval model
    """
    model_name = model_config['name']
    model_type = model_config.get('type', 'text')
    device = model_config.get('device', 'cuda')
    
    if model_type == 'sentence_transformer':
        return SentenceTransformerModel(model_name, device, **model_config)
    elif model_type == 'huggingface':
        return HuggingFaceModel(model_name, device, **model_config)
    elif model_type == 'multimodal':
        return MultiModalModel(model_name, device, **model_config)
    elif model_type == 'local':
        # For local models, use the model path and specify model type
        model_path = model_config.get('path', model_name)
        local_model_type = model_config.get('local_model_type', 'sentence_transformer')
        
        # Filter out parameters that are already explicitly passed
        filtered_config = {k: v for k, v in model_config.items() 
                          if k not in ['device', 'local_model_type', 'path', 'name', 'type']}
        return LocalModel(model_path, device, local_model_type, **filtered_config)
    else:
        # Default to Sentence Transformer
        return SentenceTransformerModel(model_name, device, **model_config)