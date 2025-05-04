import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from utils.config import HYDE_MODEL

logger = logging.getLogger(__name__)

class HypotheticalDocumentEmbedding:
    def __init__(
        self,
        model_name: str = HYDE_MODEL,
        device: str = None,
        max_length: int = 512,
    ):
        """Initialize the HyDE model for generating hypothetical answers.
        
        Args:
            model_name: Name of the HuggingFace model to use
            device: Device to run the model on (None for auto-detection, "cpu", "cuda", etc.)
            max_length: Maximum length of generated text
        """
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.max_length = max_length
        
        # Load model and tokenizer
        logger.info(f"Loading HyDE model: {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(device)
        
        logger.info(f"HyDE model loaded successfully")
    
    def generate(self, question: str) -> str:
        """Generate a hypothetical answer for a given question.
        
        Args:
            question: The question to generate an answer for
            
        Returns:
            The generated hypothetical answer
        """
        # Create a prompt for the model
        prompt = f"Answer the following question without any context. Be concise and factual:\nQuestion: {question}\nAnswer:"
        
        # Tokenize the input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate the output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        # Decode the output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer part (remove the prompt)
        answer = generated_text.split("Answer:")[-1].strip()
        
        return answer

# Global instance for singleton access
_hyde_model = None

def get_hyde_model():
    """Get or create the HyDE model instance."""
    global _hyde_model
    if _hyde_model is None:
        _hyde_model = HypotheticalDocumentEmbedding()
    return _hyde_model

def generate_hyde_answer(question: str) -> str:
    """Generate a hypothetical answer for a given question.
    
    This function takes a question, uses a lightweight language model to
    generate a hypothetical answer without any context, and returns this
    hypothetical answer for retrieval purposes.
    
    Args:
        question: The original question
        
    Returns:
        The hypothetical answer that can be used for retrieval
    """
    hyde_model = get_hyde_model()
    return hyde_model.generate(question) 