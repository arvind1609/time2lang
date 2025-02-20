from transformers.models.llama import LlamaForCausalLM
from transformers import pipeline, AutoTokenizer
from torch import nn

class CustomLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = None  # To store the output of the hook
        
        # Register the hook during initialization
        self.model.norm.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, inputs, outputs):
        self.embeddings = outputs
        
    def forward(self, inputs_embeds=None, **kwargs):
        # Reset embeddings for the current forward pass
        self.embeddings = None
        
        # Call the parent forward method
        super().forward(inputs_embeds=inputs_embeds, **kwargs)
        
        # Return the hooked embeddings
        return self.embeddings