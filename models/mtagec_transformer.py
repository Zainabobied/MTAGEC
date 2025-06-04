"""
Unified Transformer model for MTAGEC (correction, error-type, evidence extraction).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    T5ForConditionalGeneration,
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    BertModel,
    BertConfig
)
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from typing import Optional, Tuple, List, Dict, Any, Union


class MTAGECModel(nn.Module):
    """
    MTAGEC model based on T5 architecture with unified softmax for:
    1. Token generation (correction)
    2. Pointer mechanism (evidence extraction)
    3. Error type classification
    
    This implements the architecture described in Section 4.2 of the paper.
    """
    
    def __init__(self, config, error_types=None, alpha=0.5, model_type=None):
        """
        Initialize MTAGEC model with T5 or BERT backbone
        
        Args:
            config: Model configuration
            error_types: List of error type labels
            alpha: Blending factor for encoder states (Eq. in Section 4.2)
            model_type: Type of model ('t5' or 'bert')
        """
        super().__init__()
        self.alpha = alpha
        self.config = config
        
        # Determine model type based on config
        if model_type is None:
            if hasattr(config, "model_type"):
                self.model_type = config.model_type
            else:
                self.model_type = "t5"  # Default to T5
        else:
            self.model_type = model_type
            
        # Initialize base model based on type
        if self.model_type == "t5":
            self.base_model = T5ForConditionalGeneration(config)
        elif self.model_type in ["bert", "arabert"]:
            self.base_model = BertModel(config)
            # For BERT, we need additional components for seq2seq
            self.decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.num_attention_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.hidden_dropout_prob,
                    activation="gelu",
                    batch_first=True
                ),
                num_layers=6  # Use 6 decoder layers
            )
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Error type classification
        if error_types is None:
            # Default 25 error types from the paper
            self.error_types = [
                "OA", "OH", "OT", "OW", "OC", "ON", "OS", "OG", "OR", "OD", "OM",  # Orthographic
                "XF", "XG", "XN", "XT", "XM", "MI", "MT",  # Morphological
                "SC", "SW",  # Semantic
                "PC", "PM", "PT",  # Punctuation
                "MG", "SP"  # Merge/Split
            ]
        else:
            self.error_types = error_types
            
        self.num_error_types = len(self.error_types)
        
        # Error type embeddings
        self.error_type_embeddings = nn.Embedding(self.num_error_types, config.d_model)
        
        # MLP for transforming encoder states for pointer network
        self.pointer_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )
        
        # Error type projection
        self.error_type_proj = nn.Linear(config.d_model, self.num_error_types)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pointer_mask=None,
        error_type_labels=None,
        lambda_weight=0.7,  # Down-weight for explanation loss (Eq. 30)
    ):
        """
        Forward pass with unified softmax over vocabulary, pointer indices, and error types
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Process based on model type
        if self.model_type == "t5":
            # Run the base T5 model
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=None,  # We'll handle the loss ourselves
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,  # We need hidden states
                return_dict=True,
            )
            
            # Get encoder and decoder hidden states
            encoder_hidden_states = outputs.encoder_last_hidden_state
            decoder_hidden_states = outputs.decoder_hidden_states[-1]
            token_logits = outputs.logits
            
        elif self.model_type in ["bert", "arabert"]:
            # Run BERT encoder
            encoder_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            
            # Get encoder hidden states
            encoder_hidden_states = encoder_outputs.last_hidden_state
            
            # Run decoder if decoder_input_ids is provided, otherwise use encoder outputs
            if decoder_input_ids is not None:
                # Create decoder attention mask if not provided
                if decoder_attention_mask is None and decoder_input_ids is not None:
                    decoder_attention_mask = (decoder_input_ids != self.config.pad_token_id).float()
                
                # Create a memory mask for the encoder-decoder attention
                memory_mask = torch.zeros(
                    (decoder_input_ids.size(0), decoder_input_ids.size(1), input_ids.size(1)),
                    device=input_ids.device
                )
                
                # Run decoder
                decoder_outputs = self.decoder(
                    tgt=decoder_inputs_embeds if decoder_inputs_embeds is not None
                        else self.base_model.embeddings(decoder_input_ids),
                    memory=encoder_hidden_states,
                    tgt_mask=None,  # Let PyTorch create the causal mask
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=~decoder_attention_mask.bool() if decoder_attention_mask is not None else None,
                    memory_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None,
                )
                
                decoder_hidden_states = decoder_outputs
                
                # Project to vocabulary
                token_logits = self.lm_head(decoder_hidden_states)
            else:
                # For inference, we'll need to handle this differently
                # For now, just use encoder outputs
                decoder_hidden_states = encoder_hidden_states
                token_logits = self.lm_head(decoder_hidden_states)
        
        # Prepare for unified softmax (Eq. 29)
        batch_size, decoder_seq_len, hidden_size = decoder_hidden_states.shape
        
        # 1. Token logits (already computed above)
        
        # 2. Pointer logits for evidence extraction
        # Transform encoder states with MLP
        transformed_encoder_states = self.pointer_mlp(encoder_hidden_states)
        
        # Blend with original encoder embeddings (Eq. in Section 4.2)
        blended_encoder_states = (
            self.alpha * encoder_hidden_states + 
            (1 - self.alpha) * transformed_encoder_states
        )
        
        # Compute pointer logits
        # [batch_size, decoder_seq_len, encoder_seq_len]
        pointer_logits = torch.bmm(
            decoder_hidden_states,  # [batch_size, decoder_seq_len, hidden_size]
            blended_encoder_states.transpose(1, 2)  # [batch_size, hidden_size, encoder_seq_len]
        )
        
        # 3. Error type logits
        # [batch_size, decoder_seq_len, num_error_types]
        error_type_logits = self.error_type_proj(decoder_hidden_states)
        
        # Combine logits for unified softmax
        # We need to ensure the logits are properly aligned in the same space
        # Vocabulary: [0, vocab_size)
        # Pointers: [vocab_size, vocab_size + encoder_seq_len)
        # Error types: [vocab_size + encoder_seq_len, vocab_size + encoder_seq_len + num_error_types)
        
        # Apply masks if needed
        if pointer_mask is not None:
            # Zero out pointer logits where mask is 0
            pointer_logits = pointer_logits * pointer_mask.unsqueeze(1)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Prepare loss mask to separate vocabulary tokens from pointer/error tokens
            vocab_size = token_logits.shape[-1]
            encoder_seq_len = pointer_logits.shape[-1]
            
            # Identify which positions in labels correspond to vocabulary tokens
            vocab_positions = (labels >= 0) & (labels < vocab_size)
            
            # Identify which positions correspond to pointer indices
            pointer_positions = (labels >= vocab_size) & (labels < vocab_size + encoder_seq_len)
            
            # Identify which positions correspond to error types
            error_positions = labels >= vocab_size + encoder_seq_len
            
            # Adjust pointer and error type labels to their respective spaces
            pointer_labels = labels.clone()
            pointer_labels[pointer_positions] = pointer_labels[pointer_positions] - vocab_size
            
            error_type_labels = labels.clone()
            error_type_labels[error_positions] = error_type_labels[error_positions] - (vocab_size + encoder_seq_len)
            
            # Compute losses for each component
            # 1. Token loss
            token_loss = F.cross_entropy(
                token_logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100,
                reduction='none'
            ).view(batch_size, -1)
            
            # Only consider token loss where labels are vocabulary tokens
            token_loss = token_loss * vocab_positions.float()
            
            # 2. Pointer loss
            # Create a new label tensor for pointer positions
            pointer_label_mask = pointer_positions.float()
            pointer_loss = F.cross_entropy(
                pointer_logits.view(-1, encoder_seq_len),
                pointer_labels.view(-1),
                ignore_index=-100,
                reduction='none'
            ).view(batch_size, -1)
            
            pointer_loss = pointer_loss * pointer_label_mask
            
            # 3. Error type loss
            error_label_mask = error_positions.float()
            error_loss = F.cross_entropy(
                error_type_logits.view(-1, self.num_error_types),
                error_type_labels.view(-1),
                ignore_index=-100,
                reduction='none'
            ).view(batch_size, -1)
            
            error_loss = error_loss * error_label_mask
            
            # Combine losses with weighting (Eq. 30)
            # Token loss has weight 1.0
            # Pointer and error losses have weight lambda
            combined_loss = token_loss + lambda_weight * (pointer_loss + error_loss)
            
            # Average over non-padding positions
            non_padding = (labels != -100).float()
            loss = combined_loss.sum() / non_padding.sum()
        
        if not return_dict:
            output = (token_logits, pointer_logits, error_type_logits) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
            
        return MTAGECOutput(
            loss=loss,
            token_logits=token_logits,
            pointer_logits=pointer_logits,
            error_type_logits=error_type_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
    
    def generate_with_explanation(
        self,
        input_ids,
        attention_mask=None,
        max_length=None,
        min_length=None,
        do_sample=None,
        early_stopping=None,
        num_beams=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        bad_words_ids=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        length_penalty=None,
        no_repeat_ngram_size=None,
        encoder_no_repeat_ngram_size=None,
        num_return_sequences=None,
        decoder_start_token_id=None,
        use_cache=None,
        **model_kwargs
    ):
        """
        Generate sequences with unified decoding for correction and explanation
        """
        # Handle generation based on model type
        if self.model_type == "t5":
            # Standard generation with beam search for T5
            outputs = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                early_stopping=early_stopping,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                bad_words_ids=bad_words_ids,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
                num_return_sequences=num_return_sequences,
                decoder_start_token_id=decoder_start_token_id,
                use_cache=use_cache,
                **model_kwargs
            )
        elif self.model_type in ["bert", "arabert"]:
            # For BERT, we need to implement custom generation
            # This is a simplified version - in practice, you'd implement beam search
            batch_size = input_ids.shape[0]
            device = input_ids.device
            
            # Encode input
            encoder_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            encoder_hidden_states = encoder_outputs.last_hidden_state
            
            # Initialize decoder inputs with start token
            decoder_input_ids = torch.full(
                (batch_size, 1),
                decoder_start_token_id if decoder_start_token_id is not None else self.config.pad_token_id,
                dtype=torch.long,
                device=device
            )
            
            # Generate tokens auto-regressively
            for _ in range(max_length if max_length is not None else 100):
                # Get decoder embeddings
                decoder_inputs_embeds = self.base_model.embeddings(decoder_input_ids)
                
                # Create decoder attention mask
                decoder_attention_mask = torch.ones_like(decoder_input_ids)
                
                # Run decoder
                decoder_outputs = self.decoder(
                    tgt=decoder_inputs_embeds,
                    memory=encoder_hidden_states,
                    tgt_key_padding_mask=None,
                    memory_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None,
                )
                
                # Project to vocabulary
                next_token_logits = self.lm_head(decoder_outputs[:, -1:])
                
                # Apply temperature and top-k/top-p sampling
                if do_sample:
                    if temperature is not None and temperature > 0:
                        next_token_logits = next_token_logits / temperature
                    if top_k is not None and top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = -float("Inf")
                    if top_p is not None and top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            dim=1, index=sorted_indices, src=sorted_indices_to_remove
                        )
                        next_token_logits[indices_to_remove] = -float("Inf")
                    
                    # Sample from the distribution
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                # Append to decoder_input_ids
                decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=-1)
                
                # Check if we've generated an EOS token
                if (next_tokens == eos_token_id).any():
                    break
            
            outputs = decoder_input_ids
        
        return outputs
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, error_types=None, *model_args, **kwargs):
        """Load pretrained model with MTAGEC-specific parameters"""
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        
        # Determine model type from pretrained path
        model_type = None
        if "t5" in pretrained_model_name_or_path.lower():
            model_type = "t5"
        elif "bert" in pretrained_model_name_or_path.lower() or "arabert" in pretrained_model_name_or_path.lower():
            model_type = "bert"
        
        # Initialize MTAGEC model
        mtagec_model = cls(config, error_types=error_types, model_type=model_type)
        
        # Load pretrained weights
        if model_type == "t5":
            pretrained_model = T5ForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
            mtagec_model.base_model = pretrained_model
        elif model_type in ["bert", "arabert"]:
            pretrained_model = BertModel.from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
            mtagec_model.base_model = pretrained_model
            # Initialize decoder and LM head with random weights
            # These will be trained from scratch
        
        return mtagec_model


class MTAGECOutput(Seq2SeqLMOutput):
    """
    Output class for MTAGEC model with additional fields for pointer and error type logits
    """
    
    def __init__(
        self,
        loss=None,
        token_logits=None,
        pointer_logits=None,
        error_type_logits=None,
        past_key_values=None,
        decoder_hidden_states=None,
        decoder_attentions=None,
        cross_attentions=None,
        encoder_last_hidden_state=None,
        encoder_hidden_states=None,
        encoder_attentions=None,
    ):
        super().__init__(
            loss=loss,
            logits=token_logits,
            past_key_values=past_key_values,
            decoder_hidden_states=decoder_hidden_states,
            decoder_attentions=decoder_attentions,
            cross_attentions=cross_attentions,
            encoder_last_hidden_state=encoder_last_hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attentions=encoder_attentions,
        )
        self.pointer_logits = pointer_logits
        self.error_type_logits = error_type_logits
