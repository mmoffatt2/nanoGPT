# OLMo Fine-Tuning with Custom Attention

This script fine-tunes the pre-trained OLMo model using a custom attention variant. Currently, it supports the attention variants: clipped, relu, relu_norm, gated, softplus, softplus_norm, sigmoid, sigmoid_norm, obo, and learned_obo. It fine-tunes using the Wikitext-2 dataset for 20000 steps. 

## Usage

```bash
python3 OLMo_train.py --model_variant <variant>
```