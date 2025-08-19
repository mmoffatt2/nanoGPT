# Mixing Block Outputs Before Final Layer Norm

The `use_ln_f_input_mixer` option blends hidden states from each transformer block
(including the initial embedding) into a single representation prior to the final
layer normalization (`ln_f`). When enabled, a learnable weight is assigned to every
layer's output so the model can linearly combine them before normalization.

## Usage

Enable the feature in code:

```python
config.use_ln_f_input_mixer = True
```

or from the command line:

```bash
python train.py ... --use_ln_f_input_mixer
```

Weights are initialized to focus on the last block's output, preserving the
standard behavior when the option is disabled.

