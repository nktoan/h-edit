# :rocket: MasaCtrl Control Documentation

Hi Folks,

Our implementation differs slightly from the original **MasaCtrl**! Following **PnP-Inv**'s approach, we do **not** use `source prompt`, as we found that this results in more stable output images. Please see the appendix of our paper for more details!

## :hammer_and_wrench: How It Works

There is one key argument for controlling attention maps of MasaCtrl:

- `use_editor` - Determines whether attention control is applied (similar to other inversion methods).

## :mag_right: Usage Example

To activate MasaCtrl control, structure your inputs as follows:

```python
noise_preds = model.unet(
    xt, #often contains both reconstructed and edited samples
    t, 
    encoder_hidden_states=prompts, #often contains both unconditional and conditional embeddings
    cross_attention_kwargs=attn_kwargs_attn
).sample
```

By default, `attn_kwargs_attn` is set to True, you may set to `False` for deactivating it.

---

Happy coding and researching! :rocket: :bulb:
