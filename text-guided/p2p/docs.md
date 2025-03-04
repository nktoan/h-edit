# :rocket: Prompt-to-Prompt (P2P) Control Documentation

Hi Folks,

Thank you for checking out our documentation on using **Prompt-to-Prompt (P2P)** control in our implementation! Below, you'll find details and code examples to help you integrate and use this feature effectively. Our implementation introduces some key differences, so read on!

---

## :fire: Key Implementation Detail

A crucial line in our implementation is:

```python
# one line change --- IMPORTANT!!!
if use_controller: #if use_controller is set, P2P will be activated!!!!
    self.controller(attention_probs, is_cross, self.place_in_unet, save_attn)
```

## :hammer_and_wrench: How It Works

We introduce two key arguments for controlling attention maps:

- `use_controller` - Determines whether attention control is applied (similar to other inversion methods).
- `save_attn` - Specifically used for the implicit h-edit version, which involves multiple optimization loops over editing terms. Only the last optimization steps will save the attentions, while the previous steps do not.


## :mag_right: Usage Example

To activate P2P control, structure your inputs as follows:

```python
noise_preds = model.unet(
    xt, #often contains both reconstructed and edited samples
    t, 
    encoder_hidden_states=prompts, #often contains both unconditional and conditional embeddings
    cross_attention_kwargs=attn_kwargs_attn
).sample
```

You can configure `cross_attention_kwargs` using the following options (can be both of them):  

- `attn_kwargs_attn = {'save_attn': False/True}`, by default `save_attn` is set to `True`
- `attn_kwargs_attn = {'use_controller': False/True}`, by default `use_controller` is set to `True`

For more details, refer to our **h-edit** implementation.

---

Happy coding and researching! :rocket: :bulb: