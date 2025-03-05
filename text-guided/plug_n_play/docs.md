# :rocket: Plug-n-Play (PnP) Control Documentation

Hi Folks,

Thank you for checking out our documentation for using **Plug-n-Play** control in our implementation. Below are the details and code examples to help you integrate and use this feature effectively.

---

## :gear: Code Overview

In our code, we define the `source_batch_size` as follows:

```python
source_batch_size = int(q.shape[0] // 2)
# Note: Plug-n-Play is only applicable when using TWO inputs.
# Refer to our documentation (docs.md) for details on its implementation.
```

When `source_batch_size == 1` (i.e., when `q.shape[0]` is 2 or 3), Plug-n-Play (PnP) will be activated by injecting the unconditional attention maps and features:

```python
if source_batch_size == 1:
    # Inject unconditional data
    q[source_batch_size:2 * source_batch_size] = q[:source_batch_size]
    k[source_batch_size:2 * source_batch_size] = k[:source_batch_size]
```

Similarly, for the injection of features, the following line is used:

```python
hidden_states[source_batch_size:2 * source_batch_size] = hidden_states[:source_batch_size]
```

## :mag_right: Usage Example

To ensure PnP is activated, you should structure your inputs as follows:

```python
uncond_out_src = model.unet(xt[0:1], t, encoder_hidden_states=uncond_embedding[0:1]).sample
uncond_out_tar = model.unet(xt[1:2], t, encoder_hidden_states=uncond_embedding[1:2]).sample

noise_pred_text = model.unet(xt, t, encoder_hidden_states=text_embeddings).sample
##"xt" often contains both the reconstructed and edited samples, while "text_embeddings" oftens contain both the source and target prompt embeddings
```
- **Important:** Here, `text_embeddings` must have a shape where the first dimension is 2, and similarly, `xt` should have a first dimension of 2. This configuration activates Plug-n-Play.
- For unconditional noise computation, we separate the computations into two distinct lines (as shown above) to ensure Plug-n-Play is deactivated.
- Also, when using Plug-n-Play, ensure that you register the time using: `register_time(model, t.item())`.


## :tada: Happy coding and researching

Thanks again for using our Plug-n-Play control implementation. We hope this documentation helps you integrate the feature smoothly into your projects.

Happy coding and researching! :rocket: :bulb:


