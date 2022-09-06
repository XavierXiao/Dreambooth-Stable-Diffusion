# Dreambooth on Stable Diffusion

This is an implementtaion of Google's [Dreambooth](https://arxiv.org/abs/2208.12242) with [Stable Diffusion](https://github.com/CompVis/stable-diffusion). The original Dreambooth is based on [Imagen](https://imagen.research.google/) text-to-image model. However, neither the model nor the pre-trained weights of Imagen is available. To enable people to fine-tune a text-to-image model with a few examples, I implemented the idea of Dreambooth on Stable diffusion.

This code repository is based on that of [Textual Inversion](https://github.com/rinongal/textual_inversion). Note that Textual Inversion only optimizes word ebedding, while dreambooth fine-tunes the whole diffusion model.

The implementation makes minimum changes over the official codebase of Textual Inversion, and in fact some components in Textual Inversion, such as the embedding manager, are not deleted, although they will never be used here.

## Usage
