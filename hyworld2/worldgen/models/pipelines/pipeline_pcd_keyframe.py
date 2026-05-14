# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Dict, List, Optional, Union

import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

try:
    from ._pipeline_common import ConditionInputs, KeyframePipelineMixin, PipelineCallConfig
except ImportError:
    from models.pipelines._pipeline_common import ConditionInputs, KeyframePipelineMixin, PipelineCallConfig

try:
    from ...models.worldstereo import WorldStereoModel
except ImportError:
    from models.worldstereo import WorldStereoModel


class KFPCDControllerPipeline(KeyframePipelineMixin, DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Pipeline for image-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer ([`T5Tokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        image_encoder ([`CLIPVisionModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModel), specifically
            the
            [clip-vit-huge-patch14](https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md#vit-h14-xlm-roberta-large)
            variant.
        transformer ([`WanTransformer3DModel`]):
            Conditional Transformer to denoise the input latents.
        scheduler ([`UniPCMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        image_encoder: CLIPVisionModel,
        image_processor: CLIPImageProcessor,
        transformer: WorldStereoModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self._init_keyframe_pipeline_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            image_processor=image_processor,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        render_video: torch.Tensor,
        render_mask: torch.Tensor,
        camera_embedding: torch.Tensor = None,
        extrinsics: torch.Tensor = None,
        intrinsics: torch.Tensor = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 768,
        num_frames: int = 81,
        num_inference_steps: int = 40,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        latent_cond_mode: str = "full_vae",
    ):
        config = PipelineCallConfig(
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_videos_per_prompt=num_videos_per_prompt,
            output_type=output_type,
            return_dict=return_dict,
            max_sequence_length=max_sequence_length,
            latent_cond_mode=latent_cond_mode,
        )
        condition_inputs = ConditionInputs(
            image=image,
            render_video=render_video,
            render_mask=render_mask,
            camera_embedding=camera_embedding,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )
        callback_on_step_end_tensor_inputs = self._normalize_callback_inputs(
            callback_on_step_end, callback_on_step_end_tensor_inputs
        )
        config = self._validate_and_init_call(
            prompt=prompt,
            negative_prompt=negative_prompt,
            condition_inputs=condition_inputs,
            config=config,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image_embeds=image_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )
        self._attention_kwargs = attention_kwargs

        device = self._pipeline_execution_device()
        timesteps = self._prepare_standard_timesteps(config, device)
        prepared = self._prepare_standard_batch(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image_embeds=image_embeds,
            condition_inputs=condition_inputs,
            config=config,
            generator=generator,
            latents=latents,
            device=device,
            timesteps=timesteps,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        latents = self._run_standard_denoise_loop(
            prepared=prepared,
            condition_inputs=condition_inputs,
            config=config,
            attention_kwargs=attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            transformer_extra_kwargs={
                "extrinsics": extrinsics,
                "intrinsics": intrinsics,
            },
        )

        output = self._decode_or_return_latents(
            latents, output_type=output_type, return_dict=return_dict
        )
        self.maybe_free_model_hooks()
        return output
