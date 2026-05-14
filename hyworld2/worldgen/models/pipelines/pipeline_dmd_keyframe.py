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
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

try:
    from ._pipeline_common import ConditionInputs, KeyframePipelineMixin, PipelineCallConfig
except ImportError:
    from models.pipelines._pipeline_common import ConditionInputs, KeyframePipelineMixin, PipelineCallConfig

try:
    from ...models.worldstereo import WorldStereoRefSModel
except ImportError:
    from models.worldstereo import WorldStereoRefSModel


class RefKFDMDGeneratorPipeline(KeyframePipelineMixin, DiffusionPipeline, WanLoraLoaderMixin):
    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
            self,
            tokenizer: AutoTokenizer,
            text_encoder: UMT5EncoderModel,
            image_encoder: CLIPVisionModel,
            image_processor: CLIPImageProcessor,
            transformer: WorldStereoRefSModel,
            vae: AutoencoderKLWan,
            scheduler,
            device,
            vae_compile: bool = False,
            vae_compile_mode: str = "max-autotune"
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
        # self.transformer = transformer
        self.device_ = device
        
        # VAE compile settings
        self.vae_compile = vae_compile
        self.vae_compile_mode = vae_compile_mode

    def _execution_device(self):
        return self.device_

    # @torch.no_grad()
    # we need to update generator through the full generator inference
    # NOTE: generator does not need CFG
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
            # new params for reference
            reference_video=None,
            ref_index=None,
            camera_qt=None,
            camera_qt_ref=None,
            latent_cond_mode="full_vae",
            # new params end
            mode: str = "train",
            height: int = 480,
            width: int = 768,
            num_frames: int = 81,
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
            **kwargs
    ):
        config = PipelineCallConfig(
            height=height,
            width=width,
            num_frames=num_frames,
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
            reference_video=reference_video,
            ref_index=ref_index,
            camera_qt=camera_qt,
            camera_qt_ref=camera_qt_ref,
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
            guidance_scale=1.0,
        )
        self._attention_kwargs = attention_kwargs

        device = self._pipeline_execution_device()
        timesteps = self._prepare_dmd_timesteps(mode)
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
            do_classifier_free_guidance=False,
            no_grad_prompt=True,
            use_compile=self.vae_compile,
            compile_mode=self.vae_compile_mode,
            latent_autocast=True,
            cast_reference_to_vae_dtype=False,
        )
        latents = self._run_dmd_denoise_loop(
            prepared=prepared,
            condition_inputs=condition_inputs,
            attention_kwargs=attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            mode=mode,
        )

        return self._decode_or_return_latents(
            latents, output_type=output_type, return_dict=return_dict, use_autocast=True
        )

    # only used for training
    def fast_infer(
            self,
            random_noise_latent,
            condition_latent,
            render_latent,
            render_mask,
            prompt_embeds,
            image_embeds,
            camera_embedding=None,
            # new params for reference
            reference_latent=None,
            camera_qt=None,
            camera_qt_ref=None,
            ref_index=None
    ):

        timesteps = self.scheduler.gen_train_timesteps()
        latents = random_noise_latent
        device = self._execution_device()

        # print("[DEBUG] FAST INFER Timesteps:", timesteps)

        pred_latents = None

        for i, t in enumerate(timesteps):

            latent_model_input = torch.cat([latents, condition_latent], dim=1)
            timestep = t.expand(latents.shape[0]).to(device)

            # Only the last step allows gradient backpropagation
            if i < len(timesteps) - 1:
                with torch.no_grad():
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        render_latent=render_latent,
                        point_map_latent=None,
                        point_map_ref_latent=None,
                        reference_latent=reference_latent,
                        render_mask=render_mask,
                        camera_embedding=camera_embedding,
                        camera_qt=camera_qt,
                        camera_qt_ref=camera_qt_ref,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        return_dict=False,
                        ref_index=ref_index,
                    )[0]
                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, latents, timesteps, i)
            else:
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    render_latent=render_latent,
                    point_map_latent=None,
                    point_map_ref_latent=None,
                    reference_latent=reference_latent,
                    render_mask=render_mask,
                    camera_embedding=camera_embedding,
                    camera_qt=camera_qt,
                    camera_qt_ref=camera_qt_ref,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    return_dict=False,
                    ref_index=ref_index,
                )[0]
                # compute the previous noisy sample x_t -> x_t-1, if t is the last step of generator, pred x_0
                pred_latents = self.scheduler.step(noise_pred, latents, timesteps, i)

        return pred_latents