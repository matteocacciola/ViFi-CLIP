from logging import Logger

import torch
import torch.nn as nn

from clip import clip


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.ARCH
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    design_details = {
        'trainer': 'ViFi_CLIP',
        'vision_depth': cfg.TRAINER.ViFi_CLIP.PROMPT_DEPTH_VISION,
        'language_depth': cfg.TRAINER.ViFi_CLIP.PROMPT_DEPTH_TEXT,
        'vision_ctx': cfg.TRAINER.ViFi_CLIP.N_CTX_VISION,
        'language_ctx': cfg.TRAINER.ViFi_CLIP.N_CTX_TEXT
    }
    try:
        # loading JIT archive
        model = clip.build_model(torch.jit.load(model_path, map_location='cpu').eval().state_dict(), design_details)
    except RuntimeError:
        model = clip.build_model(torch.load(model_path, map_location='cpu'), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor):
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        x = x[
            torch.arange(x.shape[0]),
            tokenized_prompts.argmax(dim=-1)
        ].to(self.text_projection.dtype) @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, logger):
        super().__init__()
        clip_dtype = clip_model.dtype
        self.use_prompt_stage = cfg.TRAINER.ViFi_CLIP.PROMPT_MODEL
        ctx_init = cfg.TRAINER.ViFi_CLIP.CTX_INIT
        ZS_evaluation = cfg.TRAINER.ViFi_CLIP.ZS_EVAL
        if ZS_evaluation:
            text_aug = '{{}}'
            tokenized_prompts = torch.cat([clip.tokenize(text_aug.format(c), context_length=77) for c in classnames])
            embedding = clip_model.token_embedding(tokenized_prompts).to(clip_dtype)
            self.register_buffer('complete_text_embeddings', embedding)
            self.register_buffer('tokenized_prompts', tokenized_prompts)
        elif self.use_prompt_stage:
            n_cls = len(classnames)
            # Make sure Language depth >= 1
            assert cfg.TRAINER.ViFi_CLIP.PROMPT_DEPTH_TEXT >= 1, 'In VL prompting, Language prompt depth should be >=1' \
                                                                 '\nPlease use VPT trainer if you want to learn only vision ' \
                                                                 'branch  '
            n_ctx = cfg.TRAINER.ViFi_CLIP.N_CTX_TEXT
            ctx_dim = clip_model.ln_final.weight.shape[0]

            if ctx_init and (n_ctx) <= 4:
                # use given words to initialize context vectors
                ctx_init = ctx_init.replace('_', ' ')
                n_ctx = n_ctx
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).to(clip_dtype)
                ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
                prompt_prefix = ctx_init
            else:
                # random initialization
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=clip_dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = ' '.join(['X'] * n_ctx)
            logger.info('V-L design')
            logger.info(f'Initial text context: \'{prompt_prefix}\'')
            logger.info(f'Number of context words (tokens) for Language prompting: {n_ctx}')
            logger.info(f'Number of context words (tokens) for Vision prompting: {cfg.TRAINER.ViFi_CLIP.N_CTX_VISION}')
            self.ctx = nn.Parameter(ctx_vectors)

            classnames = [name.replace('_', ' ') for name in classnames]
            prompts = [prompt_prefix + ' ' + name + '.' for name in classnames]

            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).to(clip_dtype)

            # These token vectors will be saved when in save_model(),
            # but they should be ignored in load_model() as we want to use
            # those computed using the current class names
            self.register_buffer('token_prefix', embedding[:, :1, :])  # SOS
            self.register_buffer('token_suffix', embedding[:, 1 + n_ctx:, :])  # CLS, EOS
            self.n_cls = n_cls
            self.register_buffer('tokenized_prompts', tokenized_prompts)
        else:
            # No prompting
            ctx_init = ctx_init.replace('_', ' ')
            prompt_prefix = ctx_init
            prompts = [prompt_prefix + ' ' + name + '.' for name in classnames]
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).to(clip_dtype)
            self.register_buffer('complete_text_embeddings', embedding)
            self.register_buffer('tokenized_prompts', tokenized_prompts)

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        if self.use_prompt_stage:
            ctx = self.ctx
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

            prefix = self.token_prefix
            suffix = self.token_suffix
            prompts = self.construct_prompts(ctx, prefix, suffix)
        else:
            prompts = self.complete_text_embeddings

        return prompts


class ViFiCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, logger):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model, logger)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale

    def forward(self, image):
        logit_scale = self.logit_scale.exp()
        prompts = self.prompt_learner()

        # b = image.shape[0]
        # Lets encode the video into required format
        b, t, c, h, w = image.size()
        # Remove the batch dimensions
        image = image.reshape(-1, c, h, w)
        # Now pass the image into CLIP visual encoder
        image_features = self.image_encoder(image)
        # Now again attach the batch dimensions
        image_features = image_features.view(b, t, -1)  # [B, T, 512]
        # Now take the mean along the temporal direction
        image_features = image_features.mean(dim=1, keepdim=False)  # image features are now ready

        # Finally, make the text features
        text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        return logits


def returnCLIP(config, logger: Logger | None =None,
               class_names=None):
    if logger:
        logger.info(f'Loading CLIP (backbone: {config.MODEL.ARCH})')
    clip_model = load_clip_to_cpu(config)

    if logger:
        logger.info('Building ViFi-CLIP CLIP')
    model = ViFiCLIP(config, class_names, clip_model, logger)

    if config.TRAINER.ViFi_CLIP.PROMPT_MODEL:
        if logger:
            logger.info('Turning off gradients in both the image and the text encoder')
        name_to_update = 'prompt_learner'
        for name, param in model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if 'VPT' in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
    else:
        # Now need to control freezing of CLIP for fine-tuning
        train_complete_clip = config.TRAINER.ViFi_CLIP.USE
        if train_complete_clip == 'both':
            if logger:
                logger.info('Turning on gradients for COMPLETE ViFi-CLIP model')
            for name, param in model.named_parameters():
                param.requires_grad_(True)
        elif train_complete_clip == 'image':
            if logger:
                logger.info('Turning on gradients for image side the ViFi-CLIP model')
            for name, param in model.named_parameters():
                if 'image_encoder' in name:  # replace by 'text_encoder' incase you want to freeze text
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
        elif train_complete_clip == 'text':
            if logger:
                logger.info('Turning on gradients for TEXT side the ViFi-CLIP model')
            for name, param in model.named_parameters():
                if 'text_encoder' in name:  # replace by 'text_encoder' incase you want to freeze text
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
        elif train_complete_clip == 'custom_freeze':  # Custom freezing for ViT-B/16
            if logger:
                logger.info('Freezing all but the last two blocks of the Vision Transformer and the text encoder.')
            # Freeze all parameters initially
            for name, param in model.named_parameters():
                param.requires_grad_(False)

            # Unfreeze the last two blocks of the image encoder
            # ViT-B/16 has 12 blocks, so blocks 10 and 11 are the last two
            num_blocks = len(model.image_encoder.transformer.resblocks)
            for i in range(num_blocks - 2, num_blocks):
                for name, param in model.image_encoder.transformer.resblocks[i].named_parameters():
                    param.requires_grad_(True)
                    if logger:
                        logger.info(f'Unfreezing image_encoder.transformer.resblocks[{i}].{name}')

            # Unfreeze the text encoder
            for name, param in model.text_encoder.named_parameters():
                param.requires_grad_(True)
                if logger:
                    logger.info(f'Unfreezing text_encoder.{name}')

            # Unfreeze the prompt_learner (if used)
            for name, param in model.prompt_learner.named_parameters():
                param.requires_grad_(True)
                if logger:
                 logger.info(f'Unfreezing prompt_learner.{name}')

            # Unfreeze logit_scale
            model.logit_scale.requires_grad_(True)
            if logger:
                logger.info('Unfreezing logit_scale')

    # Double check
    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    if logger:
        logger.info(f'Parameters to be updated: {enabled}')
        logger.info(f'Total learnable items: {len(enabled)}')
    model.float()
    return model
