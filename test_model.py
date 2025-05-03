# https://github.com/pytorch/torchtune/issues/1188

checkpointer = FullModelMetaCheckpointer(
    checkpoint_dir=checkpoint_dir,
    checkpoint_files=files,
    output_dir=checkpoint_dir,
    adapter_checkpoint="adapter_1.pt",
    recipe_checkpoint="recipe_state.pt",
    model_type=ModelType.LLAMA3
)

checkpoint_dict = checkpointer.load_checkpoint()
model = lora_llama3_8b(
    lora_attn_modules=['q_proj', 'v_proj'],
    apply_lora_to_mlp=False,
    apply_lora_to_output=False,
    lora_rank=8,
    lora_alpha=16,
)
tune["model"].update(checkpoint_dict["adapter"])
del checkpoint_dict["adapter"]
model.load_state_dict(checkpoint_dict["model"])

    def setup(self, cfg: DictConfig) -> None:
        checkpointer = config.instantiate(cfg.checkpointer)
        if self._quantization_mode is None:
            ckpt_dict = checkpointer.load_checkpoint()
        else:
            # weights_only needs to be False when loading a quantized model
            # currently loading a quantized model is only supported with the
            # FullModelTorchTuneCheckpointer
            ckpt_dict = checkpointer.load_checkpoint(weights_only=False)

        self._model = self._setup_model(
            model_cfg=cfg.model,
            model_state_dict=ckpt_dict[utils.MODEL_KEY],
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)