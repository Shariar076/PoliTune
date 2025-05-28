# the original was kept unchanged
# tune cp lora_dpo_single_device lora_dpo_single_device_org.py
# config was changed
# tune cp llama3_1/8B_lora_dpo custom_config.yaml

tune run lora_dpo_single_device_w_eval.py --config llama3_1_8B_lora_dpo_single_device.yaml checkpointer.output_dir=checkpoints/ output_dir=outputs/ dataset._component_=dataset.politune_right_pref
# tune run --nproc_per_node 2 lora_dpo_distributed --config custom_config_dpo_llama_2.yaml checkpointer.output_dir=checkpoints/ output_dir=outputs/ dataset._component_=dataset.politune_right_pref
# tune run ppo_full_finetune_single_device --config mistral_7B_full_ppo_single_device.yaml checkpointer.output_dir=checkpoints/ output_dir=outputs/ dataset._component_=dataset.politune_right


# echo ">>>>>>>>>> RUNNING FT FOR 100R0L <<<<<<<<<<"
# tune run lora_dpo_single_device --config llama3_1_8B_lora_dpo_single_device.yaml checkpointer.output_dir=checkpoints/ output_dir=outputs/ dataset._component_=dataset.politune_right_pref
# huggingface-cli upload shariar076/Llama-3.1-8B-Instruct-DPO-100R0L checkpoints/epoch_3
# rm -rf checkpoints/epoch_*

# echo ">>>>>>>>>> RUNNING FT FOR 75R25L <<<<<<<<<<"
# tune run lora_dpo_single_device --config llama3_1_8B_lora_dpo_single_device.yaml checkpointer.output_dir=checkpoints/ output_dir=outputs/ dataset._component_=dataset.politune_75r25l_pref
# huggingface-cli upload shariar076/Llama-3.1-8B-Instruct-DPO-75R25L checkpoints/epoch_3
# rm -rf checkpoints/epoch_*

# echo ">>>>>>>>>> RUNNING FT FOR 50R50L <<<<<<<<<<"
# tune run lora_dpo_single_device --config llama3_1_8B_lora_dpo_single_device.yaml checkpointer.output_dir=checkpoints/ output_dir=outputs/ dataset._component_=dataset.politune_50r50l_pref
# huggingface-cli upload shariar076/Llama-3.1-8B-Instruct-DPO-50R50L checkpoints/epoch_3
# rm -rf checkpoints/epoch_*

# echo ">>>>>>>>>> RUNNING FT FOR 25R75L <<<<<<<<<<"
# tune run lora_dpo_single_device --config llama3_1_8B_lora_dpo_single_device.yaml checkpointer.output_dir=checkpoints/ output_dir=outputs/ dataset._component_=dataset.politune_25r75l_pref
# huggingface-cli upload shariar076/Llama-3.1-8B-Instruct-DPO-25R75L checkpoints/epoch_3
# rm -rf checkpoints/epoch_*

# echo ">>>>>>>>>> RUNNING FT FOR 0R100L <<<<<<<<<<"
# tune run lora_dpo_single_device --config llama3_1_8B_lora_dpo_single_device.yaml checkpointer.output_dir=checkpoints/ output_dir=outputs/ dataset._component_=dataset.politune_left_pref
# huggingface-cli upload shariar076/Llama-3.1-8B-Instruct-DPO-0R100L checkpoints/epoch_3
# rm -rf checkpoints/epoch_*
