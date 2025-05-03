# tune cp lora_dpo_single_device lora_dpo_single_device_org.py
# tune cp llama3_1/8B_lora_dpo custom_config.yaml

# tune run lora_dpo_single_device.py --config custom_config_dpo_llama_3_1.yaml checkpointer.output_dir=checkpoints/ output_dir=outputs/ dataset._component_=dataset.politune_right_pref
tune run --nproc_per_node 2 lora_dpo_distributed --config custom_config_dpo_llama_2.yaml checkpointer.output_dir=checkpoints/ output_dir=outputs/ dataset._component_=dataset.politune_right_pref