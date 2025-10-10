# cd "you own code path."

python finetune_llm.py --model_name llama3.2-3b-instruct --output_dir outputs_100600 --data_config mix_data --adapter_name vaelora --adapter_config vaelora
python finetune_llm.py --model_name llama3.1-8b-instruct --output_dir outputs_100600 --data_config mix_data --adapter_name vaelora --adapter_config vaelora
python finetune_llm.py --model_name mistral-7b-instruct --output_dir outputs_100600 --data_config mix_data --adapter_name vaelora --adapter_config vaelora


