```
pip install torch==1.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

```
python predict.py --model_path ./roberta_1_gectorv2.th --vocab_path ./data/output_vocabulary/ --input_file input.txt --output_file output.txt --transformer_model roberta
```