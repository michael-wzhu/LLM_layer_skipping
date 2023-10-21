# LLM_layer_skipping

## STEP 1. Further pretraining/fine-tuning for layer skipping

train a LM with the ability to skipping layers with less performance degradation

- lora + adapter: lora to fine-tune the weights, adapter to alleviate the skipping
- skipping attn/fnn module instead of the whole block
- Further pretraining on UltraChat (only use split 1, 2 of 10 for now, consider increasing the data size in TODO )
- consistency loss 

按照模型大小，验证在不同的模型大小上的结果对比：
- GPT-2 large:
  - 结果（dev集损失）显示：skip whole block 明显差于我们的 skipping attn/fnn modules; adapter添加是明显有效的； consistency loss有用
  - 
- falcon-1b
- llama2-7b / 13b


## step 2

skipping layer selection adaptively

- method 1: learn a skipping plan for each query; 
- method 2: learn to skip for each sentence block (seperated with \n or 句号))
- method 2: learn to skip for each token