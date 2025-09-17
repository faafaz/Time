from transformers import BertConfig, BertModel, BertTokenizer, LlamaConfig, LlamaTokenizer, LlamaModel, GPT2Config, \
    GPT2Model, GPT2Tokenizer


# from .ChatGLM3_6B_Base.configuration_chatglm import ChatGLMConfig


def load_llm(llm_name, llm_layers=32):
    if llm_name == 'ChatGLM3-6B-Base':
        return load_ChatGLM3_6B_Base(llm_layers)
    elif llm_name == 'GPT2_124M':
        return load_GPT2_124M(llm_layers)
    elif llm_name == 'GPT2_355M':
        return load_GPT2_355M(llm_layers)
    elif llm_name == 'BERT':
        return load_bert(llm_layers)
    elif llm_name == 'LLAMA_7B':
        return load_llama_7b(llm_layers)


def load_ChatGLM3_6B_Base(llm_layers=32):
    # chatglm_config = ChatGLMConfig.from_pretrained('./models/LLM/ChatGLM3-6B-Base/')
    return None


def load_GPT2_124M(llm_layers=32):
    bert_config = GPT2Config.from_pretrained('./models/LLM/GPT2_124M/')
    bert_config.num_hidden_layers = llm_layers
    bert_config.output_attentions = True
    bert_config.output_hidden_states = True
    gpt2_model = GPT2Model.from_pretrained(
        './models/LLM/GPT2_124M/',
        trust_remote_code=True,
        local_files_only=True,
        config=bert_config
    )
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(
        './models/LLM/GPT2_124M/',
        trust_remote_code=True,
        local_files_only=True
    )
    # 大模型中的参数不更新
    for param in gpt2_model.parameters():
        param.requires_grad = False
    return gpt2_model, gpt2_tokenizer


def load_GPT2_355M(llm_layers=32):
    bert_config = GPT2Config.from_pretrained('./models/LLM/GPT2_355M/')
    bert_config.num_hidden_layers = llm_layers
    bert_config.output_attentions = True
    bert_config.output_hidden_states = True
    gpt2_model = GPT2Model.from_pretrained(
        './models/LLM/GPT2_355M/',
        trust_remote_code=True,
        local_files_only=True,
        config=bert_config
    )
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(
        './models/LLM/GPT2_355M/',
        trust_remote_code=True,
        local_files_only=True
    )
    # 大模型中的参数不更新
    for param in gpt2_model.parameters():
        param.requires_grad = False
    return gpt2_model, gpt2_tokenizer

# def load_bert(llm_layers=32):
#     bert_config = BertConfig.from_pretrained('../../layers/LLM/BERT/',)
#     bert_config.num_hidden_layers = llm_layers
#     bert_config.output_attentions = True
#     bert_config.output_hidden_states = True
#     bert_model = BertModel.from_pretrained(
#         '../../layers/LLM/BERT/',
#         trust_remote_code=True,
#         local_files_only=True,
#         config=bert_config
#     )
#     bert_tokenizer = BertTokenizer.from_pretrained(
#         '../../layers/LLM/BERT/',
#         trust_remote_code=True,
#         local_files_only=True
#     )
#     # 大模型中的参数不更新
#     for param in bert_model.parameters():
#         param.requires_grad = False
#     return bert_model, bert_tokenizer
def load_bert(llm_layers=32):
    bert_config = BertConfig.from_pretrained('./layers/LLM/BERT/',)
    bert_config.num_hidden_layers = llm_layers
    bert_config.output_attentions = True
    bert_config.output_hidden_states = True
    bert_model = BertModel.from_pretrained(
        './layers/LLM/BERT/',
        trust_remote_code=True,
        local_files_only=True,
        config=bert_config
    )
    bert_tokenizer = BertTokenizer.from_pretrained(
        './layers/LLM/BERT/',
        trust_remote_code=True,
        local_files_only=True
    )
    # 大模型中的参数不更新
    for param in bert_model.parameters():
        param.requires_grad = False
    return bert_model, bert_tokenizer


def load_llama_7b(llm_layers=32):
    llama_config = LlamaConfig.from_pretrained('./models/LLM/Llama_7b/')
    llama_config.num_hidden_layers = llm_layers
    llama_config.output_attentions = True
    llama_config.output_hidden_states = True
    llama_model = LlamaModel.from_pretrained(
        './models/LLM/Llama_7b/',
        trust_remote_code=True,
        local_files_only=True,
        config=llama_config
    )
    llama_tokenizer = LlamaTokenizer.from_pretrained(
        './models/LLM/Llama_7b/',
        trust_remote_code=True,
        local_files_only=True
    )
    # 大模型中的参数不更新
    for param in llama_model.parameters():
        param.requires_grad = False
    return llama_model, llama_tokenizer
