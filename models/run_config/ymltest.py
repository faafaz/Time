import yaml

# 读取YAML文件
with open('TimeXer.yml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# 使用配置
model_name = config['model_name']
print(model_name)
d_model = config['d_model']
print(d_model)
