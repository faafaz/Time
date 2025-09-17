import torch


def get_simple_prompt(batch_x, seq_len, pred_len):
    # 这个高一些
    description = ("This is the multivariate monitoring data of a solar power plant, including hourly recorded "
                   "radiation levels (in W/m ²) and power generation (in W/m ²) kW）。 Power is the target "
                   "predictor variable.")
    batch_size, T, N = batch_x.size()
    # description = ("There is a positive correlation between power and total radiation intensity, "
    #                "The power output usually increases with the increase of radiation intensity, "
    #                "but there is a brief lag effect.")
    prompt = []
    for b in range(batch_size):
        prompt_ = (
            f"<|start_prompt|>Background Knowledge: {description};"
            f"Task description: forecast the next {str(pred_len)} steps given the previous {str(seq_len)} steps information; "
            "<|<end_prompt>|>"
        )
        prompt.append(prompt_)
    return prompt


def get_calculate_prompt(batch_x, seq_len, pred_len):
    description = ("This is the multivariate monitoring data of a solar power plant, including hourly recorded "
                   "radiation levels (in W/m ²) and power generation (in W/m ²) kW）。 Power is the target "
                   "predictor variable.")
    # (32,2,96)batch_x -> (32,2) -> (32)
    min_values = torch.min(batch_x, dim=2)[0][:, -1:]
    max_values = torch.max(batch_x, dim=2)[0][:, -1:]
    medians = torch.median(batch_x, dim=2).values
    medians = medians[:, -1:]
    lags = calculate_lags(batch_x)  # 计算滞后值
    trends = batch_x.diff(dim=2).sum(dim=2)[:, -1:]  # 数据趋势
    prompt = []
    for b in range(batch_x.shape[0]):
        min_values_str = str(min_values[b].tolist()[0])
        max_values_str = str(max_values[b].tolist()[0])
        median_values_str = str(medians[b].tolist()[0])
        lags_values_str = str(lags[b].tolist())
        prompt_ = (
            f"<|start_prompt|>Dataset description: {description}"
            f"Task description: forecast the next {str(pred_len)} steps given the previous {str(seq_len)} steps information; "
            "Input statistics: "
            f"min value {min_values_str}, "
            f"max value {max_values_str}, "
            f"median value {median_values_str}, "
            f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
            f"top 5 lags are : {lags_values_str} ."
            "<|<end_prompt>|>"
        )
        prompt.append(prompt_)
    return prompt


def calculate_lags(x_enc, top_k=5):
    x_enc = x_enc.transpose(-1, -2)  # (64, 1, 96)
    q_fft = torch.fft.rfft(x_enc, dim=-1)
    k_fft = torch.fft.rfft(x_enc, dim=-1)
    res = q_fft * torch.conj(k_fft)
    corr = torch.fft.irfft(res, dim=-1)
    mean_value = torch.mean(corr, dim=1)
    _, lags = torch.topk(mean_value, top_k, dim=-1)
    return lags
