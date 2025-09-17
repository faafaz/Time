def calculate_data_index(total_len, split_ratios):
    assert len(split_ratios) == 3, "split_ratios必须包含3个元素: [train_ratio, valid_ratio, test_ratio]"
    assert abs(sum(split_ratios) - 1.0) < 1e-6, "比例之和必须等于1"
    assert all(ratio > 0 for ratio in split_ratios), "所有比例必须大于0"

    train_ratio, valid_ratio, test_ratio = split_ratios
    train_end = int(total_len * train_ratio)
    valid_end = int(total_len * (train_ratio + valid_ratio))

    return train_end, valid_end
