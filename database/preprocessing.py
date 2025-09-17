from database.mysql_dataloader import querySolarData

if __name__ == "__main__":
    # 查询参数
    farmid = "1"  # 替换为实际的电站ID
    start_time = "2022-03-01 00:00:00"
    end_time = "2022-03-31 23:45:00"

    # 执行查询
    df = querySolarData(farmid, start_time, end_time)
    print(df.head(10))
