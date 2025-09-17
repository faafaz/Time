import records

# 连接MySQL
db = records.Database('mysql+pymysql://root:haosql@localhost:3306/solar_liu')

# 2. 查询sf_power_sk表
rows = db.query('SELECT * FROM sf_power_sk LIMIT 10')

# 3. 查看结果
print("查询结果:")
for row in rows:
    print(row)

# 4. 转换为DataFrame
df = rows.export('df')
print("\nDataFrame格式:")
print(df)
