import records
from functools import wraps

from database.data_check import check_missing_dates, print_missing_dates_summary
from database.sql_wrapper import db_connector_logger, print_sql


@db_connector_logger()
def querySolarData(db, farmid, start_time, end_time):
    """
    查询指定电站和时间范围的光伏功率和辐射数据

    Parameters:
    - db_url: 数据库连接URL
    - farmid: 电站ID (FARMID)
    - start_time: 开始时间 '2024-01-01 00:00:00'
    - end_time: 结束时间 '2024-01-01 23:59:59'

    Returns:
    - pandas.DataFrame: 包含OBSERVETIME, ACTIVEPOWER, TOTALRADIATION的数据
    """

    # 方案4：先去重再JOIN（适用于完全重复的情况）
    sql_query = """
    SELECT 
        p.OBSERVETIME,
        p.ACTIVEPOWER,
        m.TOTALRADIATION
    FROM (
        SELECT DISTINCT OBSERVETIME, FARMID, ACTIVEPOWER
        FROM sf_power_sk
        WHERE FARMID = :farmid
            AND OBSERVETIME >= :start_time
            AND OBSERVETIME <= :end_time
    ) p
    INNER JOIN (
        SELECT DISTINCT OBSERVETIME, FARMID, TOTALRADIATION
        FROM sf_met_sk
        WHERE FARMID = :farmid
            AND OBSERVETIME >= :start_time
            AND OBSERVETIME <= :end_time
    ) m ON p.OBSERVETIME = m.OBSERVETIME 
        AND p.FARMID = m.FARMID
    ORDER BY p.OBSERVETIME;
    """

    # 查询参数
    params = {
        'farmid': farmid,
        'start_time': start_time,
        'end_time': end_time
    }

    print_sql(sql_query, params)

    result = db.query(sql_query, **params)
    res = result.export('df')
    print(len(res))
    return res


@db_connector_logger()
def querySolarDataWithNWP(db, farmid, start_time, end_time):
    """
    查询指定电站和时间范围的综合光伏数据（功率、气象、天气预报）

    Parameters:
    - db: 数据库连接对象
    - farmid: 电站ID (FARMID)
    - start_time: 开始时间 '2024-01-01 00:00:00'
    - end_time: 结束时间 '2024-01-01 23:59:59'

    Returns:
    - pandas.DataFrame: 包含OBSERVETIME, ACTIVEPOWER, TOTALRADIATION, WINDSPEED, PRESSURE, HUMIDITY, TEMP的数据
    """

    # 三表关联查询，使用LEFT JOIN保证功率数据完整性
    sql_query = """
    SELECT 
        p.OBSERVETIME,
        p.ACTIVEPOWER,
        m.TOTALRADIATION,
        n.WINDSPEED,
        n.PRESSURE,
        n.HUMIDITY,
        n.TEMP
    FROM (
        SELECT DISTINCT OBSERVETIME, FARMID, ACTIVEPOWER
        FROM sf_power_sk
        WHERE FARMID = :farmid
            AND OBSERVETIME >= :start_time
            AND OBSERVETIME <= :end_time
    ) p
    LEFT JOIN (
        SELECT DISTINCT OBSERVETIME, FARMID, TOTALRADIATION
        FROM sf_met_sk
        WHERE FARMID = :farmid
            AND OBSERVETIME >= :start_time
            AND OBSERVETIME <= :end_time
    ) m ON p.OBSERVETIME = m.OBSERVETIME 
        AND p.FARMID = m.FARMID
    LEFT JOIN (
        SELECT DISTINCT FORETIME, FARMID, WINDSPEED, PRESSURE, HUMIDITY, TEMP
        FROM sf_nwp
        WHERE FARMID = :farmid
            AND FORETIME >= :start_time
            AND FORETIME <= :end_time
    ) n ON p.OBSERVETIME = n.FORETIME 
        AND p.FARMID = n.FARMID
    ORDER BY p.OBSERVETIME;
    """

    # 查询参数
    params = {
        'farmid': farmid,
        'start_time': start_time,
        'end_time': end_time
    }

    print_sql(sql_query, params)

    result = db.query(sql_query, **params)
    res = result.export('df')
    print(f"查询到 {len(res)} 条记录")
    return res


@db_connector_logger()
def queryNWPData(db, farmid, limit=10):
    """
    查询指定电站最早的天气预报数据

    Parameters:
    - db: 数据库连接对象
    - farmid: 电站ID (FARMID)
    - limit: 返回记录数量，默认10条

    Returns:
    - pandas.DataFrame: 包含sf_nwp表所有字段的最早数据
    """

    sql_query = """
    SELECT 
        farmid,
        FORETIME,
        WINDSPEED,
        PRESSURE,
        HUMIDITY,
        TEMP
    FROM sf_nwp
    WHERE FARMID = :farmid
    ORDER BY FORETIME ASC
    LIMIT :limit;
    """

    # 查询参数
    params = {
        'farmid': farmid,
        'limit': limit
    }

    print_sql(sql_query, params)

    result = db.query(sql_query, **params)
    res = result.export('df')
    print(f"查询到 {len(res)} 条记录")
    return res



if __name__ == "__main__":
    # 查询参数
    farmid = "1"  # 替换为实际的电站ID
    start_time = "2022-09-09 00:00:00"
    end_time = "2024-09-09 23:45:00"

    # 执行查询
    # df = queryNWPData(farmid)
    df = querySolarDataWithNWP(farmid, start_time, end_time)

    # 检查15分钟间隔的缺失数据
    missing_df = check_missing_dates(df, freq=15, time_column='OBSERVETIME')

    # 打印结果
    print("\n" + "=" * 50)
    print_missing_dates_summary(missing_df)


