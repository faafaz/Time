from functools import wraps
from datetime import datetime
import records

DB_URL = "mysql+pymysql://root:haosql@localhost:3306/solar_liu"


def db_connector_logger():
    """
    数据库连接和SQL日志装饰器 - 只负责连接数据库和打印SQL
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            db = records.Database(DB_URL)

            try:
                # 调用原函数，传入数据库连接对象
                result = func(db, *args, **kwargs)
                return result

            except Exception as e:
                print(f"函数执行失败: {e}")
                return None

            finally:
                # 关闭数据库连接
                db.close()
                execution_time = (datetime.now() - start_time).total_seconds()
                print(f"数据库连接已关闭 (总耗时: {execution_time:.3f}s)")

        return wrapper

    return decorator


def print_sql(sql, params=None):
    """
    打印SQL语句和参数的工具函数
    """
    print("🚀 === SQL查询信息 ===")
    print("-" * 50)

    if params:
        print("\n📋 查询参数:")
        for key, value in params.items():
            print(f"  :{key} => '{value}'")

        # 生成可直接执行的SQL
        executable_sql = sql
        for key, value in params.items():
            executable_sql = executable_sql.replace(f':{key}', f"'{value}'")

        print("\n🔗 可直接复制执行的SQL:")
        print("-" * 50)
        print(executable_sql)
        print("-" * 50)
    else:
        print("\n📋 无查询参数")
