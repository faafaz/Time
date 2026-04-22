import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_china_highlight_static(file_path):
    try:
        # 1. 读取 GeoJSON 文件 (确保是全国数据)
        gdf = gpd.read_file(file_path, encoding='utf-8')

        # 2. 设置绘图参数
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal') # 确保地图不会变形
        ax.set_facecolor('#AEEEEE') # 设置海洋背景色(可选)

        # ================== 核心绘制逻辑 ==================

        # 3. 绘制底图 (所有省份)
        # zorder=1 确保它在最底层
        gdf.plot(ax=ax, edgecolor='white', facecolor='lightgrey', linewidth=0.8, zorder=1)

        # 4. 筛选并高亮指定区域 (湖北和重庆)
        # 使用 str.contains 进行模糊匹配，因为数据中可能是 "湖北省" 或 "重庆市"
        highlight_names = ['湖北', '重庆']
        pattern = '|'.join(highlight_names) # 生成正则表达式 '湖北|重庆'
        
        highlight_gdf = gdf[gdf['name'].str.contains(pattern, na=False)]

        if not highlight_gdf.empty:
            # zorder=2 确保它绘制在底图之上
            highlight_gdf.plot(ax=ax, edgecolor='black', facecolor='#FF4500', linewidth=1.2, zorder=2, alpha=0.8)
            print(f"已高亮区域: {highlight_gdf['name'].tolist()}")
        else:
            print("警告: 未在数据中找到'湖北'或'重庆'，请检查 GeoJSON 的 name 字段。")

        # ================== 添加指北针 ==================

        # 5. 手动添加指北针
        # xycoords='axes fraction' 表示使用相对坐标 (0,0 是左下, 1,1 是右上)
        # 我们放在右上角 (0.92, 0.92) 的位置
        # ax.annotate('N', xy=(0.92, 0.95), xytext=(0.92, 0.85),
        #             arrowprops=dict(facecolor='black', width=2, headwidth=5, headlength=5),
        #             ha='center', va='center', fontsize=9,
        #             xycoords='axes fraction', textcoords='axes fraction', zorder=5)


        # 6. 添加标题和其他修饰
        # plt.title("中国地图 - 高亮湖北与重庆", fontsize=18, pad=20)
        plt.grid(True, linestyle='--', alpha=0.3, zorder=0)
        # plt.xlabel('经度')
        # plt.ylabel('纬度')

        # 创建自定义图例
        # highlight_patch = mpatches.Patch(color='#FF4500', label='重点关注区域 (湖北/重庆)')
        # base_patch = mpatches.Patch(color='lightgrey', label='其他省份')
        # plt.legend(handles=[highlight_patch, base_patch], loc='lower left')

        plt.tight_layout()
        # 保存图片
        plt.savefig('china_highlight_static.png', dpi=300)
        plt.show()
        
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 【重要】请确保这里替换成了你包含全国数据的 geojson 文件路径
    # 如果你用之前的北京数据运行，代码不会报错，但也就没有东西可以高亮了。
    plot_china_highlight_static("C:/Users/Administrator/Downloads/中华人民共和国.geojson")