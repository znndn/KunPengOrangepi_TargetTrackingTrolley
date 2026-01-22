import matplotlib.pyplot as plt
import matplotlib


# === 1. 设置中文字体以避免乱码 ===
# 尝试设置常见的中文字体，适配 Windows/Mac/Linux
def set_chinese_font():
    system_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]

    # 常见中文字体列表
    preferred_fonts = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Heiti TC', 'WenQuanYi Micro Hei']

    for font in preferred_fonts:
        if font in system_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            print(f"已启用字体: {font}")
            return

    # 如果没找到常用字体，尝试设置通用回退
    plt.rcParams['font.sans-serif'] = ['sans-serif']
    print("未检测到常用中文字体，使用系统默认字体，中文可能显示为方框。")


set_chinese_font()

# === 2. 准备数据 ===
# 剔除了 "无法识别" 的数据点，仅保留有效区间 (30cm - 90cm)
distances = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
sizes = [
    102971.09, 75986.24, 58318.40, 46007.51, 37296.77,
    31078.25, 25980.07, 22033.98, 19178.52, 16846.03,
    14985.49, 13373.71, 11946.39
]

# === 3. 绘制折线图 ===
plt.figure(figsize=(10, 6), dpi=100)  # 设置画布大小和清晰度

# 绘图：折线 + 数据点标记
plt.plot(distances, sizes, marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='YOLO 算出的面积')

# === 4. 添加图表细节 ===
plt.title('乘算面积和车头与目标距离之间的关系', fontsize=16, pad=20)
plt.xlabel('车头和目标之间的距离 (cm)', fontsize=12)
plt.ylabel('YOLO 算出的面积大小 (Size)', fontsize=12)

# 设置网格，方便读数
plt.grid(True, linestyle='--', alpha=0.6)

# 标记具体的数值点（可选，防止重叠仅每隔一个点显示，或者全部显示）
for x, y in zip(distances, sizes):
    # 在点上方稍微偏移显示数值
    plt.text(x, y + 2000, f'{int(y)}', ha='center', va='bottom', fontsize=9, color='#333333')

# 添加图例
plt.legend()

# 调整布局防止标签被截断
plt.tight_layout()

# === 5. 展示图表 ===
plt.show()