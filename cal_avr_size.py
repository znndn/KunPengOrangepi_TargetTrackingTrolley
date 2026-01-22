# deepseek写的，用来复制粘贴小车控制台打印的东西，自动算一下size的平均值

import sys
import re


def calculate_average_size():
    """
    从标准输入读取物品检测数据，提取并计算size的平均值
    """
    size_values = []
    pattern = r"size=([\d.]+)"

    # 从标准输入读取所有行
    lines = sys.stdin.readlines()

    for line_num, line in enumerate(lines, 1):
        line = line.strip()

        # 跳过空行
        if not line:
            continue

        # 检查是否是物品检测数据行
        if "物品检测:" not in line:
            print(f"警告: 第{line_num}行格式不正确，跳过: {line}", file=sys.stderr)
            continue

        # 使用正则表达式提取size值
        match = re.search(pattern, line)
        if match:
            try:
                size_value = float(match.group(1))
                size_values.append(size_value)
            except ValueError:
                print(f"警告: 第{line_num}行size值不是有效数字: {line}", file=sys.stderr)
        else:
            print(f"警告: 第{line_num}行未找到size值: {line}", file=sys.stderr)

    # 检查是否提取到了有效的size值
    if not size_values:
        print("错误: 未找到任何有效的size值", file=sys.stderr)
        return None

    # 计算平均值
    average = sum(size_values) / len(size_values)

    # 输出结果
    print(f"size值的数量: {len(size_values)}")
    print(f"size的平均值: {average:.2f}")

    return average


def main():
    """
    主函数，处理命令行参数
    """
    # 检查是否从文件读取
    if len(sys.argv) > 1:
        try:
            with open(sys.argv[1], 'r', encoding='utf-8') as file:
                # 重定向标准输入到文件
                original_stdin = sys.stdin
                sys.stdin = file
                result = calculate_average_size()
                sys.stdin = original_stdin
        except FileNotFoundError:
            print(f"错误: 文件 '{sys.argv[1]}' 未找到", file=sys.stderr)
            return
        except IOError as e:
            print(f"错误: 读取文件时出错: {e}", file=sys.stderr)
            return
    else:
        # 从标准输入读取
        print("请输入物品检测数据 (Ctrl+D 或 Ctrl+Z 结束输入):")
        result = calculate_average_size()

    if result is not None:
        print(f"计算完成，平均值为: {result:.2f}")


if __name__ == "__main__":
    main()