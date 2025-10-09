import subprocess
import os


def run_graphrag_init():
    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建命令
    command = ["graphrag", "init", "--root", current_dir]

    try:
        # 执行命令
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("命令执行成功！")
        if result.stderr:
            print("错误输出:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败，错误码: {e.returncode}")
        print("错误输出:", e.stderr)
    except FileNotFoundError:
        print("错误: 未找到 graphrag 命令，请确保已正确安装 GraphRAG")


# 如果这个脚本放在 ragtest 文件夹中，直接运行
if __name__ == "__main__":
    run_graphrag_init()
