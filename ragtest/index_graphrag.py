import os
import asyncio
from pathlib import Path
from graphrag.config.load_config import load_config
from graphrag.api.index import build_index
from graphrag.index.typing.pipeline_run_result import PipelineRunResult


async def main():
    # 获取当前脚本所在的目录
    PROJECT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

    # 加载配置
    graphrag_config = load_config(Path(PROJECT_DIRECTORY))
    print(graphrag_config)

    # 构建索引
    index_result: list[PipelineRunResult] = await build_index(config=graphrag_config)

    # 输出结果
    for workflow_result in index_result:
        status = f"error\n{workflow_result.errors}" if workflow_result.errors else "success"
        print(f"Workflow Name: {workflow_result.workflow}\tstatus: {status}")


# 运行异步主函数
if __name__ == "__main__":
    asyncio.run(main())
