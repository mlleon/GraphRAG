# -*- coding: utf-8 -*-
import os
import asyncio
import pandas as pd
from graphrag.api.query import global_search
from pathlib import Path
from graphrag.config.load_config import load_config

# 设置 pandas 显示选项，确保所有行和列都显示
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 不限制宽度
pd.set_option('display.max_colwidth', None)  # 不限制列宽

# 获取当前脚本所在的目录
PROJECT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
entities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/entities.parquet")
communities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/communities.parquet")
community_reports = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/community_reports.parquet")
graphrag_config = load_config(Path(PROJECT_DIRECTORY))


async def main(query):
    response, context = await global_search(
        config=graphrag_config,
        community_level=2,
        entities=entities,
        communities=communities,
        community_reports=community_reports,
        dynamic_community_selection=False,
        response_type="Multiple Paragraphs",
        query=query,
        verbose=True
    )
    print("=== 响应 ===")
    print(response)
    print("\n=== 上下文 ===")
    print(context)


if __name__ == "__main__":
    asyncio.run(main(query="这篇文档写的怎么样"))
