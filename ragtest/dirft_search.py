# -*- coding: utf-8 -*-

import os
import asyncio
import pandas as pd
import sys
from graphrag.api.query import drift_search
from pathlib import Path
from graphrag.config.load_config import load_config

# 强制设置标准输出的编码
if sys.stdout.encoding != 'gbk':
    sys.stdout.reconfigure(encoding='gbk')

# 获取当前脚本所在的目录
PROJECT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
entities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/entities.parquet")
communities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/communities.parquet")
community_reports = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/community_reports.parquet")
text_units = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/text_units.parquet")
relationships = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/relationships.parquet")
graphrag_config = load_config(Path(PROJECT_DIRECTORY))


async def main(query):
    response, context = await drift_search(
        config=graphrag_config,
        community_level=2,
        entities=entities,
        communities=communities,
        community_reports=community_reports,
        relationships=relationships,
        text_units=text_units,
        covariates=None,
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
