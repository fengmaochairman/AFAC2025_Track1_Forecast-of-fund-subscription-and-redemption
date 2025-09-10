# AFAC2025_track1 - 基金产品的长周期申购和赎回预测

【队伍：CEDC冲冲冲】

## 简介

本项目是AFAC 2025赛题一：基金产品的长周期申购和赎回预测 Top2 解决方案。本方案通过构建两个独立的LightGBM模型（静态多目标模型与滚动单目标模型），结合大模型特征、时序特征工程及异常值处理策略，实现基金未来7天申购和赎回量的预测，并通过对两个模型的预测结果取平均得到最终输出。  

赛题链接🔗：https://tianchi.aliyun.com/competition/entrance/532352/information

## 项目结构
AFAC2025_track1/  
├── README.md                      # 项目说明文件  
├── data/                          # 数据目录  
│   └── 20250724_update/           # 最新数据  
│       ├── fund_apply_redeem_series.csv    
├── main.py                        # 主程序（14个LightGBM模型）  
├── main_roll.py                   # 主程序-滚动预测 （2个LightGBM模型）  
├── integrate_result.py            # 结果融合  
├── tool_feature_server.py         # 基础特征处理服务  
├── tool_feature_client.py         # 基础特征处理客户端  
├── mcp_fetch_client.py            # MCP数据抓取客户端  
└── servers_config.json            # MCP服务配置（fetch服务）  


## 配置说明
1、创建新的 conda 环境：
```bash
conda create -n track1 python=3.10
conda activate track1
```
2、安装依赖
```bash
pip install -r requirements.txt
```
3、MCP服务配置
使用modelscope的fetch服务，复制fetch服务的url，替换server_config.json中的url。

fetch服务链接🔗：https://www.modelscope.cn/mcp/servers/@modelcontextprotocol/fetch

```json
{
    "mcpServers": {
        "fetch": {
            "type": "sse",
            "url": "https://mcp.api-inference.modelscope.net/xxxx/sse"
        }
    }
}
```
4、Qwen模型配置
api_key = "sk-xxxx"

## 使用方法
1、启动基础特征处理服务
```bash
python tool_feature_server.py
```
2、运行主程序
再打开一个终端，运行主程序
```bash
# 激活环境
conda activate track1
# 静态多目标模型
python main.py
# 滚动单目标模型
python main_roll.py
```
3、结果融合
```bash
python integrate_result.py
```