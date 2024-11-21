## CCF4DBLP: 根据CCF分级从DBLP中检索论文


该脚本用于从DBLP中，根据CCF期刊和会议分级，基于指定的关键词，检索最近几年相关的论文信息

脚本支持：
1. 根据关键词检索论文：输入一个或多个关键词，脚本会筛选出标题包含这些关键词的论文
2. 检索指定的CCF期刊/会议论文：支持输入一个或多个期刊/会议的缩写、类别和排名等级等属性检索目标论文
3. 时间范围设置：支持指定检索的年份范围，该脚本会抓取指定年份范围内的论文

### 目录结构
```
fetch_papers.py                 # 脚本文件
ccf2022.csv                     # 记录CCF期刊和会议信息
requirements.txt                # 脚本依赖的python库

examples/                       # 两个论文检索示例

    blockchain/                 # 近三年区块链相关顶会顶刊论文检索
        command.sh              # 该示例所使用的运行参数
        fetch_papers.log        # 该示例运行过程的日志信息
        papers_results.xlsx     # 该示例的检索结果

    tee/                        # 近三年TEE相关顶会顶刊论文检索
        command.sh              # 该示例所使用的运行参数
        fetch_papers.log        # 该示例运行过程的日志信息
        papers_results.xlsx     # 该示例的检索结果
        analysis.xlsx           # 近三年TEE相关论文的分析

output/                         # 检索结果的保存位置
    # 运行后生成
    papers_results.xlsx         # 检索结果文件，包含符合要求的论文信息
    fetch_papers.log            # 日志文件，记录检索过程的详细运行信息
```

### 依赖配置

执行前安装依赖
```
pip install -r requirements.txt
```

### 运行说明

#### 命令行参数
运行该脚本时，可以使用以下命令行参数来配置抓取的内容：

- `--keywords`：必填，用逗号分隔的关键词列表，脚本将抓取标题中包含这些关键词的论文（使用下划线代替空格，如：smart_contract）
- `--csv`：指定包含CCF期刊和会议信息的CSV文件路径，默认值为`./ccf2022.csv`，可按照该文件格式自定义检索列表
- `--level`：指定期刊和会议的CCF等级，取值范围为`[A,B,C]`，默认值为`C`，表示筛选CCF-C类及以上的论文
- `--years`：指定要抓取的年份范围，默认为近3年论文。输入一个整数，例如5表示抓取最近5年的论文
- `--journals`：可选，用逗号分隔的期刊或会议的缩写。如果不提供此参数，将抓取所有期刊和会议的数据（使用下划线代替空格，如：USENIX_ATC）
- `--categories`：可选，用逗号分隔的类别列表，如果不提供此参数，将抓取所有类别的数据（1：计算机体系结构/并行与分布计算/存储系统；2：计算机网络；3：网络与信息安全；4：软件工程/系统软件/程序设计语言；5：数据库/数据挖掘/内容检索；6：计算机科学理论；7：计算机图形学与多媒体；8：人工智能；9：人机交互与普适计算；10：交叉/综合/新兴）

#### 运行示例：
```
python fetch_papers.py --keywords blockchain --categories 1,2,3,4,5 --level A
```
运行日志及结果见`/example`目录，运行用时`26min 55s`

#### 结果输出

脚本将检索到的论文保存于`output/papers_results.xlsx`，结果中每个期刊或会议为一个单独的sheet，检索的论文信息包括：
```
Name：期刊或会议的全名
Abbreviation：期刊或会议的缩写
Type：论文类型（会议或期刊）
Year：发表的年份
Keyword：匹配的关键词
Title：论文标题
DOI：论文的DOI链接
```
可根据该结果进一步筛选感兴趣的论文，通过doi下载

PS：根据log也可分析出检索的领域在某些方向的顶会顶刊中是否被广泛关注，见/examples/tee/analysis.xlsx
