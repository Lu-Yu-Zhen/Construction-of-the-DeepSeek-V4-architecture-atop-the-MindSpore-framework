# 贡献指南

感谢您对本项目的关注！我们欢迎社区的贡献。

## 如何贡献

### 报告问题

如果您发现 Bug 或有功能需求，请在 GitHub 上提交 Issue：

1. 检查该问题是否已存在
2. 提供清晰的标题和描述
3. 包含复现步骤（针对 Bug）
4. 说明您的运行环境（操作系统、MindSpore 版本、Python 版本）

### 提交代码

1. **Fork** 本仓库
2. **创建特性分支**：`git checkout -b feature/your-feature-name`
3. **按照编码规范进行修改**
4. **为修改编写测试**
5. **运行测试**：`python -m pytest tests/ -v`
6. **提交**，使用遵循约定式提交规范的清晰信息
7. **推送**到您的 Fork
8. **发起 Pull Request**

### 提交信息规范

我们遵循[约定式提交](https://www.conventionalcommits.org/)规范：

```
<type>(<scope>): <description>

[optional body]
```

类型说明：
- `feat`：新功能
- `fix`：Bug 修复
- `docs`：文档变更
- `style`：代码风格变更（格式化等）
- `refactor`：代码重构
- `test`：添加或更新测试
- `chore`：构建流程或工具变更

示例：
```
feat(attention): 添加可变压缩率支持
fix(moe): 修正专家路由归一化
docs(architecture): 更新 mHC 说明
```

## 开发规范

### 代码风格

- Python 代码遵循 PEP 8
- 函数签名使用类型注解
- 为公开方法编写文档字符串（NumPy 风格）
- 保持函数简洁和模块化
- 最大行长度：100 个字符

### 测试

- 为新组件编写单元测试
- 提交前确保现有测试通过
- 使用 Flash 和 Pro 两种配置进行测试
- 包含边界情况（空输入、极端值）

```bash
# 运行测试
python -m pytest tests/ -v

# 带覆盖率运行
python -m pytest tests/ --cov=src --cov-report=term-missing
```

### 文档

- 更改功能时更新相关文档
- 使用清晰简洁的语言
- 适当包含代码示例
- 提供英文和中文文档（`README_zh.md`）

## 项目结构

```
DeepSeek-V4/
├── src/              # 核心模型实现
├── configs/          # YAML 配置文件
├── scripts/          # 训练和推理脚本
├── tests/            # 单元测试
├── docs/             # 文档
├── data/             # 数据目录
└── checkpoints/      # 模型检查点
```

## 获取帮助

- 在 GitHub 上提交 Issue
- 查阅 [MindSpore 文档](https://www.mindspore.cn/docs)
- 阅读 [DeepSeek-V4 论文](https://arxiv.org/abs/2504.09286)

## 许可证

通过贡献代码，您同意您的贡献将在 MIT 许可证下进行许可。