# kasegu

波動率策略研究專案，使用 [uv](https://docs.astral.sh/uv/) 管理。

## 環境需求

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Python 3.13（uv 會依 `.python-version` 自動安裝）

## 快速開始

```bash
uv sync                 # 建立 .venv 並安裝相依套件（含 dev 群組）
uv run kasegu           # 執行 CLI 進入點
uv run pytest           # 執行測試
uv run ruff check .     # Lint
uv run ruff format .    # 格式化
```

## 新增套件

```bash
uv add numpy pandas     # 執行期相依
uv add --dev ipykernel  # 開發用相依
```

## 專案結構

```
.
├── pyproject.toml      # 專案設定與相依
├── uv.lock             # 鎖定的相依版本（請一併提交）
├── src/kasegu/         # 套件原始碼
└── tests/              # 測試
```
