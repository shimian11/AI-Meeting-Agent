# 斑马云面试 · Python Agent 服务（banma-agent-server）
# 基础镜像固定 3.12，与 pyproject.toml 的 requires-python 保持一致（ADR-011：禁止本地/容器版本漂移）
FROM python:3.12-slim

# PYTHONDONTWRITEBYTECODE=1：容器内不生成 .pyc
# PYTHONUNBUFFERED=1：日志实时输出，不被缓冲吞掉
# UV_LINK_MODE=copy：容器内不使用硬链接缓存
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# 只安装 uv；版本请按你本地 `uv --version` 的实际值修改
# -i 清华镜像：构建机上直连 pypi.org 可能被重置（与 apt 的 GPG 坑同源），走镜像更稳
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple "uv==0.12.23"

# 先拷依赖清单再装依赖：依赖不变时这一层能命中缓存
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 业务代码最后拷：改代码不会让上面装依赖的层失效
COPY app ./app

EXPOSE 8100

# --no-sync：镜像里依赖已装好，启动时不再联网校验
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]
