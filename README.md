# ComfyPicShow

BS 架构的 ComfyUI 图片/视频浏览器。通过浏览器浏览服务端目录下的图片和视频，支持 ComfyUI 元数据提取、提示词标签显示与搜索、收藏、自定义列表、备注等功能。

## 功能

- **目录浏览** — 面包屑导航，缩略图网格展示图片和视频
- **弹窗详情** — 点击缩略图弹窗查看大图/播放视频，支持键盘 ← → 切换上/下一张
- **ComfyUI 元数据** — 自动解析 PNG/视频中的 prompt/workflow，多采样以 TAB 切换显示
- **提示词标签** — 正向/反向提示词拆分为标签，点击搜索、一键复制
- **全文搜索** — 按文件名或提示词内容搜索图片和视频
- **视频播放** — 自动静音播放，点击切换暂停/播放，静音/取消静音
- **收藏** — 收藏图片/视频，收藏页支持缩略图/列表/大图三种展示模式
- **自定义列表** — 创建多个列表归类图片/视频，支持拖拽排序
- **备注** — 每张图片/视频可添加备注，大字体显示
- **排序** — 按文件名或修改时间正序/逆序排列
- **无限滚动** — 首页和搜索结果页自动加载下一页
- **GPU 加速** — 视频缩略图生成自动检测 NVIDIA/VAAPI/QSV 硬件加速
- **清理工具** — 一键清理已删除文件的缓存数据

## 安装

```bash
git clone https://github.com/yourname/comfypicshow.git
cd comfypicshow
pip install -r requirements.txt
python app.py
```

浏览器访问 `http://localhost:5000`，将图片/视频放入 `images/` 目录即可。

### 依赖

- Python 3.10+
- Flask
- Pillow
- ffmpeg（用于视频缩略图生成和元数据提取）

## Docker 部署

```bash
# 使用 docker-compose
IMAGES_DIR=/path/to/your/images PORT=8080 docker-compose up -d

# 或直接 docker run
docker run -d -p 5000:5000 \
  -v /path/to/images:/images \
  -v ./data:/data \
  --device /dev/dri:/dev/dri \
  comfypicshow
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PORT` | `5000` | 服务端口 |
| `IMAGES_DIR` | `/app/images` | 图片/视频根目录 |
| `DATA_DIR` | `/app` | 缓存和配置数据目录 |
| `THUMBNAIL_SIZE` | `300` | 缩略图最大尺寸（像素） |

### 数据持久化

`/data` 目录包含所有需要持久化的数据：

```
/data/cache/
├── thumbnails/          # 缩略图缓存
├── file_index.json      # 文件索引缓存
├── metadata_cache.json  # 元数据缓存
├── favorites.json       # 收藏
├── notes.json           # 备注
└── lists.json           # 自定义列表
```

## 配置

编辑 `config.py` 或通过环境变量设置：

```python
IMAGE_ROOT_DIR = "./images"  # 图片根目录
PORT = 5000                   # 服务端口
THUMBNAIL_SIZE = 300          # 缩略图最大尺寸
```

## 许可

MIT
