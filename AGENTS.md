# AGENTS.md

## 笔记配图

带图片的笔记要写成一个目录：正文命名为 `index.md`，图片和它放在同一目录下。

```
notes/
└─ Mini-SGLang/
   ├─ index.md        ← 正文，文件名必须是 index.md
   ├─ Image_xxx.png
   └─ Image_yyy.png
```

正文中引用图片时只写文件名，不带任何路径前缀：

```markdown
![架构图](Image_xxx.png)
```

### 错误示例

以下两种写法都会导致图片 404：

- 正文写成 `notes/Mini-SGLang.md`，图片平铺在同一层。
- 图片统一放进 `notes/assets/`，正文用 `./assets/x.png` 引用。

### 原因

Hugo 根据文件名生成页面地址。`Mini-SGLang.md` 对应的页面地址是 `/notes/mini-sglang/`，而平铺的图片会被发布到 `/notes/Image_xxx.png`。页面位于子目录，图片却在上一层，相对路径自然解析不到。

改用 `目录/index.md` 后，整个目录会被当作一篇笔记（页面包），图片随页面一起发布到 `/notes/mini-sglang/` 下，引用就能正常解析。

这一结论经过实测：同一张图，按目录方式组织可以正常访问，平铺则返回 404。

## 站点挂载配置

站点挂载使用 `excludeFiles` 排除不需要的文件，不要改成 `files` 白名单。`files` 的 glob 中 `**` 不会跨目录匹配，子目录（即页面包）会被整体排除，配图随之失效。
