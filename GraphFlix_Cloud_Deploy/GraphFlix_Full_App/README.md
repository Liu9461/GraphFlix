# GraphFlix Full App

完整 Streamlit 網頁版，包含：
- 註冊 / 登入 / 登出
- 喜好類型設定
- LightGCN 個人推薦
- 換一批推薦
- 8 大電影分類
- TMDb 電影搜尋
- 收藏與觀看紀錄
- TMDb 海報、評分、上映日期、簡介

## Streamlit Cloud
若沿用目前 GitHub 專案，將這個資料夾中的檔案上傳並覆蓋原本 `GraphFlix_Cloud_Deploy` 內的同名檔案即可。

Main file path：
`GraphFlix_Cloud_Deploy/app.py`

Secrets：
```toml
TMDB_API_KEY = "你的 TMDb API Key"
```

> 注意：免費 Streamlit Community Cloud 的本機檔案系統不是永久資料庫。
> 本版帳號、收藏、觀看紀錄用 SQLite，適合專題展示；雲端環境重建時可能重置。
> 若要正式多人長期使用，下一階段建議把使用者資料改接 Supabase / PostgreSQL。
