# GraphFlix Cloud

這是可部署到 Streamlit Community Cloud 的 GraphFlix 版本。

## GitHub 要上傳的檔案
- app.py
- requirements.txt
- runtime.txt
- .gitignore
- .env.example
- README.md

## Streamlit Cloud 設定
1. 建立 GitHub repository，將本資料夾的檔案上傳。
2. 在 Streamlit Community Cloud 建立 App。
3. Repository 選剛建立的 repo。
4. Main file path 填 `app.py`。
5. Advanced settings / Secrets 加入：

```toml
TMDB_API_KEY = "你的 TMDb API Key"
```

6. Deploy。

部署完成後會得到 `https://...streamlit.app` 公開網址。

## 注意
- 不要把真正的 TMDb API Key 寫進 app.py、.env.example 或 GitHub。
- 本版會優先讀取 Streamlit Secrets，也支援本機的 `TMDB_API_KEY` 環境變數。
- 為了讓免費雲端第一次訓練比較實際，預設 epochs 設為 30。
- MovieLens 100K 會在第一次啟動時自動下載。
- 免費雲端執行環境可能休眠或重建；若暫存模型消失，需再次按初始化/訓練。
