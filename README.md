# RAG Chatbot (Streamlit)

Chatbot ภาษาไทยที่ให้ผู้ใช้อัปโหลดเอกสารเพื่อสร้างฐานความรู้ (RAG) เอง
หากไม่พบข้อมูลในเอกสาร จะตอบด้วย LLM ฟรีที่ไม่ต้องใช้ API key

## รันในเครื่อง
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy บน Streamlit Community Cloud
1. สร้าง GitHub repository แล้วอัปโหลด `app.py`, `requirements.txt`, `.gitignore`, `README.md` ไว้ที่ root
2. เข้า https://share.streamlit.io แล้วเข้าสู่ระบบด้วย GitHub
3. เลือก New app → เลือก repository, branch และ Main file path = `app.py`
4. ใน Advanced settings เลือก Python 3.11 แล้วกด Deploy
