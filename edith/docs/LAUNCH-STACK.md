To run the stack:

```bash
# Backend (terminal 1)
cd edith/backend
source ../../gym-form/.venv/bin/activate
uvicorn main:app --reload --port 8000

# Frontend (terminal 2)
cd edith/frontend
npm run dev

```
