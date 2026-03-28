# Update & Rollback

1. Виконати `git pull origin main`.
2. Оновити залежності: `pip install -r requirements.txt`.
3. Перезапустити процес застосунку.

### Процедура Rollback (Відкат)
Якщо нова версія нестабільна:
`git checkout [назва_попереднього_тегу]` (наприклад: `git checkout v0.1`)
