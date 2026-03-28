# Deployment Guide (Production)

### Апаратні вимоги
- CPU: 4+ ядра (для стабільного FPS).
- RAM: 8 GB.
- Disk: 5 GB вільного місця.

### Кроки
1. Оновити системні пакети: `sudo apt update`.
2. Встановити Python та `libgl1` (потрібно для OpenCV).
3. Клонувати репозиторій.
4. Налаштувати автозапуск через **Systemd** або **Docker**.
