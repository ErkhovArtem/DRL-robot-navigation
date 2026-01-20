#!/usr/bin/env python3
"""
Простой скрипт для запуска сцены с роботом Unitree A1.
Робот просто отображается в визуализаторе, никаких действий не выполняется.
"""

import mujoco
import mujoco.viewer
import os

def main():
    # Путь к файлу сцены
    scene_path = os.path.join(os.path.dirname(__file__), "scene.xml")
    
    if not os.path.exists(scene_path):
        print(f"Ошибка: файл сцены не найден: {scene_path}")
        return
    
    print(f"Загрузка сцены: {scene_path}")
    
    # Загружаем модель
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    
    # Сбрасываем в начальное состояние (keyframe "home")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    print("Сцена загружена успешно!")
    print("Робот находится в начальной позе (keyframe 'home')")
    print("Закройте окно визуализатора для выхода")
    
    # Запускаем визуализатор
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Просто держим окно открытым, не применяем никаких управлений
        while viewer.is_running():
            # Шаг симуляции (робот будет падать под действием гравитации, если не зафиксирован)
            mujoco.mj_step(model, data)
            viewer.sync()
    
    print("Визуализация завершена")

if __name__ == "__main__":
    main()
