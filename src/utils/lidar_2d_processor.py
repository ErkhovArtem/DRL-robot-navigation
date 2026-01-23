#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray


class Lidar2DProcessor(Node):
    """
    ROS2 нода для обработки облака точек лидара:
    1. Фильтрует точки на уровне пола и проецирует их в 2D плоскость
    2. Создает массив векторов (секторная визуализация препятствий)
    """

    def __init__(self):
        super().__init__('lidar_2d_processor')

        # Параметры для фильтрации точек пола
        self.declare_parameter('floor_height_threshold', 1.2)  # м от пола
        
        # Параметры для секторной визуализации
        self.declare_parameter('min_range', 0.25)  # минимальная дальность (исключаем робота)
        self.declare_parameter('max_range', 3.0)  # максимальная дальность в метрах (соответствует обучению)
        self.declare_parameter('num_sectors', 40)  # количество секторов по кругу
        self.declare_parameter('obstacle_threshold', 0.37)  # порог детекции препятствий (метры)
        self.declare_parameter('lidar_noise_std', 0.0)  # СКО гауссовского шума (метры)
        
        # Параметры топиков
        self.declare_parameter('lidar_topic', '/livox/lidar')  # Топик от livox_ros_driver2
        self.declare_parameter('floor_points_topic', '/floor_points_2d')
        self.declare_parameter('obstacles_sectors_topic', '/obstacles_sectors')
        self.declare_parameter('lidar_observations_topic', '/lidar_observations')  # Топик для политики
        
        # Получаем параметры
        self.floor_height_threshold = float(self.get_parameter('floor_height_threshold').value)
        self.min_range = float(self.get_parameter('min_range').value)
        self.max_range = float(self.get_parameter('max_range').value)
        self.num_sectors = int(self.get_parameter('num_sectors').value)
        self.obstacle_threshold = float(self.get_parameter('obstacle_threshold').value)
        self.lidar_noise_std = float(self.get_parameter('lidar_noise_std').value)
        
        lidar_topic = str(self.get_parameter('lidar_topic').value)
        floor_points_topic = str(self.get_parameter('floor_points_topic').value)
        obstacles_sectors_topic = str(self.get_parameter('obstacles_sectors_topic').value)
        lidar_observations_topic = str(self.get_parameter('lidar_observations_topic').value)

        # Инициализация массивов для секторной визуализации
        self.sector_min_distances = np.full(self.num_sectors, self.max_range)
        self._debug_counter = 0

        # Определяем поля для PointCloud2
        self.pc_fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # QoS профиль для подписки (livox может использовать BEST_EFFORT)
        sub_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )
        
        # QoS профиль для публикации (rviz требует RELIABLE)
        pub_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        # Подписка на облако точек лидара
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            lidar_topic,
            self.lidar_callback,
            sub_qos
        )

        # Публикаторы
        self.floor_points_pub = self.create_publisher(
            PointCloud2,
            floor_points_topic,
            pub_qos
        )
        
        self.obstacles_sectors_pub = self.create_publisher(
            MarkerArray,
            obstacles_sectors_topic,
            pub_qos
        )
        
        # Публикатор для данных лидара для политики (40 значений в метрах)
        self.lidar_observations_pub = self.create_publisher(
            Float32MultiArray,
            lidar_observations_topic,
            pub_qos
        )

        self.get_logger().info(
            f'Lidar2DProcessor initialized:\n'
            f'  Subscribed to: {lidar_topic}\n'
            f'  Publishing floor points to: {floor_points_topic}\n'
            f'  Publishing obstacles sectors to: {obstacles_sectors_topic}\n'
            f'  Publishing lidar observations to: {lidar_observations_topic}\n'
            f'  Parameters: floor_threshold={self.floor_height_threshold}m, '
            f'range=[{self.min_range}, {self.max_range}]m, '
            f'sectors={self.num_sectors}, obstacle_threshold={self.obstacle_threshold}m'
        )

    def create_floor_points_2d(self, points):
        """
        Фильтрует точки на уровне пола и проецирует их в 2D (обнуляет Z).
        
        Args:
            points: numpy array shape (N, 3) с координатами точек [x, y, z]
            
        Returns:
            numpy array shape (M, 3) с 2D точками пола [x, y, 0] или None
        """
        if len(points) == 0:
            return None
        
        # Находим минимальную Z координату (самые нижние точки - это пол)
        min_z = points[:, 2].min()
        
        # Фильтруем точки близко к минимальной высоте
        # Эти точки находятся на полу или близко к нему
        floor_mask = (points[:, 2] - min_z) < self.floor_height_threshold
        floor_points = points[floor_mask].copy()
        
        # Отладочная информация (выводим раз в 50 кадров)
        self._debug_counter += 1
        if self._debug_counter % 50 == 0:
            self.get_logger().info(
                f"[Floor 2D] Min Z: {min_z:.2f}m, "
                f"Total points: {len(points)}, Floor points: {len(floor_points)}, "
                f"Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]"
            )
        
        if len(floor_points) == 0:
            return None
        
        # Проецируем точки в 2D (обнуляем Z координату)
        floor_points[:, 2] = 0.0
        
        return floor_points

    def create_obstacle_sectors(self, points, time_stamp, frame_id):
        """
        Создает MarkerArray с визуализацией препятствий по секторам.
        Также обновляет self.sector_min_distances для публикации в lidar_observations.
        
        ВАЖНО: Логика полностью совместима с process_lidar_to_sectors из train.py.
        Используется та же обработка: фильтрация по min_range (0.25 м),
        обрезка до max_range (3.0 м), деление на 40 секторов.
        
        Args:
            points: numpy array shape (N, 3) с 2D точками [x, y, z]
            time_stamp: ROS2 Time message
            frame_id: строка с именем фрейма для маркеров
            
        Returns:
            MarkerArray или None
        """
        if len(points) == 0:
            # Если нет точек, устанавливаем все секторы на max_range
            self.sector_min_distances = np.full(self.num_sectors, self.max_range, dtype=np.float32)
            return None
        
        # Фильтруем точки в радиусе от min_range до max_range
        # ВАЖНО: Обрабатываем точно как process_lidar_to_sectors в train.py
        # Сначала обрезаем до max_range (как np.clip), затем фильтруем по min_range
        distances_2d = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        
        # Обрабатываем как в train.py: заменяем inf на max_range, обрезаем до [0, max_range]
        distances_2d = np.where(np.isinf(distances_2d), self.max_range, distances_2d)
        distances_2d = np.clip(distances_2d, 0, self.max_range)
        
        # Фильтруем точки слишком близко (как valid_mask в train.py)
        range_mask = distances_2d >= self.min_range
        filtered_points = points[range_mask]
        filtered_distances = distances_2d[range_mask]
        
        # Добавляем гауссовский шум если настроен
        if self.lidar_noise_std > 0:
            noise = np.random.normal(0, self.lidar_noise_std, size=filtered_distances.shape).astype(np.float32)
            filtered_distances = filtered_distances + noise
            filtered_distances = np.clip(filtered_distances, 0.0, self.max_range)
        
        # Вычисляем углы для каждой точки
        angles = np.arctan2(filtered_points[:, 1], filtered_points[:, 0])
        
        # Сбрасываем массив дистанций (инициализируем максимальной дальностью)
        self.sector_min_distances = np.full(self.num_sectors, self.max_range, dtype=np.float32)
        
        # Создаем массив маркеров
        marker_array = MarkerArray()
        sector_angle = 2 * np.pi / self.num_sectors
        
        for sector_idx in range(self.num_sectors):
            # Определяем границы сектора (точно как в process_lidar_to_sectors)
            angle_min = -np.pi + sector_idx * sector_angle
            angle_max = angle_min + sector_angle
            angle_center = (angle_min + angle_max) / 2
            
            # Находим точки в этом секторе
            # Для последнего сектора (39) включаем правую границу π
            if sector_idx == self.num_sectors - 1:
                # Последний сектор: [π-Δ, π] (включает π)
                sector_mask = (angles >= angle_min) & (angles <= angle_max)
            else:
                # Обычные секторы: [angle_min, angle_max)
                sector_mask = (angles >= angle_min) & (angles < angle_max)
            
            sector_distances = filtered_distances[sector_mask]
            
            if len(sector_distances) == 0:
                # Если нет препятствий, показываем максимальную дальность (зеленый)
                min_dist = self.max_range
                color = (0.0, 1.0, 0.0, 0.3)  # зеленый, полупрозрачный
            else:
                # Находим ближайшее препятствие
                min_dist = float(sector_distances.min())
                # Цвет от зеленого (далеко) до красного (близко)
                ratio = min_dist / self.max_range
                color = (1.0 - ratio, ratio, 0.0, 0.7)  # от красного к зеленому
            
            # Сохраняем минимальную дистанцию для этого сектора
            self.sector_min_distances[sector_idx] = min_dist
            
            # Создаем маркер - стрелку от центра
            marker = Marker()
            marker.header.frame_id = frame_id  # Используем переданный frame_id
            marker.header.stamp = time_stamp
            marker.ns = "obstacle_sectors"
            marker.id = sector_idx
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            # Начальная точка (центр)
            marker.points = []
            start_point = Point()
            start_point.x = 0.0
            start_point.y = 0.0
            start_point.z = 0.0
            
            # Конечная точка (на расстоянии min_dist в направлении angle_center)
            end_point = Point()
            end_point.x = float(min_dist * np.cos(angle_center))
            end_point.y = float(min_dist * np.sin(angle_center))
            end_point.z = 0.0
            
            marker.points.append(start_point)
            marker.points.append(end_point)
            
            # Размеры стрелки
            marker.scale.x = 0.05  # толщина стрелки
            marker.scale.y = 0.1   # ширина головки
            marker.scale.z = 0.1   # высота головки
            
            # Цвет
            marker.color.r = float(color[0])
            marker.color.g = float(color[1])
            marker.color.b = float(color[2])
            marker.color.a = float(color[3])
            
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 200000000  # 0.2 секунды
            
            marker_array.markers.append(marker)
        
        return marker_array

    def lidar_callback(self, msg: PointCloud2):
        """
        Обработчик сообщений облака точек лидара.
        
        Args:
            msg: PointCloud2 сообщение
        """
        try:
            # Читаем точки из сообщения
            # read_points возвращает итератор кортежей (x, y, z, ...)
            points_gen = point_cloud2.read_points(
                msg, field_names=('x', 'y', 'z'), skip_nans=True
            )
            
            # Преобразуем в numpy array более эффективным способом
            # Используем list comprehension для извлечения координат
            points_list = [[p[0], p[1], p[2]] for p in points_gen]
            
            if len(points_list) == 0:
                self.get_logger().warn('Received empty point cloud')
                return
            
            # Преобразуем в numpy array
            points = np.array(points_list, dtype=np.float32)
            
            # Проверяем валидность данных
            if not np.isfinite(points).all():
                self.get_logger().warn('Point cloud contains non-finite values, filtering...')
                valid_mask = np.isfinite(points).all(axis=1)
                points = points[valid_mask]
                if len(points) == 0:
                    return
            
            # Создаем 2D точки пола
            floor_points_2d = self.create_floor_points_2d(points)
            
            if floor_points_2d is not None and len(floor_points_2d) > 0:
                # Проверяем, что есть ненулевые точки
                non_zero_mask = np.any(floor_points_2d[:, :2] != 0, axis=1)
                if non_zero_mask.sum() == 0:
                    self.get_logger().debug('All floor points are zero, skipping floor points publication')
                else:
                    # Публикуем 2D облако точек пола
                    floor_pc_msg = PointCloud2()
                    floor_pc_msg.header.stamp = msg.header.stamp
                    floor_pc_msg.header.frame_id = msg.header.frame_id
                    floor_pc_msg.fields = self.pc_fields
                    floor_pc_msg.is_bigendian = False
                    floor_pc_msg.point_step = 12  # 3 * 4 bytes (float32)
                    floor_pc_msg.height = 1
                    floor_pc_msg.is_dense = True
                    floor_pc_msg.row_step = 12 * floor_points_2d.shape[0]
                    floor_pc_msg.width = floor_points_2d.shape[0]
                    floor_pc_msg.data = floor_points_2d.tobytes()
                    
                    self.floor_points_pub.publish(floor_pc_msg)
                
                # Создаем и публикуем секторную визуализацию препятствий
                obstacle_markers = self.create_obstacle_sectors(
                    floor_points_2d, 
                    msg.header.stamp,
                    msg.header.frame_id  # Передаем frame_id из сообщения
                )
                if obstacle_markers is not None:
                    # Публикуем визуализацию препятствий
                    self.obstacles_sectors_pub.publish(obstacle_markers)
            else:
                # Если нет точек пола, устанавливаем все секторы на max_range
                self.sector_min_distances = np.full(self.num_sectors, self.max_range, dtype=np.float32)
            
            # ВСЕГДА публикуем данные лидара для политики (40 значений в метрах, не нормализованные)
            # Формат точно соответствует process_lidar_to_sectors из train.py
            # Если нет препятствий, все значения будут max_range (3.0 м)
            lidar_msg = Float32MultiArray()
            lidar_msg.data = self.sector_min_distances.tolist()
            self.lidar_observations_pub.publish(lidar_msg)
                        
        except Exception as e:
            import traceback
            self.get_logger().error(
                f'Ошибка при обработке облака точек: {e}\n{traceback.format_exc()}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = Lidar2DProcessor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

