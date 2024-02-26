# ROS2 u Kompjuterskoj viziji

## Uvod
Ovaj direktorijum koristi ROS2 kako bi pokrenuo razlicite delove procesa nezavisno. Realizovan je gaze capture po uzoru na kod iz gaze capture foldera ovog repozitorijuma, takodje i drowsiness detection i user recognition.

## Namestanje okruzenja

Uputstvo instalacije ROS2 moze se pronaci u ros2_basics folderu ovog repozitorijuma.

Napraviti folder u kom ce se nalaziti projekat.
```python
mkdir ros2_ws
cd ros2_ws
mkdir src
colcon build
cd src
ros2 pkg create system --build-type ament_python --dependencies rclpy
```
Kada su svi fajlovi generisani, kopirati sadrzaj iz ovog repozitorijuma:

- ros2_ws/launch;
- ros2_ws/src/custom_interfaces;
- ros2_ws/src/system/system;
- ispraviti generisane package.xml i setup.py, tako da izgledaju kao prilozeni u repozitorijumu.

```python
cd ros2_ws
colcon build --symlink-install
```

## Pokretanje koda

Kako bi se pokrenuli svi node-ovi, a zatim i interaktivni prozor, pratiti ove korake.
```python
cd ros2_ws
colcon build --symlink-install

cd launch
ros2 launch launch_file.py

# u novom terminalu:
cd ros2_ws/src/system/system
python3 gui.py
```

