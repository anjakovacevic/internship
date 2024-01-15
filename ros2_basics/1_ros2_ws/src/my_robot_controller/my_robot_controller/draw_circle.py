#?/usr/bin/env python3
import rclpy
from rclpy.node import Node
# ros2 topic list
# ros2 topic info /turtle1/cmd_vel
from geometry_msgs.msg import Twist

class DrawCircleNode(Node):
    def __init__(self):
        super().__init__("draw_circle")
        # Creating a publisher (10 is queue size - buffer for msg sending succesfully)
        self.cmd_vel_pub = self.create_publisher(Twist, "/turtle1/cmd_vel", 10)
        self.timer_ = self.create_timer(0.5, self.send_velocity_command)
        self.get_logger().info("Darw circle node has beed started.")
    
    def send_velocity_command(self):
        # ros2 interface show geometry_msgs/msg/Twist
        msg = Twist()
        msg.linear.x = 2.0
        msg.angular.z = 1.0
        self.cmd_vel_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DrawCircleNode()
    rclpy.spin(node)
    rclpy.shutdown()