#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
# To see the interfaces go to GItHub or :
# ros2 interface show example_interfaces/srv/SetBool 
from example_interfaces.msg import String
from example_interfaces.msg import Float64
from example_interfaces.srv import SetBool

# Don't spend much time in one callback!
# spin() runs all of the code - so all of the callbacks in one thread.
# To achieve long callbacks, implement multithreading.

class PubSubNode(Node):
    def __init__(self):
        super().__init__("pub_sub")
        self.temperature_ = 0.0  # Combining couple of callbacks with the same theme
        self.pub_ = self.create_publisher(String, "some_text", 10)

        self.timer_ = self.create_timer(0.1, self.publish_text)

        self.create_subscription(Float64, "temperature", self.temperature_callback, 10)
        self.create_service(SetBool, "start_robot", self.start_robot_callback)
    
    def publish_text(self):
        msg = String()
        msg.data = "Hello"
        # msg.data = str(self.temperature_)
        self.pub_.publish(msg)
    
    def temperature_callback(self, msg:Float64):
        self.get_logger().info(str(msg.data))
        self.temperature_ = msg.data
        # You can also call one callback from another:
        # self.publish_text()
        # But don't call the callbacks for service or subscriber here,
        # because those are best called when needed in the __init__ function.

    def start_robot_callback(self, request: SetBool.Request, response: SetBool.Response):
        if request.data:
            self.get_logger().info("Success")
        else:
            self.get_logger().info("Error starting the robot.")
        response.success = True
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PubSubNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__=="__main__":
    main()