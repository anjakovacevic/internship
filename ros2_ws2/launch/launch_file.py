from launch import LaunchDescription           
from launch_ros.actions import Node            

def generate_launch_description():             
    return LaunchDescription([                 
        Node(                                 
            package='system',         
            executable='image_raw', 
        ),
        Node(                               
            package='system',        
            executable='all_landmarks',
        ),
        Node(                               
            package='system',        
            executable='get_eyes',
        ),
        Node(                               
            package='system',        
            executable='face_det',
        ),
        Node(
           package='system',
           executable='gaze_capture',
        )
       
    ])
