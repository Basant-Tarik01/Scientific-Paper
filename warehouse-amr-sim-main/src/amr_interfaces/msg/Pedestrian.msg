int32 id
bool active                    # false = parked outside the building (not part of the scenario)
geometry_msgs/Pose pose        # map frame, z = 0
geometry_msgs/Twist twist      # linear.x/y = world-frame velocity
float32 radius
bool aware                     # false = distracted worker (does not avoid the robot)
bool dwelling                  # standing still (picking / packing)
