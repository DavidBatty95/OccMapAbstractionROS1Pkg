;; Auto-generated. Do not edit!


(when (boundp 'gazebo_radiation_plugins::Simulated_Radiation_Msg)
  (if (not (find-package "GAZEBO_RADIATION_PLUGINS"))
    (make-package "GAZEBO_RADIATION_PLUGINS"))
  (shadow 'Simulated_Radiation_Msg (find-package "GAZEBO_RADIATION_PLUGINS")))
(unless (find-package "GAZEBO_RADIATION_PLUGINS::SIMULATED_RADIATION_MSG")
  (make-package "GAZEBO_RADIATION_PLUGINS::SIMULATED_RADIATION_MSG"))

(in-package "ROS")
;;//! \htmlinclude Simulated_Radiation_Msg.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass gazebo_radiation_plugins::Simulated_Radiation_Msg
  :super ros::object
  :slots (_header _value _pose _type ))

(defmethod gazebo_radiation_plugins::Simulated_Radiation_Msg
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:value __value) 0.0)
    ((:pose __pose) (instance geometry_msgs::Pose :init))
    ((:type __type) "")
    )
   (send-super :init)
   (setq _header __header)
   (setq _value (float __value))
   (setq _pose __pose)
   (setq _type (string __type))
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:value
   (&optional __value)
   (if __value (setq _value __value)) _value)
  (:pose
   (&rest __pose)
   (if (keywordp (car __pose))
       (send* _pose __pose)
     (progn
       (if __pose (setq _pose (car __pose)))
       _pose)))
  (:type
   (&optional __type)
   (if __type (setq _type __type)) _type)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; float32 _value
    4
    ;; geometry_msgs/Pose _pose
    (send _pose :serialization-length)
    ;; string _type
    4 (length _type)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; float32 _value
       (sys::poke _value (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; geometry_msgs/Pose _pose
       (send _pose :serialize s)
     ;; string _type
       (write-long (length _type) s) (princ _type s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; float32 _value
     (setq _value (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; geometry_msgs/Pose _pose
     (send _pose :deserialize buf ptr-) (incf ptr- (send _pose :serialization-length))
   ;; string _type
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _type (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(setf (get gazebo_radiation_plugins::Simulated_Radiation_Msg :md5sum-) "7dee165b32bb3e3f6a1f812a165a3eff")
(setf (get gazebo_radiation_plugins::Simulated_Radiation_Msg :datatype-) "gazebo_radiation_plugins/Simulated_Radiation_Msg")
(setf (get gazebo_radiation_plugins::Simulated_Radiation_Msg :definition-)
      "Header header
float32 value
geometry_msgs/Pose pose
string type
================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w

")



(provide :gazebo_radiation_plugins/Simulated_Radiation_Msg "7dee165b32bb3e3f6a1f812a165a3eff")


