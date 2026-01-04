; Auto-generated. Do not edit!


(cl:in-package gazebo_radiation_plugins-msg)


;//! \htmlinclude Simulated_Radiation_Msg.msg.html

(cl:defclass <Simulated_Radiation_Msg> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (value
    :reader value
    :initarg :value
    :type cl:float
    :initform 0.0)
   (pose
    :reader pose
    :initarg :pose
    :type geometry_msgs-msg:Pose
    :initform (cl:make-instance 'geometry_msgs-msg:Pose))
   (type
    :reader type
    :initarg :type
    :type cl:string
    :initform ""))
)

(cl:defclass Simulated_Radiation_Msg (<Simulated_Radiation_Msg>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Simulated_Radiation_Msg>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Simulated_Radiation_Msg)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_radiation_plugins-msg:<Simulated_Radiation_Msg> is deprecated: use gazebo_radiation_plugins-msg:Simulated_Radiation_Msg instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <Simulated_Radiation_Msg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_radiation_plugins-msg:header-val is deprecated.  Use gazebo_radiation_plugins-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'value-val :lambda-list '(m))
(cl:defmethod value-val ((m <Simulated_Radiation_Msg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_radiation_plugins-msg:value-val is deprecated.  Use gazebo_radiation_plugins-msg:value instead.")
  (value m))

(cl:ensure-generic-function 'pose-val :lambda-list '(m))
(cl:defmethod pose-val ((m <Simulated_Radiation_Msg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_radiation_plugins-msg:pose-val is deprecated.  Use gazebo_radiation_plugins-msg:pose instead.")
  (pose m))

(cl:ensure-generic-function 'type-val :lambda-list '(m))
(cl:defmethod type-val ((m <Simulated_Radiation_Msg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_radiation_plugins-msg:type-val is deprecated.  Use gazebo_radiation_plugins-msg:type instead.")
  (type m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Simulated_Radiation_Msg>) ostream)
  "Serializes a message object of type '<Simulated_Radiation_Msg>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'value))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'pose) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'type))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'type))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Simulated_Radiation_Msg>) istream)
  "Deserializes a message object of type '<Simulated_Radiation_Msg>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'value) (roslisp-utils:decode-single-float-bits bits)))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'pose) istream)
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'type) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'type) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Simulated_Radiation_Msg>)))
  "Returns string type for a message object of type '<Simulated_Radiation_Msg>"
  "gazebo_radiation_plugins/Simulated_Radiation_Msg")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Simulated_Radiation_Msg)))
  "Returns string type for a message object of type 'Simulated_Radiation_Msg"
  "gazebo_radiation_plugins/Simulated_Radiation_Msg")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Simulated_Radiation_Msg>)))
  "Returns md5sum for a message object of type '<Simulated_Radiation_Msg>"
  "7dee165b32bb3e3f6a1f812a165a3eff")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Simulated_Radiation_Msg)))
  "Returns md5sum for a message object of type 'Simulated_Radiation_Msg"
  "7dee165b32bb3e3f6a1f812a165a3eff")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Simulated_Radiation_Msg>)))
  "Returns full string definition for message of type '<Simulated_Radiation_Msg>"
  (cl:format cl:nil "Header header~%float32 value~%geometry_msgs/Pose pose~%string type~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Simulated_Radiation_Msg)))
  "Returns full string definition for message of type 'Simulated_Radiation_Msg"
  (cl:format cl:nil "Header header~%float32 value~%geometry_msgs/Pose pose~%string type~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Simulated_Radiation_Msg>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'pose))
     4 (cl:length (cl:slot-value msg 'type))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Simulated_Radiation_Msg>))
  "Converts a ROS message object to a list"
  (cl:list 'Simulated_Radiation_Msg
    (cl:cons ':header (header msg))
    (cl:cons ':value (value msg))
    (cl:cons ':pose (pose msg))
    (cl:cons ':type (type msg))
))
