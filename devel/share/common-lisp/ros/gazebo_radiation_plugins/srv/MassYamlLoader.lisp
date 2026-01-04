; Auto-generated. Do not edit!


(cl:in-package gazebo_radiation_plugins-srv)


;//! \htmlinclude MassYamlLoader-request.msg.html

(cl:defclass <MassYamlLoader-request> (roslisp-msg-protocol:ros-message)
  ((folder
    :reader folder
    :initarg :folder
    :type cl:string
    :initform ""))
)

(cl:defclass MassYamlLoader-request (<MassYamlLoader-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <MassYamlLoader-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'MassYamlLoader-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_radiation_plugins-srv:<MassYamlLoader-request> is deprecated: use gazebo_radiation_plugins-srv:MassYamlLoader-request instead.")))

(cl:ensure-generic-function 'folder-val :lambda-list '(m))
(cl:defmethod folder-val ((m <MassYamlLoader-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_radiation_plugins-srv:folder-val is deprecated.  Use gazebo_radiation_plugins-srv:folder instead.")
  (folder m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <MassYamlLoader-request>) ostream)
  "Serializes a message object of type '<MassYamlLoader-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'folder))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'folder))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <MassYamlLoader-request>) istream)
  "Deserializes a message object of type '<MassYamlLoader-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'folder) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'folder) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<MassYamlLoader-request>)))
  "Returns string type for a service object of type '<MassYamlLoader-request>"
  "gazebo_radiation_plugins/MassYamlLoaderRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'MassYamlLoader-request)))
  "Returns string type for a service object of type 'MassYamlLoader-request"
  "gazebo_radiation_plugins/MassYamlLoaderRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<MassYamlLoader-request>)))
  "Returns md5sum for a message object of type '<MassYamlLoader-request>"
  "bf1b291db59c399c5c5b25859bd3e2e3")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'MassYamlLoader-request)))
  "Returns md5sum for a message object of type 'MassYamlLoader-request"
  "bf1b291db59c399c5c5b25859bd3e2e3")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<MassYamlLoader-request>)))
  "Returns full string definition for message of type '<MassYamlLoader-request>"
  (cl:format cl:nil "string folder~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'MassYamlLoader-request)))
  "Returns full string definition for message of type 'MassYamlLoader-request"
  (cl:format cl:nil "string folder~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <MassYamlLoader-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'folder))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <MassYamlLoader-request>))
  "Converts a ROS message object to a list"
  (cl:list 'MassYamlLoader-request
    (cl:cons ':folder (folder msg))
))
;//! \htmlinclude MassYamlLoader-response.msg.html

(cl:defclass <MassYamlLoader-response> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass MassYamlLoader-response (<MassYamlLoader-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <MassYamlLoader-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'MassYamlLoader-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_radiation_plugins-srv:<MassYamlLoader-response> is deprecated: use gazebo_radiation_plugins-srv:MassYamlLoader-response instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <MassYamlLoader-response>) ostream)
  "Serializes a message object of type '<MassYamlLoader-response>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <MassYamlLoader-response>) istream)
  "Deserializes a message object of type '<MassYamlLoader-response>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<MassYamlLoader-response>)))
  "Returns string type for a service object of type '<MassYamlLoader-response>"
  "gazebo_radiation_plugins/MassYamlLoaderResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'MassYamlLoader-response)))
  "Returns string type for a service object of type 'MassYamlLoader-response"
  "gazebo_radiation_plugins/MassYamlLoaderResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<MassYamlLoader-response>)))
  "Returns md5sum for a message object of type '<MassYamlLoader-response>"
  "bf1b291db59c399c5c5b25859bd3e2e3")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'MassYamlLoader-response)))
  "Returns md5sum for a message object of type 'MassYamlLoader-response"
  "bf1b291db59c399c5c5b25859bd3e2e3")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<MassYamlLoader-response>)))
  "Returns full string definition for message of type '<MassYamlLoader-response>"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'MassYamlLoader-response)))
  "Returns full string definition for message of type 'MassYamlLoader-response"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <MassYamlLoader-response>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <MassYamlLoader-response>))
  "Converts a ROS message object to a list"
  (cl:list 'MassYamlLoader-response
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'MassYamlLoader)))
  'MassYamlLoader-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'MassYamlLoader)))
  'MassYamlLoader-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'MassYamlLoader)))
  "Returns string type for a service object of type '<MassYamlLoader>"
  "gazebo_radiation_plugins/MassYamlLoader")