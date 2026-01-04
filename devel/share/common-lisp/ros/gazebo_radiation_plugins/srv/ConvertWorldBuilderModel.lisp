; Auto-generated. Do not edit!


(cl:in-package gazebo_radiation_plugins-srv)


;//! \htmlinclude ConvertWorldBuilderModel-request.msg.html

(cl:defclass <ConvertWorldBuilderModel-request> (roslisp-msg-protocol:ros-message)
  ((world_builder_file
    :reader world_builder_file
    :initarg :world_builder_file
    :type cl:string
    :initform "")
   (gazebo_model_path
    :reader gazebo_model_path
    :initarg :gazebo_model_path
    :type cl:string
    :initform "")
   (noise_file
    :reader noise_file
    :initarg :noise_file
    :type cl:string
    :initform ""))
)

(cl:defclass ConvertWorldBuilderModel-request (<ConvertWorldBuilderModel-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ConvertWorldBuilderModel-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ConvertWorldBuilderModel-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_radiation_plugins-srv:<ConvertWorldBuilderModel-request> is deprecated: use gazebo_radiation_plugins-srv:ConvertWorldBuilderModel-request instead.")))

(cl:ensure-generic-function 'world_builder_file-val :lambda-list '(m))
(cl:defmethod world_builder_file-val ((m <ConvertWorldBuilderModel-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_radiation_plugins-srv:world_builder_file-val is deprecated.  Use gazebo_radiation_plugins-srv:world_builder_file instead.")
  (world_builder_file m))

(cl:ensure-generic-function 'gazebo_model_path-val :lambda-list '(m))
(cl:defmethod gazebo_model_path-val ((m <ConvertWorldBuilderModel-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_radiation_plugins-srv:gazebo_model_path-val is deprecated.  Use gazebo_radiation_plugins-srv:gazebo_model_path instead.")
  (gazebo_model_path m))

(cl:ensure-generic-function 'noise_file-val :lambda-list '(m))
(cl:defmethod noise_file-val ((m <ConvertWorldBuilderModel-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_radiation_plugins-srv:noise_file-val is deprecated.  Use gazebo_radiation_plugins-srv:noise_file instead.")
  (noise_file m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ConvertWorldBuilderModel-request>) ostream)
  "Serializes a message object of type '<ConvertWorldBuilderModel-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'world_builder_file))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'world_builder_file))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'gazebo_model_path))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'gazebo_model_path))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'noise_file))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'noise_file))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ConvertWorldBuilderModel-request>) istream)
  "Deserializes a message object of type '<ConvertWorldBuilderModel-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'world_builder_file) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'world_builder_file) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'gazebo_model_path) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'gazebo_model_path) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'noise_file) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'noise_file) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ConvertWorldBuilderModel-request>)))
  "Returns string type for a service object of type '<ConvertWorldBuilderModel-request>"
  "gazebo_radiation_plugins/ConvertWorldBuilderModelRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ConvertWorldBuilderModel-request)))
  "Returns string type for a service object of type 'ConvertWorldBuilderModel-request"
  "gazebo_radiation_plugins/ConvertWorldBuilderModelRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ConvertWorldBuilderModel-request>)))
  "Returns md5sum for a message object of type '<ConvertWorldBuilderModel-request>"
  "ca5346b14c3e8fe8c1488efbdb6aac5b")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ConvertWorldBuilderModel-request)))
  "Returns md5sum for a message object of type 'ConvertWorldBuilderModel-request"
  "ca5346b14c3e8fe8c1488efbdb6aac5b")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ConvertWorldBuilderModel-request>)))
  "Returns full string definition for message of type '<ConvertWorldBuilderModel-request>"
  (cl:format cl:nil "string world_builder_file~%string gazebo_model_path~%string noise_file~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ConvertWorldBuilderModel-request)))
  "Returns full string definition for message of type 'ConvertWorldBuilderModel-request"
  (cl:format cl:nil "string world_builder_file~%string gazebo_model_path~%string noise_file~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ConvertWorldBuilderModel-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'world_builder_file))
     4 (cl:length (cl:slot-value msg 'gazebo_model_path))
     4 (cl:length (cl:slot-value msg 'noise_file))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ConvertWorldBuilderModel-request>))
  "Converts a ROS message object to a list"
  (cl:list 'ConvertWorldBuilderModel-request
    (cl:cons ':world_builder_file (world_builder_file msg))
    (cl:cons ':gazebo_model_path (gazebo_model_path msg))
    (cl:cons ':noise_file (noise_file msg))
))
;//! \htmlinclude ConvertWorldBuilderModel-response.msg.html

(cl:defclass <ConvertWorldBuilderModel-response> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass ConvertWorldBuilderModel-response (<ConvertWorldBuilderModel-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ConvertWorldBuilderModel-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ConvertWorldBuilderModel-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_radiation_plugins-srv:<ConvertWorldBuilderModel-response> is deprecated: use gazebo_radiation_plugins-srv:ConvertWorldBuilderModel-response instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ConvertWorldBuilderModel-response>) ostream)
  "Serializes a message object of type '<ConvertWorldBuilderModel-response>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ConvertWorldBuilderModel-response>) istream)
  "Deserializes a message object of type '<ConvertWorldBuilderModel-response>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ConvertWorldBuilderModel-response>)))
  "Returns string type for a service object of type '<ConvertWorldBuilderModel-response>"
  "gazebo_radiation_plugins/ConvertWorldBuilderModelResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ConvertWorldBuilderModel-response)))
  "Returns string type for a service object of type 'ConvertWorldBuilderModel-response"
  "gazebo_radiation_plugins/ConvertWorldBuilderModelResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ConvertWorldBuilderModel-response>)))
  "Returns md5sum for a message object of type '<ConvertWorldBuilderModel-response>"
  "ca5346b14c3e8fe8c1488efbdb6aac5b")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ConvertWorldBuilderModel-response)))
  "Returns md5sum for a message object of type 'ConvertWorldBuilderModel-response"
  "ca5346b14c3e8fe8c1488efbdb6aac5b")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ConvertWorldBuilderModel-response>)))
  "Returns full string definition for message of type '<ConvertWorldBuilderModel-response>"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ConvertWorldBuilderModel-response)))
  "Returns full string definition for message of type 'ConvertWorldBuilderModel-response"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ConvertWorldBuilderModel-response>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ConvertWorldBuilderModel-response>))
  "Converts a ROS message object to a list"
  (cl:list 'ConvertWorldBuilderModel-response
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'ConvertWorldBuilderModel)))
  'ConvertWorldBuilderModel-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'ConvertWorldBuilderModel)))
  'ConvertWorldBuilderModel-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ConvertWorldBuilderModel)))
  "Returns string type for a service object of type '<ConvertWorldBuilderModel>"
  "gazebo_radiation_plugins/ConvertWorldBuilderModel")