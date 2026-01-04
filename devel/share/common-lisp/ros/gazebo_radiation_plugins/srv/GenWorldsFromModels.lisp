; Auto-generated. Do not edit!


(cl:in-package gazebo_radiation_plugins-srv)


;//! \htmlinclude GenWorldsFromModels-request.msg.html

(cl:defclass <GenWorldsFromModels-request> (roslisp-msg-protocol:ros-message)
  ((folder
    :reader folder
    :initarg :folder
    :type cl:string
    :initform "")
   (model_filename
    :reader model_filename
    :initarg :model_filename
    :type cl:string
    :initform ""))
)

(cl:defclass GenWorldsFromModels-request (<GenWorldsFromModels-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GenWorldsFromModels-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GenWorldsFromModels-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_radiation_plugins-srv:<GenWorldsFromModels-request> is deprecated: use gazebo_radiation_plugins-srv:GenWorldsFromModels-request instead.")))

(cl:ensure-generic-function 'folder-val :lambda-list '(m))
(cl:defmethod folder-val ((m <GenWorldsFromModels-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_radiation_plugins-srv:folder-val is deprecated.  Use gazebo_radiation_plugins-srv:folder instead.")
  (folder m))

(cl:ensure-generic-function 'model_filename-val :lambda-list '(m))
(cl:defmethod model_filename-val ((m <GenWorldsFromModels-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_radiation_plugins-srv:model_filename-val is deprecated.  Use gazebo_radiation_plugins-srv:model_filename instead.")
  (model_filename m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GenWorldsFromModels-request>) ostream)
  "Serializes a message object of type '<GenWorldsFromModels-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'folder))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'folder))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'model_filename))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'model_filename))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GenWorldsFromModels-request>) istream)
  "Deserializes a message object of type '<GenWorldsFromModels-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'folder) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'folder) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'model_filename) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'model_filename) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GenWorldsFromModels-request>)))
  "Returns string type for a service object of type '<GenWorldsFromModels-request>"
  "gazebo_radiation_plugins/GenWorldsFromModelsRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GenWorldsFromModels-request)))
  "Returns string type for a service object of type 'GenWorldsFromModels-request"
  "gazebo_radiation_plugins/GenWorldsFromModelsRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GenWorldsFromModels-request>)))
  "Returns md5sum for a message object of type '<GenWorldsFromModels-request>"
  "aab9442741bfdbb3be9f5e426f1408ca")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GenWorldsFromModels-request)))
  "Returns md5sum for a message object of type 'GenWorldsFromModels-request"
  "aab9442741bfdbb3be9f5e426f1408ca")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GenWorldsFromModels-request>)))
  "Returns full string definition for message of type '<GenWorldsFromModels-request>"
  (cl:format cl:nil "string folder~%string model_filename~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GenWorldsFromModels-request)))
  "Returns full string definition for message of type 'GenWorldsFromModels-request"
  (cl:format cl:nil "string folder~%string model_filename~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GenWorldsFromModels-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'folder))
     4 (cl:length (cl:slot-value msg 'model_filename))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GenWorldsFromModels-request>))
  "Converts a ROS message object to a list"
  (cl:list 'GenWorldsFromModels-request
    (cl:cons ':folder (folder msg))
    (cl:cons ':model_filename (model_filename msg))
))
;//! \htmlinclude GenWorldsFromModels-response.msg.html

(cl:defclass <GenWorldsFromModels-response> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass GenWorldsFromModels-response (<GenWorldsFromModels-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GenWorldsFromModels-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GenWorldsFromModels-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_radiation_plugins-srv:<GenWorldsFromModels-response> is deprecated: use gazebo_radiation_plugins-srv:GenWorldsFromModels-response instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GenWorldsFromModels-response>) ostream)
  "Serializes a message object of type '<GenWorldsFromModels-response>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GenWorldsFromModels-response>) istream)
  "Deserializes a message object of type '<GenWorldsFromModels-response>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GenWorldsFromModels-response>)))
  "Returns string type for a service object of type '<GenWorldsFromModels-response>"
  "gazebo_radiation_plugins/GenWorldsFromModelsResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GenWorldsFromModels-response)))
  "Returns string type for a service object of type 'GenWorldsFromModels-response"
  "gazebo_radiation_plugins/GenWorldsFromModelsResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GenWorldsFromModels-response>)))
  "Returns md5sum for a message object of type '<GenWorldsFromModels-response>"
  "aab9442741bfdbb3be9f5e426f1408ca")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GenWorldsFromModels-response)))
  "Returns md5sum for a message object of type 'GenWorldsFromModels-response"
  "aab9442741bfdbb3be9f5e426f1408ca")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GenWorldsFromModels-response>)))
  "Returns full string definition for message of type '<GenWorldsFromModels-response>"
  (cl:format cl:nil "~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GenWorldsFromModels-response)))
  "Returns full string definition for message of type 'GenWorldsFromModels-response"
  (cl:format cl:nil "~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GenWorldsFromModels-response>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GenWorldsFromModels-response>))
  "Converts a ROS message object to a list"
  (cl:list 'GenWorldsFromModels-response
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'GenWorldsFromModels)))
  'GenWorldsFromModels-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'GenWorldsFromModels)))
  'GenWorldsFromModels-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GenWorldsFromModels)))
  "Returns string type for a service object of type '<GenWorldsFromModels>"
  "gazebo_radiation_plugins/GenWorldsFromModels")